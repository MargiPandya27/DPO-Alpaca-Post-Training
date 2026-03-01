import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    # TRL <=0.11 exposed OnlineDPO at top-level
    from trl import OnlineDPOTrainer, OnlineDPOConfig
except Exception:
    # Newer TRL moved OnlineDPO under experimental
    from trl.experimental.online_dpo import OnlineDPOTrainer, OnlineDPOConfig

try:
    from trl.trainer.judges import BasePairwiseJudge
except Exception:
    from trl.experimental.judges import BasePairwiseJudge
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from config.load import load_config

cfg = load_config()
models_cfg = cfg["models"]
dpo_cfg = cfg["dpo_online"]

# ── 1. 4-bit QLoRA config ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # nf4 is best for QLoRA
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # double quantization saves more memory
)

# ── 2. Load dataset ───────────────────────────────────────────────────
dataset = load_dataset(models_cfg["alpaca_dataset"], split="train").select(range(dpo_cfg["num_examples"]))
dataset = dataset.train_test_split(test_size=0.1, seed=42)

def format_dataset(example):
    return {"prompt": example["instruction"]}

train_dataset = dataset["train"].map(format_dataset)
eval_dataset = dataset["test"].map(format_dataset)

# ── 3. Config ────────────────────────────────────────────────────────
BASE_MODEL_ID = models_cfg["base_model_id"]
SFT_MODEL_PATH = models_cfg["sft_model_path"]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ── 4. Policy model (QLoRA) ───────────────────────────────────────────
# Step 1: Load base model in 4-bit
policy_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Step 2: Prepare for kbit training BEFORE adding LoRA
policy_base = prepare_model_for_kbit_training(
    policy_base,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# Step 3: Load existing DPO adapter as trainable
policy_model = PeftModel.from_pretrained(
    policy_base,
    models_cfg["dpo_adapter"],
    is_trainable=True,
)

# Optional: add NEW LoRA adapter on top if you want fresh QLoRA layers
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# policy_model = get_peft_model(policy_base, lora_config)

policy_model.print_trainable_parameters()

# ── 5. Reference model (frozen SFT) ──────────────────────────────────
ref_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
ref_model = PeftModel.from_pretrained(
    ref_base,
    SFT_MODEL_PATH,
    is_trainable=False,  # ← frozen
)
for param in ref_model.parameters():
    param.requires_grad = False

# ── 6. Custom Judge ───────────────────────────────────────────────────
class CustomJudge(BasePairwiseJudge):
    def __init__(self, model_id=None):
        model_id = model_id or dpo_cfg["judge_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def judge(self, prompts, completions, shuffle_order=True):
        ranks = []
        for prompt, (completion_a, completion_b) in zip(prompts, completions):
            input_text = f"""Which response is better? Reply ONLY with 'A' or 'B'.

Prompt: {prompt}
Response A: {completion_a}
Response B: {completion_b}

Answer:"""
            inputs = self.tokenizer(
                input_text, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                )
            verdict = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            ranks.append(0 if "A" in verdict[-5:] else 1)
        return ranks

judge = CustomJudge()

# ── 7. QLoRA-compatible Config ────────────────────────────────────────
config = OnlineDPOConfig(
    output_dir=dpo_cfg["output_dir"],
    num_train_epochs=dpo_cfg["num_train_epochs"],
    per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
    learning_rate=dpo_cfg["learning_rate"],
    beta=dpo_cfg["beta"],
    max_length=dpo_cfg["max_length"],
    max_new_tokens=dpo_cfg["max_new_tokens"],
    bf16=False,
    fp16=False,
    logging_steps=dpo_cfg["logging_steps"],
    eval_strategy="no",
    save_steps=dpo_cfg["save_steps"],
    save_total_limit=dpo_cfg["save_total_limit"],
    load_best_model_at_end=False,
    report_to="wandb",
    run_name=dpo_cfg["run_name"],
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
)

# ── 8. Trainer ────────────────────────────────────────────────────────
trainer = OnlineDPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    judge=judge,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# ── 9. Train ──────────────────────────────────────────────────────────
trainer.train()
trainer.save_model(dpo_cfg["save_dir"])
tokenizer.save_pretrained(dpo_cfg["save_dir"])
print("QLoRA Online DPO training complete!")

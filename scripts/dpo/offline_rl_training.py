import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from config.load import load_config
from peft import PeftModel

cfg = load_config()
models_cfg = cfg["models"]
dpo_cfg = cfg["dpo_offline"]

MODEL_PATH = models_cfg["sft_model_path"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# DPO requires left padding for proper completion generation/eval
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# TRL DPO appends BOS/EOS during tokenization; if token ids are None -> collator crash (see trl#1073, Qwen)
_eos_id = getattr(tokenizer, "eod_id", None) or tokenizer.eos_token_id or tokenizer.pad_token_id or 0
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = _eos_id
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.decode([tokenizer.eos_token_id]) if tokenizer.eos_token_id else "<|endoftext|>"
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.bos_token_id = tokenizer.eos_token_id
if tokenizer.bos_token_id is None:
    tokenizer.bos_token_id = tokenizer.eos_token_id

BASE_MODEL_ID = models_cfg["base_model_id"]
is_peft_adapter = Path(MODEL_PATH).joinpath("adapter_config.json").exists()

if is_peft_adapter:
    # SFT checkpoint is LoRA adapter: load base then adapter (trainable)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH, is_trainable=True)
    ref_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model = PeftModel.from_pretrained(ref_base, MODEL_PATH, is_trainable=False)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

for p in ref_model.parameters():
    p.requires_grad = False

model.train()
assert any(p.requires_grad for p in model.parameters()), (
    "Model has no trainable parameters. Check that the checkpoint is a full model or PEFT adapter with is_trainable=True."
)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_trainable:,}")

# ── 2. Load & Pre-process Dataset ────────────────────────────────────
dataset = load_dataset(models_cfg["preference_dataset"])
raw_data = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]

def format_and_filter(example):
    # Guard against None or missing fields
    prompt = example.get("instruction")
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    if prompt is None or chosen is None or rejected is None:
        return None
    prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
    chosen = chosen.strip() if isinstance(chosen, str) else str(chosen)
    rejected = rejected.strip() if isinstance(rejected, str) else str(rejected)
    if not prompt or len(chosen) < 5 or len(rejected) < 5:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

# Map (returns None for bad rows so they are removed) then drop any remaining bad rows
formatted_dataset = raw_data.map(format_and_filter, desc="Format preference pairs")
def is_valid_row(x):
    p, c, r = x.get("prompt"), x.get("chosen"), x.get("rejected")
    if not p or not c or not r:
        return False
    p, c, r = (p or "").strip(), (c or "").strip(), (r or "").strip()
    return len(p) > 0 and len(c) >= 5 and len(r) >= 5

formatted_dataset = formatted_dataset.filter(is_valid_row, desc="Drop invalid rows")

# Keep only columns DPO expects (extra columns can cause None in collator)
cols_to_keep = ["prompt", "chosen", "rejected"]
cols_to_drop = [c for c in formatted_dataset.column_names if c not in cols_to_keep]
if cols_to_drop:
    formatted_dataset = formatted_dataset.remove_columns(cols_to_drop)

# Split the dataset
dataset_splits = formatted_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_splits["train"]
eval_dataset = dataset_splits["test"]

# ── 3. DPO Config ────────────────────────────────────────────────────
dpo_config = DPOConfig(
    output_dir=dpo_cfg["output_dir"],
    num_train_epochs=dpo_cfg["num_train_epochs"],
    per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
    learning_rate=dpo_cfg["learning_rate"],
    lr_scheduler_type=dpo_cfg["lr_scheduler_type"],
    warmup_steps=dpo_cfg["warmup_steps"],
    beta=dpo_cfg["beta"],
    max_length=dpo_cfg["max_length"],
    max_prompt_length=dpo_cfg["max_prompt_length"],
    bf16=dpo_cfg["bf16"],
    logging_steps=dpo_cfg["logging_steps"],
    eval_strategy="steps",
    eval_steps=dpo_cfg["eval_steps"],
    save_strategy="steps",
    save_steps=dpo_cfg["save_steps"],
    save_total_limit=dpo_cfg["save_total_limit"],
    load_best_model_at_end=True,
    report_to="wandb",
    run_name=dpo_cfg["run_name"],
)

# ── 4. Initialize Trainer ────────────────────────────────────────────
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# ── 5. Train & Save ──────────────────────────────────────────────────
print("Starting DPO Training...")
trainer.train()

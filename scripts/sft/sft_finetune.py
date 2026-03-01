import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from transformers.optimization import SchedulerType as LRScheduler
from peft import LoraConfig
from config.load import load_config

cfg = load_config()
models_cfg = cfg["models"]
sft_cfg = cfg["sft"]

# 1. Load Dataset
dataset = load_dataset(models_cfg["alpaca_dataset"], split="train")
dataset = dataset.shuffle(seed=42).select(range(sft_cfg["num_examples"]))
dataset_splits = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_splits["train"]
eval_dataset = dataset_splits["test"]

# 2. Prepare Model (4-bit for efficiency)
model_id = models_cfg["base_model_id"]
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto"
)

# 3. Define Formatting Function (Alpaca -> Qwen ChatML)
def format_instruction(sample):
    # Combine instruction and input
    user_prompt = sample['instruction']
    if sample['input']:
        user_prompt += f"\n\nInput: {sample['input']}"

    # Create conversation list
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": sample['output']}
    ]
    # Apply Qwen's specific template
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Map the dataset
train_dataset = train_dataset.map(format_instruction)
eval_dataset = eval_dataset.map(format_instruction)

# 4. Configure LoRA & Training
lora_cfg = sft_cfg["lora"]
peft_config = LoraConfig(
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["lora_alpha"],
    target_modules=lora_cfg["target_modules"],
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir=sft_cfg["output_dir"],
    per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
    learning_rate=sft_cfg["learning_rate"],
    lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "linear"),
    warmup_steps=sft_cfg.get("warmup_steps", 0),
    max_seq_length=sft_cfg.get("max_length", 512),
    bf16=sft_cfg.get("bf16", False),
    num_train_epochs=sft_cfg["num_train_epochs"],
    eval_strategy="steps",
    logging_steps=sft_cfg["logging_steps"],
    eval_steps=sft_cfg.get("eval_steps", 100),
    dataset_text_field="text",
    packing=True,
    report_to="wandb",
    run_name=sft_cfg["run_name"],
)

# 5. Launch Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()

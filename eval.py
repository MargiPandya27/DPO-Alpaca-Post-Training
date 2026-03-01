"""Evaluate model using BERTScore against reference outputs on held-out Alpaca."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from bert_score import score as bertscore_score

from config.load import load_config


cfg = load_config()
print(cfg.keys())
models_cfg = cfg["models"]
eval_cfg = cfg["eval"]
sft_cfg = cfg["sft"]

MODEL_PATH = eval_cfg["model_path"]
BASE_MODEL_ID = models_cfg["base_model_id"]
NUM_EVAL = eval_cfg["num_examples"]
MAX_NEW_TOKENS = eval_cfg["max_new_tokens"]
BATCH_SIZE = eval_cfg["batch_size"]
SEED = eval_cfg["seed"]

# Load model
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

adapter_config = Path(MODEL_PATH) / "adapter_config.json"
if adapter_config.exists():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, MODEL_PATH, is_trainable=False)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

model.eval()

# Load held-out eval data (different from training split)
dataset = load_dataset(models_cfg["alpaca_dataset"], split="train")
dataset = dataset.shuffle(seed=SEED)
# Skip training indices to get true held-out
train_size = sft_cfg["num_examples"]
eval_dataset = dataset.select(range(train_size, min(train_size + NUM_EVAL, len(dataset))))
if len(eval_dataset) == 0:
    eval_dataset = dataset.shuffle(seed=SEED + 1).select(range(NUM_EVAL))

def build_prompt(example):
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n\nInput: {example['input']}"
    return user_content

candidates = []
references = []
prompts = [build_prompt(eval_dataset[i]) for i in range(len(eval_dataset))]

print(f"Generating on {len(prompts)} examples...")
for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i : i + BATCH_SIZE]
    messages_batch = [[{"role": "user", "content": p}] for p in batch_prompts]
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            temperature=eval_cfg.get("temperature", 0.7),
        )

    for j, out in enumerate(outputs):
        gen = tokenizer.decode(out[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        candidates.append(gen.strip())
        references.append(eval_dataset[i + j]["output"].strip())

print("Computing BERTScore...")
P, R, F1 = bertscore_score(
    candidates,
    references,
    lang="en",
    verbose=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

print(f"\nBERTScore on {len(candidates)} examples:")
print(f"  Precision: {P.mean().item():.4f}")
print(f"  Recall:    {R.mean().item():.4f}")
print(f"  F1:        {F1.mean().item():.4f}")

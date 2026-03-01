import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import warnings
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

from distilabel.steps.tasks import TextGeneration
from distilabel.models.llms import TransformersLLM
from datasets import load_dataset, Dataset
from tqdm import tqdm
from config.load import load_config

import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()

cfg = load_config()
models_cfg = cfg["models"]
pref_cfg = cfg["preference"]

# Generator LLM
generator_llm = TransformersLLM(
    model=pref_cfg["generator_model"],
    generation_kwargs={"max_new_tokens": pref_cfg["max_new_tokens"], "temperature": pref_cfg["generator_temp"]},
    device="cuda",
    disable_cuda_device_placement=True,
)

judge_llm = TransformersLLM(
    model=pref_cfg["judge_model"],
    generation_kwargs={"max_new_tokens": 64, "temperature": pref_cfg["judge_temp"]},
    device="cuda",
    disable_cuda_device_placement=True,
    trust_remote_code=True,
)

# Load data
dataset = load_dataset(models_cfg["alpaca_dataset"], split="train").select(range(pref_cfg["num_examples"]))
inputs = [{"instruction": row["instruction"]} for row in dataset]

BATCH_SIZE = pref_cfg["generate_batch_size"]

def batch_generate(llm, inputs, batch_size, desc="Generating"):
    results = []
    for i in tqdm(range(0, len(inputs), batch_size), desc=desc):
        batch = inputs[i:i+batch_size]
        gen = TextGeneration(name=f"gen_{i}", llm=llm, num_generations=1)
        gen.load()
        batch_results = list(next(gen.process(batch)))
        results.extend(batch_results)
    return results

# Step 1: Generate two responses in batches
print("Generating responses (temp=0.7)...")
responses1 = batch_generate(generator_llm, inputs, BATCH_SIZE, desc="Generation 1")

generator_llm.generation_kwargs["temperature"] = pref_cfg["generator_temp_alt"]
print("Generating responses (temp=1.2)...")
responses2 = batch_generate(generator_llm, inputs, BATCH_SIZE, desc="Generation 2")



# Step 2: Load judge
judge_llm.load()

def judge_batch(instructions, responses_a, responses_b, batch_size=64):
    """Judge multiple pairs in batches"""
    verdicts = []

    prompts = []
    for instruction, response_a, response_b in zip(instructions, responses_a, responses_b):
        prompt = f"""You are an expert judge. Given an instruction and two responses, decide which is better.

Instruction: {instruction}

Response A: {response_a}

Response B: {response_b}

Which response is better? Reply with ONLY 'A' or 'B'."""
        prompts.append([{"role": "user", "content": prompt}])

    for i in tqdm(range(0, len(prompts), batch_size), desc="Judging"):
        batch = prompts[i:i+batch_size]
        results = judge_llm.generate(inputs=batch, num_generations=1)
        for result in results:
            verdict = result["generations"][0] if result and result["generations"] else "A"
            verdicts.append("A" if "A" in verdict else "B")

    return verdicts

# Step 3: Judge all pairs in batches
instructions = [r["instruction"] for r in responses1]
gens_a = [r["generation"] for r in responses1]
gens_b = [r["generation"] for r in responses2]

verdicts = judge_batch(instructions, gens_a, gens_b, batch_size=pref_cfg["judge_batch_size"])

# Step 4: Build preference dataset
preference_data = []
for instruction, gen_a, gen_b, verdict in zip(instructions, gens_a, gens_b, verdicts):
    chosen, rejected = (gen_a, gen_b) if verdict == "A" else (gen_b, gen_a)
    preference_data.append({
        "instruction": instruction,
        "chosen": chosen,
        "rejected": rejected,
        "generations": [gen_a, gen_b],
        "judge_verdict": verdict,
    })

final_dataset = Dataset.from_list(preference_data)
print(final_dataset[0])
print(f"\nTotal examples: {len(final_dataset)}")
final_dataset.save_to_disk(pref_cfg["output_dir"])

# Push to HuggingFace Hub (set HF_TOKEN env var)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    final_dataset.push_to_hub(
        pref_cfg["hub_repo"],
        token=hf_token,
        private=False,
    )
else:
    print("Skipping push to Hub: HF_TOKEN not set. To push, run: export HF_TOKEN=hf_your_token")

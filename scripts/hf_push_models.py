"""
Push the three pipeline models to Hugging Face Hub (MERGED models).
1. Empties existing repos 
2. Merges PEFT/LoRA adapters into full models
3. Pushes merged models for vLLM compatibility

Requires: pip install huggingface_hub peft transformers torch; huggingface-cli login
Usage: python scripts/push_models_to_hub.py
"""
import os
import sys
from pathlib import Path
import shutil
import torch
from typing import Optional

# Project root (parent of scripts/) 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.load import load_config
from huggingface_hub import HfApi, delete_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

cfg = load_config()
models_cfg = cfg["models"]
sft_cfg = cfg["sft"]
dpo_off_cfg = cfg["dpo_offline"]
dpo_on_cfg = cfg["dpo_online"]

# Repo IDs on the Hub
HF_USER = os.environ.get("HF_USER", "MargiPandya27")
SFT_REPO = os.environ.get("SFT_HUB_REPO", f"{HF_USER}/qwen1.5-1.8b-sft-alpaca")
DPO_OFFLINE_REPO = os.environ.get("DPO_OFFLINE_HUB_REPO", f"{HF_USER}/qwen1.5-1.8b-dpo-offline-alpaca")
DPO_ONLINE_REPO = os.environ.get("DPO_ONLINE_HUB_REPO", f"{HF_USER}/qwen1.5-1.8b-dpo-online-alpaca")

# Base model (same for all)
BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"  # Adjust if different

# Local paths (from config)
SFT_PATH = sft_cfg["output_dir"]
DPO_OFFLINE_PATH = dpo_off_cfg["output_dir"]
DPO_ONLINE_PATH = dpo_on_cfg["save_dir"]

def merge_peft_model(
    peft_path: str, 
    base_model: str, 
    output_path: str, 
    checkpoint_num: Optional[int] = None
) -> str:
    """Merge PEFT/LoRA adapter with base model"""
    print(f"Merging PEFT model from {peft_path}...")
    
    # Find checkpoint if specified
    if checkpoint_num:
        peft_path = Path(peft_path) / f"checkpoint-{checkpoint_num:03d}"
        if not peft_path.exists():
            raise FileNotFoundError(f"Checkpoint {peft_path} not found")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Load PEFT model
    model = PeftModel.from_pretrained(model, peft_path)
    
    # Merge and unload
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Merged model saved to {output_path}")
    return output_path

def empty_and_push(local_path: str, repo_id: str, token: str):
    """Empty repo and push new model"""
    api = HfApi(token=token)
    
    print(f"🔄 Emptying repo {repo_id}...")
    try:
        # Delete and recreate repo (empties it completely)
        delete_repo(repo_id, repo_type="model", repo_private=False)
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        print(f"✅ Repo {repo_id} emptied and recreated")
    except Exception as e:
        print(f"⚠️ Could not empty repo (might not exist): {e}")
    
    # Upload merged model
    print(f"📤 Uploading {local_path} -> {repo_id}...")
    upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload merged model (vLLM compatible)"
    )
    print(f"✅ Pushed: https://huggingface.co/{repo_id}")

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Set HF_TOKEN (huggingface-cli login or export HF_TOKEN=...)")
        return
    
    api = HfApi(token=token)
    
    # 1) SFT MODEL (usually already merged)
    print("\n" + "="*60)
    print("1. SFT MODEL")
    sft_path = os.environ.get("SFT_PUSH_PATH", SFT_PATH)
    if Path(sft_path).exists():
        empty_and_push(sft_path, SFT_REPO, token)
    else:
        print(f"❌ SFT path not found: {sft_path}")
    
    # 2) DPO OFFLINE (merge best checkpoint)
    print("\n" + "="*60)
    print("2. DPO OFFLINE")
    dpo_off_temp = "./temp_merged_dpo_offline"
    shutil.rmtree(dpo_off_temp, ignore_errors=True)
    try:
        merge_peft_model(
            DPO_OFFLINE_PATH, 
            BASE_MODEL, 
            dpo_off_temp,
            checkpoint_num=112  # Your checkpoint-15
        )
        empty_and_push(dpo_off_temp, DPO_OFFLINE_REPO, token)
    finally:
        shutil.rmtree(dpo_off_temp, ignore_errors=True)
    
    # 3) DPO ONLINE (merge final checkpoint or latest)
    print("\n" + "="*60)
    print("3. DPO ONLINE")
    dpo_on_temp = "./temp_merged_dpo_online"
    shutil.rmtree(dpo_on_temp, ignore_errors=True)
    try:
        merge_peft_model(DPO_ONLINE_PATH, 
                        BASE_MODEL, 
                        dpo_on_temp)
        empty_and_push(dpo_on_temp, DPO_ONLINE_REPO, token)
    finally:
        shutil.rmtree(dpo_on_temp, ignore_errors=True)
    
    print("\n🎉 ALL MODELS PUSHED (vLLM compatible)!")
    print("\nRepos:")
    print(f"  SFT:        https://huggingface.co/{SFT_REPO}")
    print(f"  DPO Offline: https://huggingface.co/{DPO_OFFLINE_REPO}")
    print(f"  DPO Online:  https://huggingface.co/{DPO_ONLINE_REPO}")
    print("\nTest with vLLM:")
    print(f"  llm = LLM(model='{DPO_OFFLINE_REPO}')")

if __name__ == "__main__":
    main()

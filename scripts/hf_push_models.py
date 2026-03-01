"""
Push the three pipeline models to Hugging Face Hub (model repos).
Requires: pip install huggingface_hub; then login: huggingface-cli login
Usage:
  python scripts/push_models_to_hub.py
  # Or with env vars:
  HF_USER=your_username python scripts/push_models_to_hub.py
"""
import os
import sys
from pathlib import Path

# Project root (parent of scripts/) so that "config" can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.load import load_config

cfg = load_config()
models_cfg = cfg["models"]
sft_cfg = cfg["sft"]
dpo_off_cfg = cfg["dpo_offline"]
dpo_on_cfg = cfg["dpo_online"]

# Repo IDs on the Hub (set HF_USER or override each repo)
HF_USER = os.environ.get("HF_USER", "YOUR_USERNAME")
SFT_REPO = os.environ.get("SFT_HUB_REPO", f"{HF_USER}/qwen1.5-1.8b-sft-alpaca")
DPO_OFFLINE_REPO = os.environ.get("DPO_OFFLINE_HUB_REPO", f"{HF_USER}/qwen1.5-1.8b-dpo-offline-alpaca")
DPO_ONLINE_REPO = os.environ.get("DPO_ONLINE_HUB_REPO", f"{HF_USER}/qwen1.5-1.8b-dpo-online-alpaca")

# Local paths (from config)
SFT_PATH = sft_cfg["output_dir"]           # e.g. ./qwen-sft-alpaca (or a checkpoint subdir)
DPO_OFFLINE_PATH = dpo_off_cfg["output_dir"]  # e.g. ./dpo-final-run
DPO_ONLINE_PATH = dpo_on_cfg["save_dir"]      # e.g. ./online-dpo-qlora-final


def push_folder_to_hub(local_path: str, repo_id: str, token: str = None):
    from huggingface_hub import HfApi
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    path = Path(local_path)
    if not path.exists():
        print(f"Skipping {local_path} (not found)")
        return
    # Create repo if it doesn't exist (avoids 404/401 confusion when repo is missing)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Uploading {path} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  -> https://huggingface.co/{repo_id}")


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN (huggingface-cli login or export HF_TOKEN=...)")
        return
    # If you see '401 Unauthorized' or 'token is expired', run: huggingface-cli login

    # 1) SFT: push the chosen checkpoint (e.g. last or best)
    # If you use a specific checkpoint, set SFT_PATH to e.g. ./qwen-sft-alpaca/checkpoint-30
    sft_path = os.environ.get("SFT_PUSH_PATH", SFT_PATH)
    push_folder_to_hub(sft_path, SFT_REPO, token)

    # 2) DPO offline: push the run folder (contains checkpoint-* and adapter)
    push_folder_to_hub(DPO_OFFLINE_PATH, DPO_OFFLINE_REPO, token)

    # 3) DPO online: push the final saved model
    push_folder_to_hub(DPO_ONLINE_PATH, DPO_ONLINE_REPO, token)

    print("Done. Repos:")
    print(f"  SFT:         https://huggingface.co/{SFT_REPO}")
    print(f"  DPO offline: https://huggingface.co/{DPO_OFFLINE_REPO}")
    print(f"  DPO online:  https://huggingface.co/{DPO_ONLINE_REPO}")


if __name__ == "__main__":
    main()

"""Load config from YAML with env var overrides."""
import os
from pathlib import Path

import yaml

_CONFIG = None


def load_config(config_path: str | None = None) -> dict:
    """Load config from YAML. Env vars override: SFT_MODEL_PATH, SFT_OUTPUT_DIR, WANDB_RUN_NAME."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "eval" not in cfg:
        if isinstance(cfg.get("dpo_online"), dict) and "eval" in cfg["dpo_online"]:
            cfg["eval"] = cfg["dpo_online"]["eval"].copy()

    
    if os.environ.get("SFT_MODEL_PATH"):
        cfg["models"]["sft_model_path"] = os.environ["SFT_MODEL_PATH"]
    if os.environ.get("EVAL_MODEL_PATH"):
        # make sure the top-level key exists before assigning
        if "eval" not in cfg:
            cfg["eval"] = {}
        cfg["eval"]["model_path"] = os.environ["EVAL_MODEL_PATH"]
    if os.environ.get("SFT_OUTPUT_DIR"):
        cfg["sft"]["output_dir"] = os.environ["SFT_OUTPUT_DIR"]
    if os.environ.get("WANDB_RUN_NAME"):
        cfg["sft"]["run_name"] = os.environ["WANDB_RUN_NAME"]
        cfg["dpo_offline"]["run_name"] = os.environ["WANDB_RUN_NAME"]
        cfg["dpo_online"]["run_name"] = os.environ["WANDB_RUN_NAME"]

    _CONFIG = cfg
    return cfg

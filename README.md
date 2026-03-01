# DPO Alpaca

A pipeline for training language models with Direct Preference Optimization (DPO) on the Alpaca instruction dataset. Supports both offline DPO (preference pairs) and online DPO (live judge).

## Project structure

```
DPO_Alpaca/
├── config/
│   ├── config.yaml        # Model IDs, paths, hyperparameters
│   └── load.py            # Loader with env var overrides
├── scripts/
│   ├── sft/                    # Supervised fine-tuning
│   │   └── sft_finetune.py
│   ├── preference/             # Preference data generation
│   │   └── distilabel_preference_data.py
│   └── dpo/                    # DPO training
│       ├── offline_rl_training.py
│       └── online_rl_training.py
├── requirements.txt
├── .env.example
└── README.md
```

## Pipeline

1. **SFT** (`scripts/sft/sft_finetune.py`) – Supervised fine-tuning on Alpaca with QLoRA
2. **Preference data** (`scripts/preference/distilabel_preference_data.py`) – Generate chosen/rejected pairs with a judge LLM
3. **Offline DPO** (`scripts/dpo/offline_rl_training.py`) – DPO training on preference pairs
4. **Online DPO** (`scripts/dpo/online_rl_training.py`) – Online DPO with a live judge (optional)

## Setup

```bash
pip install -r requirements.txt
```

For pushing preference data to Hugging Face Hub:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token
export HF_TOKEN=hf_your_token   # or source .env
```

**Important:** If a token was previously exposed in this repo, revoke it at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new one.

For experiment tracking with Weights & Biases:

```bash
pip install wandb
wandb login   # paste API key from wandb.ai
```

Training logs loss, eval metrics, and learning rate to wandb. To disable: `export WANDB_MODE=disabled` before running.

## Usage

### 1. SFT fine-tuning

```bash
python scripts/sft/sft_finetune.py
```

Output: `./qwen-sft-alpaca/` (or `SFT_OUTPUT_DIR`). Use the checkpoint path (e.g. `./qwen-sft-alpaca/checkpoint-XXX`) for later steps.

### 2. Generate preference data

```bash
python scripts/preference/distilabel_preference_data.py
```

Saves to `preference_data/`. Optionally pushes to Hub if `HF_TOKEN` is set.

### 3. Offline DPO

```bash
export SFT_MODEL_PATH=./qwen-dpo-offline-alpaca/checkpoint-xx   # path to your SFT checkpoint
python scripts/dpo/offline_rl_training.py
```

Output: `./dpo-final-run/`

### 4. Online DPO (optional)

```bash
export SFT_MODEL_PATH=./qwen-dpo-online-alpaca/checkpoint-xx
python scripts/dpo/online_rl_training.py
```

Output: `./online-dpo-qlora-final/`

### 5. Evaluate with BERTScore

```bash
export EVAL_MODEL_PATH=./qwen-sft-alpaca/checkpoint-xxx   # or SFT/DPO checkpoint
python eval.py
```

Uses held-out Alpaca examples, generates responses, and reports BERTScore (P, R, F1) vs references.

## Configuration

All model IDs, paths, and hyperparameters live in `config/config.yaml`. Override with env vars:

| Variable         | Description                        |
|------------------|------------------------------------|
| `HF_TOKEN`       | Hugging Face token (push to Hub)  |
| `SFT_MODEL_PATH` | Path to SFT checkpoint (used by DPO scripts) |
| `SFT_OUTPUT_DIR` | SFT output directory              |
| `WANDB_RUN_NAME` | Run name in wandb UI              |
| `CONFIG_PATH`    | Path to custom config YAML        |
| `EVAL_MODEL_PATH` | Model checkpoint for eval         |

Edit `config/config.yaml` to change batch size, learning rate, beta, max_length, epochs, etc.

## Models

- Base: `Qwen/Qwen1.5-1.8B`
- Preference generation: TinyLlama (generator), Falcon-7B (judge)
- Online DPO judge: TinyLlama (configurable)


## Evaluation

SFT
BERTScore on 20 examples:
  Precision: 0.8689
  Recall:    0.8965
  F1:        0.8819


DPO Offline

BERTScore on 20 examples:
  Precision: 0.8655
  Recall:    0.8964
  F1:        0.8801


DPO Online
  BERTScore on 20 examples:
  Precision: 0.8834
  Recall:    0.9047
  F1:        0.8936
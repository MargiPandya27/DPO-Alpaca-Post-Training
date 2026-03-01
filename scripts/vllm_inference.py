#!/usr/bin/env python3
"""
Minimal vLLM inference example – no API, no deployment.
Run: python scripts/vllm_inference_example.py

Use --backend transformers if vLLM keeps failing with "Engine core initialization failed".
Set MODEL_ID to your local path or Hugging Face repo (e.g. Qwen/Qwen2-0.5B-Instruct).
"""
import argparse
import os

# Override with env: VLLM_MODEL_ID or MODEL_ID. Use a repo with config.json + model.safetensors at root (e.g. merged DPO).
MODEL_ID = os.environ.get("VLLM_MODEL_ID") or os.environ.get("MODEL_ID") or "MargiPandya/qwen1.5-1.8b-dpo-offline-alpaca"


def _model_path_for_vllm(model_id: str, hf_token: str | None) -> str:
    """
    If model_id looks like a Hugging Face repo (org/repo), download it locally first.
    vLLM can fail when loading by repo id even when config.json exists (Hub resolution).
    Passing a local path to vLLM is more reliable.
    """
    path = model_id.strip()
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if path.startswith("./") or path.startswith("."):
        return path
    if path.count("/") == 1 and not os.path.exists(path):
        try:
            from huggingface_hub import snapshot_download
            print("Downloading model from Hub (avoids vLLM Hub resolution errors)...")
            local = snapshot_download(repo_id=path, token=hf_token)
            return local
        except Exception as e:
            print(f"Warning: pre-download failed ({e}). Using repo id for vLLM.")
    return path


def run_with_transformers(model_path: str, hf_token: str | None, prompts: list[str], max_tokens: int = 64, temperature: float = 0.7) -> None:
    """Fallback when vLLM fails: use Transformers for inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model with Transformers (fallback)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("Prompt:", prompt)
        print("Output:", text)
        print("---")


def run_with_vllm(model_path: str, prompts: list[str], max_tokens: int = 64, temperature: float = 0.7) -> None:
    """Use vLLM for inference (set env vars before import)."""
    os.environ["VLLM_USE_V1"] = "0"
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    from vllm import LLM, SamplingParams

    print("Loading model with vLLM...")
    llm = LLM(model=model_path, enforce_eager=True, gpu_memory_utilization=0.85)
    sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(prompts, sampling)
    for o in outputs:
        print("Prompt:", o.prompt)
        print("Output:", o.outputs[0].text if o.outputs else "")
        print("---")


def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM or Transformers")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm",
                        help="Use 'transformers' if vLLM fails with Engine core initialization failed")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if "/" in MODEL_ID and not os.path.isabs(MODEL_ID) and not MODEL_ID.startswith(".") and not hf_token:
        print("Hint: For private Hub models set HF_TOKEN")
    model_path = _model_path_for_vllm(MODEL_ID, hf_token)

    prompts = ["What is 2 + 2?", "Say hello in French."]
    if args.backend == "transformers":
        run_with_transformers(model_path, hf_token, prompts)
    else:
        try:
            run_with_vllm(model_path, prompts)
        except RuntimeError as e:
            err_text = str(e).lower()
            cause = e.__cause__
            while cause:
                err_text += " " + str(cause).lower()
                cause = getattr(cause, "__cause__", None)
            if "engine core" in err_text or "initialization failed" in err_text:
                print("vLLM engine failed. Retrying with Transformers backend...")
                run_with_transformers(model_path, hf_token, prompts)
            else:
                raise

if __name__ == "__main__":
    main()
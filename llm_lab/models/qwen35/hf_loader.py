from __future__ import annotations

import json
from pathlib import Path

import torch

from .configuration_qwen3_5 import Qwen3_5TextConfig
from .modeling_qwen3_5 import Qwen3_5ForCausalLM


def _remap_hf_key_to_local(key: str) -> str | None:
    if key.startswith("model.language_model."):
        return f"model.{key[len('model.language_model.'):]}"
    if key in {"lm_head.weight", "model.lm_head.weight"}:
        return "lm_head.weight"
    return None


def _collect_weight_files(weights_dir: Path) -> tuple[dict[str, list[str]], list[str]]:
    index_path = weights_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        keys_to_files: dict[str, list[str]] = {}
        for key, file_name in weight_map.items():
            remapped = _remap_hf_key_to_local(key)
            if remapped is None:
                continue
            keys_to_files.setdefault(file_name, []).append(key)
        files = sorted(keys_to_files.keys())
        return keys_to_files, files

    single_file = weights_dir / "model.safetensors"
    if not single_file.exists():
        raise FileNotFoundError(
            f"Could not find `model.safetensors.index.json` or `model.safetensors` in {weights_dir}."
        )
    return {"model.safetensors": []}, ["model.safetensors"]


def load_hf_qwen3_5_for_inference(
    weights_dir: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    strict: bool = False,
) -> tuple[Qwen3_5ForCausalLM, Qwen3_5TextConfig, dict]:
    """
    Load Qwen/Qwen3.5-* local HuggingFace weights into the local text reimplementation.
    """

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for loading HF weights. Install with `pip install safetensors`."
        ) from exc

    weights_dir = Path(weights_dir)
    config = Qwen3_5TextConfig.from_hf_config_file(weights_dir / "config.json")
    model = Qwen3_5ForCausalLM(config)
    model_state_keys = set(model.state_dict().keys())

    keys_to_files, weight_files = _collect_weight_files(weights_dir)
    remapped_state_dict = {}
    total_hf_text_tensors = 0
    loaded_text_tensors = 0

    for file_name in weight_files:
        shard_path = weights_dir / file_name
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if keys_to_files[file_name]:
                candidate_keys = keys_to_files[file_name]
            else:
                candidate_keys = list(f.keys())

            for key in candidate_keys:
                local_key = _remap_hf_key_to_local(key)
                if local_key is None:
                    continue
                total_hf_text_tensors += 1
                if local_key not in model_state_keys:
                    continue
                remapped_state_dict[local_key] = f.get_tensor(key)
                loaded_text_tensors += 1

    load_result = model.load_state_dict(remapped_state_dict, strict=strict)

    # Qwen3.5-0.8B config ties embeddings, and lm_head is usually absent in VLM checkpoints.
    if config.tie_word_embeddings and "lm_head.weight" not in remapped_state_dict:
        model.tie_weights()

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    report = {
        "loaded_text_tensors": loaded_text_tensors,
        "total_hf_text_tensors_seen": total_hf_text_tensors,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    return model, config, report

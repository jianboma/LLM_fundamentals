"""
This configuration file is adapted from llm_lab/nanochat/nanochat/gpt.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class GPT2Config:
    """
    Minimal GPT-2 style config used by the local reimplementation.
    """

    sequence_len: int = 2048
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 768
    window_pattern: str = "SSSL"
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "GPT2Config":
        remapped = dict(data)
        # Handle common HF GPT-2 naming.
        remapped.setdefault("sequence_len", remapped.get("n_positions", remapped.get("n_ctx", cls.sequence_len)))
        remapped.setdefault("n_layer", remapped.get("num_hidden_layers", cls.n_layer))
        remapped.setdefault("n_head", remapped.get("num_attention_heads", cls.n_head))
        remapped.setdefault("n_embd", remapped.get("hidden_size", cls.n_embd))
        remapped.setdefault("n_kv_head", remapped.get("n_kv_head", remapped.get("n_head", cls.n_head)))

        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in remapped.items() if k in allowed}
        return cls(**kwargs)

    @classmethod
    def from_hf_config_file(cls, config_path: str | Path) -> "GPT2Config":
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.n_kv_head <= 0 or self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head.")
        if not self.window_pattern:
            raise ValueError("window_pattern cannot be empty.")
        invalid = {c for c in self.window_pattern.upper() if c not in {"S", "L"}}
        if invalid:
            raise ValueError(f"window_pattern contains unsupported chars: {sorted(invalid)}")

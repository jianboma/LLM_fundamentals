from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


def _default_rope_parameters() -> dict:
    return {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    }


@dataclass
class Qwen3_5TextConfig:
    """
    Minimal Qwen3.5 text config used by the local reimplementation.
    """

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 256

    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    layer_types: list[str] | None = None
    full_attention_interval: int = 4

    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    rope_parameters: dict = field(default_factory=_default_rope_parameters)

    @classmethod
    def from_dict(cls, data: dict) -> "Qwen3_5TextConfig":
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)

    @classmethod
    def from_hf_config_file(cls, config_path: str | Path) -> "Qwen3_5TextConfig":
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        text_config = raw.get("text_config", raw)
        return cls.from_dict(text_config)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % self.full_attention_interval) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must match num_hidden_layers ({self.num_hidden_layers})."
            )
        invalid_types = set(self.layer_types) - {"full_attention", "linear_attention"}
        if invalid_types:
            raise ValueError(f"Unsupported layer types: {sorted(invalid_types)}")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive.")

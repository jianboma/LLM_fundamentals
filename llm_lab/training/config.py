from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    family: str
    extra_args: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        family = str(data.get("family", "nanochat_style"))
        raw_args = data.get("extra_args", [])
        if not isinstance(raw_args, list):
            raise ValueError("`extra_args` must be a list of strings.")
        extra_args = [str(x) for x in raw_args]
        return cls(family=family, extra_args=extra_args)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Train config JSON must be an object.")
        return cls.from_dict(payload)

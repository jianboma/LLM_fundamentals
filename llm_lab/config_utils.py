from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config at `{path}` must be a JSON object.")
    return payload


def resolve_value(cli_value: Any, config_value: Any, fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return fallback

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_lab.config_utils import load_json_object, resolve_value
from llm_lab.registry import get_family, list_families


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="llm_lab model construction CLI")
    parser.add_argument("--family", type=str, default=None, help="model family name")
    parser.add_argument("--family-config", type=str, default=None, help="family JSON config path")
    parser.add_argument("--list-families", action="store_true", help="print supported families")
    parser.add_argument("--config-file", type=str, default=None, help="path to JSON model config")
    parser.add_argument("--config-json", type=str, default=None, help='inline JSON model config, e.g. \'{"vocab_size":128,...}\'')
    parser.add_argument("--describe-only", action="store_true", help="print builder metadata without constructing model")
    return parser


def _load_config(config_file: str | None, config_json: str | None) -> dict[str, Any]:
    if config_file:
        with Path(config_file).open("r", encoding="utf-8") as f:
            payload = json.load(f)
    elif config_json:
        payload = json.loads(config_json)
    else:
        raise ValueError("Provide either --config-file or --config-json.")
    if not isinstance(payload, dict):
        raise ValueError("Model config must be a JSON object.")
    return payload


def _load_model_config(
    family_config_path: str | None,
    config_file: str | None,
    config_json: str | None,
) -> dict[str, Any]:
    if config_file or config_json:
        return _load_config(config_file, config_json)
    if not family_config_path:
        raise ValueError("Provide --config-file/--config-json, or use --family-config with a `model` section.")
    family_cfg = load_json_object(family_config_path)
    model_cfg = family_cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("Family config must include a `model` JSON object when build config is not provided.")
    return model_cfg


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_families:
        for family in list_families():
            print(
                f"{family.name}: builder={family.model_builder_name}, "
                f"training_supported={family.supports_training}"
            )
        return 0

    family_cfg = load_json_object(args.family_config) if args.family_config else {}
    family_name = resolve_value(args.family, family_cfg.get("family"), "qwen35")
    family = get_family(family_name)
    builder = family.create_model_builder()
    if args.describe_only:
        print(json.dumps(builder.describe(), indent=2, ensure_ascii=False))
        return 0

    config = _load_model_config(args.family_config, args.config_file, args.config_json)
    model = builder.build_from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "family": family.name,
                "builder": family.model_builder_name,
                "model_class": model.__class__.__name__,
                "num_params": int(num_params),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

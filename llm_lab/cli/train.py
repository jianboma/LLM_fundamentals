from __future__ import annotations

import argparse
import sys

from llm_lab.config_utils import load_json_object, resolve_value
from llm_lab.registry import list_families
from llm_lab.training.config import TrainConfig
from llm_lab.training.runner import run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="llm_lab training launcher")
    parser.add_argument("--family", type=str, default=None, help="model family name")
    parser.add_argument("--family-config", type=str, default=None, help="family JSON config path")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional JSON config file with keys: family, extra_args",
    )
    parser.add_argument("--list-families", action="store_true", help="print supported families")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="extra args forwarded to the family training entrypoint",
    )
    return parser


def _normalize_extra_args(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_families:
        for family in list_families():
            print(
                f"{family.name}: {family.description} | "
                f"builder={family.model_builder_name} | "
                f"training_supported={family.supports_training}"
            )
        return 0

    if args.config:
        config = TrainConfig.from_json_file(args.config)
    else:
        family_cfg = load_json_object(args.family_config) if args.family_config else {}
        family_name = resolve_value(args.family, family_cfg.get("family"), "nanochat_style")
        training_cfg = family_cfg.get("training", {})
        if training_cfg is not None and not isinstance(training_cfg, dict):
            parser.error("`training` in family config must be an object.")

        extra_args = _normalize_extra_args(args.extra_args)
        if not extra_args:
            default_args = training_cfg.get("default_args", [])
            if not isinstance(default_args, list):
                parser.error("`training.default_args` in family config must be a list.")
            extra_args = [str(x) for x in default_args]
        config = TrainConfig(family=family_name, extra_args=extra_args)
    try:
        return run_training(config)
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

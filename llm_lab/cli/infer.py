from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from llm_lab.config_utils import load_json_object, resolve_value
from llm_lab.registry import get_family, list_families


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="llm_lab inference CLI")
    parser.add_argument("--family", type=str, default=None, help="model family name")
    parser.add_argument("--family-config", type=str, default=None, help="family JSON config path")
    parser.add_argument("--list-families", action="store_true", help="print supported families")

    # qwen-related
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--assistant-model-dir", type=str, default=None)

    # nanochat-related
    parser.add_argument("--source", type=str, default=None, help="nanochat source: base|sft|rl")
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", type=str, default=None)

    # generation
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--return-full-text", action="store_true")
    return parser


def _engine_args(args: argparse.Namespace) -> dict[str, Any]:
    family_cfg = load_json_object(args.family_config) if args.family_config else {}
    inference_cfg = family_cfg.get("inference", {})
    if not isinstance(inference_cfg, dict):
        raise ValueError("`inference` in family config must be an object.")

    return {
        "weights_dir": resolve_value(args.weights_dir, inference_cfg.get("weights_dir"), None),
        "backend": resolve_value(args.backend, inference_cfg.get("backend"), "native"),
        "device": resolve_value(args.device, inference_cfg.get("device"), "cpu"),
        "assistant_model_dir": resolve_value(args.assistant_model_dir, inference_cfg.get("assistant_model_dir"), None),
        "source": resolve_value(args.source, inference_cfg.get("source"), "sft"),
        "model_tag": resolve_value(args.model_tag, inference_cfg.get("model_tag"), None),
        "step": resolve_value(args.step, inference_cfg.get("step"), None),
        "device_type": resolve_value(args.device_type, inference_cfg.get("device_type"), ""),
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_families:
        for family in list_families():
            print(
                f"{family.name}: {family.description} | "
                f"tokenizer={family.tokenizer_name} | "
                f"chat_template={family.chat_template_name} | "
                f"builder={family.model_builder_name} | "
                f"training_supported={family.supports_training}"
            )
        return 0

    if not args.prompt:
        parser.error("--prompt is required unless --list-families is set.")

    family_cfg = load_json_object(args.family_config) if args.family_config else {}
    family_name = resolve_value(args.family, family_cfg.get("family"), "qwen35")
    family = get_family(family_name)
    engine = family.create_inference_engine(_engine_args(args))
    output = engine.generate_text(
        prompt=args.prompt,
        generation_kwargs={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        return_full_text=args.return_full_text,
    )
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

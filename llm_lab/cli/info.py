from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from llm_lab.config_utils import load_json_object, resolve_value
from llm_lab.registry import get_family
from llm_lab.training.config import TrainConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect effective llm_lab configuration")
    parser.add_argument("--mode", type=str, default="infer", choices=["infer", "serve", "train", "build_model"])
    parser.add_argument("--family", type=str, default=None, help="model family name")
    parser.add_argument("--family-config", type=str, default=None, help="family JSON config path")

    # inference/serve related
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--assistant-model-dir", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)

    # training related
    parser.add_argument("--train-config", type=str, default=None, help="path to train config JSON")
    parser.add_argument("--train-extra", type=str, action="append", default=[], help="extra training arg, repeatable")

    # build-model related
    parser.add_argument("--config-file", type=str, default=None, help="path to model JSON config")
    parser.add_argument("--config-json", type=str, default=None, help="inline model JSON config")
    return parser


def _resolve_family(args: argparse.Namespace, family_cfg: dict[str, Any]) -> str:
    return resolve_value(args.family, family_cfg.get("family"), "qwen35")


def _resolve_inference_like(args: argparse.Namespace, family_cfg: dict[str, Any]) -> dict[str, Any]:
    inference_cfg = family_cfg.get("inference", {})
    if inference_cfg is not None and not isinstance(inference_cfg, dict):
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


def _resolve_serve(args: argparse.Namespace, family_cfg: dict[str, Any]) -> dict[str, Any]:
    serve_cfg = family_cfg.get("serve", {})
    if serve_cfg is not None and not isinstance(serve_cfg, dict):
        raise ValueError("`serve` in family config must be an object.")
    return {
        "model_id": resolve_value(args.model_id, serve_cfg.get("model_id"), "llm_lab-local"),
        "host": resolve_value(args.host, serve_cfg.get("host"), "127.0.0.1"),
        "port": resolve_value(args.port, serve_cfg.get("port"), 8000),
    }


def _resolve_train(args: argparse.Namespace, family_cfg: dict[str, Any], family_name: str) -> dict[str, Any]:
    if args.train_config:
        train_config = TrainConfig.from_json_file(args.train_config)
        return {
            "source": "train_config",
            "family": train_config.family,
            "extra_args": train_config.extra_args,
        }

    training_cfg = family_cfg.get("training", {})
    if training_cfg is not None and not isinstance(training_cfg, dict):
        raise ValueError("`training` in family config must be an object.")
    if args.train_extra:
        extra_args = list(args.train_extra)
        source = "cli_train_extra"
    else:
        default_args = training_cfg.get("default_args", [])
        if not isinstance(default_args, list):
            raise ValueError("`training.default_args` in family config must be a list.")
        extra_args = [str(x) for x in default_args]
        source = "family_config.training.default_args"
    return {
        "source": source,
        "family": family_name,
        "extra_args": extra_args,
    }


def _resolve_build_model(args: argparse.Namespace, family_cfg: dict[str, Any]) -> dict[str, Any]:
    if args.config_file:
        return {"source": "config_file", "config_file": args.config_file}
    if args.config_json:
        payload = json.loads(args.config_json)
        if not isinstance(payload, dict):
            raise ValueError("--config-json must decode to a JSON object.")
        return {"source": "config_json", "config": payload}
    model_cfg = family_cfg.get("model")
    if isinstance(model_cfg, dict):
        return {"source": "family_config.model", "config": model_cfg}
    return {"source": "none", "config": None}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    family_cfg = load_json_object(args.family_config) if args.family_config else {}
    family_name = _resolve_family(args, family_cfg)
    family = get_family(family_name)

    payload: dict[str, Any] = {
        "mode": args.mode,
        "family": family.name,
        "family_metadata": {
            "description": family.description,
            "tokenizer_name": family.tokenizer_name,
            "chat_template_name": family.chat_template_name,
            "model_builder_name": family.model_builder_name,
            "supports_training": family.supports_training,
        },
    }

    if args.mode in {"infer", "serve"}:
        payload["inference"] = _resolve_inference_like(args, family_cfg)
    if args.mode == "serve":
        payload["serve"] = _resolve_serve(args, family_cfg)
    if args.mode == "train":
        payload["training"] = _resolve_train(args, family_cfg, family_name)
    if args.mode == "build_model":
        payload["build_model"] = _resolve_build_model(args, family_cfg)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

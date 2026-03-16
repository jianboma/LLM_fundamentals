from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

from llm_lab.chat_templates import nanochat_conversation_template
from llm_lab.core import FamilySpec
from llm_lab.training.subprocess_launcher import launch_python_module


class NanochatStyleModelBuilder:
    def describe(self) -> dict[str, Any]:
        return {
            "family": "nanochat_style",
            "config_type": "NanochatStyleConfig",
            "build_modes": ["from_dict"],
            "model_class": "NanochatStyleLM",
        }

    def build_from_config(self, config: dict[str, Any], **kwargs: Any):
        from llm_lab.nanochat_style.model import NanochatStyleConfig, NanochatStyleLM

        model_config = NanochatStyleConfig.from_dict(config)
        model = NanochatStyleLM(model_config)
        return model


def _create_model_builder() -> NanochatStyleModelBuilder:
    return NanochatStyleModelBuilder()


class NanochatStyleInferenceEngine:
    """
    Local inference engine for llm_lab nanochat_style.
    """

    def __init__(self, source: str = "sft", model_tag: str | None = None, step: int | None = None, device_type: str = ""):
        from llm_lab.nanochat_style.inference import NanochatStyleInferenceEngine as _Engine

        self._impl = _Engine(source=source, model_tag=model_tag, step=step, device_type=device_type or "cpu")
        self.report = self._impl.report

    @staticmethod
    def _normalize_kwargs(generation_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        from llm_lab.nanochat_style.inference import NanochatStyleInferenceEngine as _Engine

        return _Engine._normalize(generation_kwargs)

    def stream_text(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> Generator[str, None, None]:
        yield from self._impl.stream_text(prompt, generation_kwargs)

    def generate_text(
        self,
        prompt: str,
        generation_kwargs: dict[str, Any] | None = None,
        return_full_text: bool = False,
    ) -> dict[str, Any]:
        return self._impl.generate_text(prompt, generation_kwargs, return_full_text)


def _create_inference_engine(args: dict[str, Any]) -> NanochatStyleInferenceEngine:
    return NanochatStyleInferenceEngine(
        source=args.get("source", "sft"),
        model_tag=args.get("model_tag"),
        step=args.get("step"),
        device_type=args.get("device_type", ""),
    )


def _launch_training(extra_args: list[str]) -> int:
    project_root = Path(__file__).resolve().parents[2]
    return launch_python_module("llm_lab.nanochat_style.train", cwd=project_root, args=extra_args)


NANOCHAT_STYLE_FAMILY = FamilySpec(
    name="nanochat_style",
    description="Nanochat-style local training/inference pipeline (decoupled from third_party)",
    tokenizer_name="byte_tokenizer",
    chat_template_name="nanochat_special_token_conversation",
    model_builder_name="nanochat_style_lm_builder",
    create_model_builder=_create_model_builder,
    supports_training=True,
    render_chat_prompt=nanochat_conversation_template,
    create_inference_engine=_create_inference_engine,
    launch_training=_launch_training,
)

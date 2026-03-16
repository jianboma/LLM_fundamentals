from __future__ import annotations

from typing import Any

from llm_lab.chat_templates import role_prompt_template
from llm_lab.core import FamilySpec


class Qwen35ModelBuilder:
    def describe(self) -> dict[str, Any]:
        return {
            "family": "qwen35",
            "config_type": "Qwen3_5TextConfig",
            "build_modes": ["from_dict", "from_hf_config_file"],
            "model_class": "Qwen3_5ForCausalLM",
        }

    def build_from_config(self, config: dict[str, Any], **kwargs: Any):
        from llm_lab.models.qwen35 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        config_path = config.get("config_path")
        if config_path:
            qwen_config = Qwen3_5TextConfig.from_hf_config_file(config_path)
        else:
            qwen_config = Qwen3_5TextConfig.from_dict(config)
        model = Qwen3_5ForCausalLM(qwen_config)
        if kwargs.get("dtype") is not None:
            model = model.to(dtype=kwargs["dtype"])
        if kwargs.get("device") is not None:
            model = model.to(kwargs["device"])
        return model


def _create_model_builder() -> Qwen35ModelBuilder:
    return Qwen35ModelBuilder()


def _create_inference_engine(args: dict[str, Any]):
    from llm_lab.models.qwen35 import Qwen3_5InferenceEngine

    weights_dir = args.get("weights_dir")
    if not weights_dir:
        raise ValueError("qwen35 requires `weights_dir`.")

    return Qwen3_5InferenceEngine(
        weights_dir=weights_dir,
        backend=args.get("backend", "native"),
        device=args.get("device", "cpu"),
        dtype=args.get("dtype"),
        assistant_model_dir=args.get("assistant_model_dir"),
    )


def _launch_training(extra_args: list[str]) -> int:
    raise NotImplementedError(
        "qwen35 training is not wired in llm_lab yet. "
        "Use this family for inference while training integration is expanded."
    )


QWEN35_FAMILY = FamilySpec(
    name="qwen35",
    description="Qwen3.5 text family with local llm_lab implementation",
    tokenizer_name="qwen35_tokenizer",
    chat_template_name="qwen35_role_prompt_template",
    model_builder_name="qwen35_text_builder",
    create_model_builder=_create_model_builder,
    supports_training=False,
    render_chat_prompt=role_prompt_template,
    create_inference_engine=_create_inference_engine,
    launch_training=_launch_training,
)

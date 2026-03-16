from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class InferenceEngine(Protocol):
    def generate_text(
        self,
        prompt: str,
        generation_kwargs: dict[str, Any] | None = None,
        return_full_text: bool = False,
    ) -> dict[str, Any]:
        ...

    def stream_text(
        self,
        prompt: str,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        ...


class ModelBuilder(Protocol):
    def describe(self) -> dict[str, Any]:
        ...

    def build_from_config(self, config: dict[str, Any], **kwargs: Any) -> Any:
        ...


CreateInferenceEngine = Callable[[dict[str, Any]], InferenceEngine]
CreateModelBuilder = Callable[[], ModelBuilder]
LaunchTraining = Callable[[list[str]], int]
RenderChatPrompt = Callable[[list[dict[str, Any]]], str]


@dataclass(frozen=True)
class FamilySpec:
    name: str
    description: str
    tokenizer_name: str
    chat_template_name: str
    model_builder_name: str
    create_model_builder: CreateModelBuilder
    supports_training: bool
    render_chat_prompt: RenderChatPrompt
    create_inference_engine: CreateInferenceEngine
    launch_training: LaunchTraining

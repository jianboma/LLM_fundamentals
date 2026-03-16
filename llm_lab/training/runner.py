from __future__ import annotations

from llm_lab.registry import get_family
from llm_lab.training.config import TrainConfig


def run_training(config: TrainConfig) -> int:
    family = get_family(config.family)
    return family.launch_training(config.extra_args)

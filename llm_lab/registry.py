from __future__ import annotations

from llm_lab.core import FamilySpec
from llm_lab.families import NANOCHAT_STYLE_FAMILY, QWEN35_FAMILY


_FAMILIES: dict[str, FamilySpec] = {
    QWEN35_FAMILY.name: QWEN35_FAMILY,
    NANOCHAT_STYLE_FAMILY.name: NANOCHAT_STYLE_FAMILY,
}


def list_families() -> list[FamilySpec]:
    return list(_FAMILIES.values())


def get_family(name: str) -> FamilySpec:
    try:
        return _FAMILIES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_FAMILIES))
        raise ValueError(f"Unknown family `{name}`. Supported: {supported}") from exc

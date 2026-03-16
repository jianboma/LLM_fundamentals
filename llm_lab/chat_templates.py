from __future__ import annotations

from typing import Any


def role_prompt_template(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def nanochat_conversation_template(messages: list[dict[str, Any]]) -> str:
    # Keep a plain text prompt here; model-family tokenizer and special-token handling
    # remain inside nanochat inference/training modules.
    lines = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        lines.append(f"[{role}]\n{content}")
    lines.append("[ASSISTANT]")
    return "\n\n".join(lines)

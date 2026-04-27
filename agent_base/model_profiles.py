from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelProfile:
    family: str
    context_window: int
    output_reserve_tokens: int
    compact_buffer_tokens: int
    recent_history_budget_tokens: int
    compact_summary_max_tokens: int
    compact_trigger_tokens_override: Optional[int] = None

    @property
    def compact_trigger_tokens(self) -> int:
        if self.compact_trigger_tokens_override is not None:
            return self.compact_trigger_tokens_override
        return max(256, self.context_window - self.output_reserve_tokens - self.compact_buffer_tokens)


def _model_family(model_name: str) -> str:
    normalized = str(model_name or "").strip().casefold()
    if "gemini" in normalized:
        return "gemini"
    if "claude" in normalized:
        return "claude"
    if "deepseek" in normalized:
        return "deepseek"
    if "qwen" in normalized:
        return "qwen"
    if "glm" in normalized:
        return "glm"
    if "gpt" in normalized or "o1" in normalized or "o3" in normalized or "o4" in normalized:
        return "gpt"
    return "generic"


def resolve_model_profile(
    model_name: str,
    *,
    configured_max_input_tokens: int,
    configured_max_output_tokens: int,
    compact_trigger_tokens: Any = None,
) -> ModelProfile:
    context_window = max(1024, int(configured_max_input_tokens))
    output_reserve_tokens = max(128, min(int(configured_max_output_tokens), max(256, context_window // 12)))
    compact_buffer_tokens = max(64, min(4096, context_window // 20))
    recent_history_budget_tokens = max(128, min(16384, context_window // 8))
    compact_summary_max_tokens = max(256, min(2048, context_window // 16))
    compact_trigger_override = parse_compact_trigger_tokens(compact_trigger_tokens, context_window=context_window)

    family = _model_family(model_name)
    if family in {"claude", "deepseek", "gemini"}:
        compact_buffer_tokens = max(compact_buffer_tokens, 1024)
        recent_history_budget_tokens = max(recent_history_budget_tokens, 1024)

    return ModelProfile(
        family=family,
        context_window=context_window,
        output_reserve_tokens=output_reserve_tokens,
        compact_buffer_tokens=compact_buffer_tokens,
        recent_history_budget_tokens=recent_history_budget_tokens,
        compact_summary_max_tokens=compact_summary_max_tokens,
        compact_trigger_tokens_override=compact_trigger_override,
    )


def parse_compact_trigger_tokens(value: Any, *, context_window: int) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("compact trigger tokens must not be a boolean.")
    if isinstance(value, int):
        parsed = value
    else:
        text = str(value).strip().casefold()
        if not text:
            return None
        multiplier = 1
        if text.endswith("k"):
            multiplier = 1024
            text = text[:-1].strip()
        elif text.endswith("m"):
            multiplier = 1024 * 1024
            text = text[:-1].strip()
        text = text.replace("_", "").replace(",", "")
        parsed = int(text) * multiplier
    parsed = max(256, parsed)
    return min(parsed, max(256, int(context_window)))

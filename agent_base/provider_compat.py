import re
from typing import Any


_MODEL_NAME_SPLIT_RE = re.compile(r"[/:\s]+")


def model_rejects_sampling_params(model_name: str) -> bool:
    normalized = str(model_name or "").strip().casefold()
    if not normalized:
        return False
    parts = [part for part in _MODEL_NAME_SPLIT_RE.split(normalized) if part]
    return any(part.startswith("claude") for part in parts)


def apply_sampling_params(
    request_kwargs: dict[str, Any],
    *,
    model_name: str,
    temperature: Any = None,
    top_p: Any = None,
    presence_penalty: Any = None,
) -> None:
    if model_rejects_sampling_params(model_name):
        return
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if top_p is not None:
        request_kwargs["top_p"] = top_p
    if presence_penalty is not None:
        request_kwargs["presence_penalty"] = presence_penalty

"""Turn each provider's usage block into one record.

How the numbers work everywhere in the harness:
- `input_tokens` is the whole prompt, cached tokens included.
- `cached_input_tokens` is how much of that came from the cache.
- `output_tokens` is everything billed as output, reasoning included.
- `reasoning_tokens` is how much of that was reasoning.

Providers count these differently, so each function below says which way it assumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Protocol = Literal["responses", "chat_completions", "messages"]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    protocol: Protocol

    @property
    def uncached_input_tokens(self) -> int:
        return self.input_tokens - self.cached_input_tokens

    @property
    def visible_output_tokens(self) -> int:
        return self.output_tokens - self.reasoning_tokens

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UsageError(ValueError):
    """The usage block is missing or doesn't add up."""


def _int(block: dict[str, Any] | None, key: str) -> int:
    if not block:
        return 0
    value = block.get(key)
    return int(value) if value is not None else 0


def _check(usage: Usage) -> Usage:
    if usage.cached_input_tokens > usage.input_tokens:
        raise UsageError(f"cached tokens exceed input tokens: {usage}")
    if usage.reasoning_tokens > usage.output_tokens:
        raise UsageError(f"reasoning tokens exceed output tokens: {usage}")
    return usage


def from_responses(usage: dict[str, Any] | None) -> Usage:
    """Responses format. `input_tokens` already includes cached tokens and `output_tokens` already includes reasoning."""
    if usage is None:
        raise UsageError("response has no usage block")
    return _check(
        Usage(
            input_tokens=_int(usage, "input_tokens"),
            cached_input_tokens=_int(usage.get("input_tokens_details"), "cached_tokens"),
            output_tokens=_int(usage, "output_tokens"),
            reasoning_tokens=_int(usage.get("output_tokens_details"), "reasoning_tokens"),
            protocol="responses",
        )
    )


def from_chat_completions(usage: dict[str, Any] | None) -> Usage:
    """Chat Completions format. `prompt_tokens` already includes cached tokens and `completion_tokens` already includes reasoning."""
    if usage is None:
        raise UsageError("response has no usage block")
    return _check(
        Usage(
            input_tokens=_int(usage, "prompt_tokens"),
            cached_input_tokens=_int(usage.get("prompt_tokens_details"), "cached_tokens"),
            output_tokens=_int(usage, "completion_tokens"),
            reasoning_tokens=_int(usage.get("completion_tokens_details"), "reasoning_tokens"),
            protocol="chat_completions",
        )
    )


def from_messages(usage: dict[str, Any] | None, *, input_includes_cached: bool = False) -> Usage:
    """Messages format (Anthropic style).

    Anthropic leaves cache reads and cache writes out of `input_tokens`. Other providers that offer this
    format might not, so it's a setting. The default is the Anthropic way. Some providers report reasoning
    as `output_tokens_details.thinking_tokens`, and I assume that's already inside `output_tokens` like
    Anthropic does it.
    """
    if usage is None:
        raise UsageError("response has no usage block")
    reported_input = _int(usage, "input_tokens")
    cache_read = _int(usage, "cache_read_input_tokens")
    cache_write = _int(usage, "cache_creation_input_tokens")
    total_input = reported_input if input_includes_cached else reported_input + cache_read + cache_write
    details = usage.get("output_tokens_details") or {}
    reasoning = _int(details, "thinking_tokens") or _int(details, "reasoning_tokens")
    return _check(
        Usage(
            input_tokens=total_input,
            cached_input_tokens=cache_read,
            output_tokens=_int(usage, "output_tokens"),
            reasoning_tokens=reasoning,
            protocol="messages",
        )
    )


def normalize(protocol: Protocol, usage: dict[str, Any] | None, **kwargs: Any) -> Usage:
    if protocol == "responses":
        return from_responses(usage)
    if protocol == "chat_completions":
        return from_chat_completions(usage)
    if protocol == "messages":
        return from_messages(usage, **kwargs)
    raise ValueError(f"unknown protocol: {protocol}")

"""Meta's Model API, for Muse Spark and Muse Wonder.

    python -m harness serve --muse --policies cache

It's the OpenAI-compatible adapter with Muse's base URL and the rules its API enforces, so a bad
request gets a clear error here instead of a 400 from Meta. From the Model API docs (read Oct 2026):

- `tool_choice` can only be `"auto"`.
- Reasoning can't be turned off. Effort `"none"` gets a 400. Use `"minimal"` or `"low"`.
- `previous_response_id` makes the usage count the newest turn only, which would make every cost number
  wrong. Send the whole conversation instead.

Caching is automatic. What `cache_layout` adds on Muse is `prompt_cache_key`, and that turned out to
matter a lot. See docs/MUSE.md.
"""

from __future__ import annotations

from typing import Any

from harness.adapters.base import RequestRejected
from harness.adapters.openai_compat import OpenAICompatAdapter
from harness.usage import Protocol

BASE_URL = "https://api.meta.ai/v1"


class MuseAdapter(OpenAICompatAdapter):
    def __init__(self, base_url: str = BASE_URL, *, provider: str = "meta", **kwargs: Any):
        super().__init__(base_url, provider=provider, **kwargs)

    def check(self, protocol: Protocol, body: dict[str, Any]) -> None:
        if protocol == "responses" and body.get("previous_response_id"):
            raise RequestRejected("previous_response_id makes Muse count usage for the new turn only. Send the full conversation")
        choice = body.get("tool_choice")
        if choice is not None and choice != "auto":
            raise RequestRejected(f'Muse only accepts tool_choice="auto", got {choice!r}')
        effort = (body.get("reasoning") or {}).get("effort") if protocol == "responses" else body.get("reasoning_effort")
        if effort == "none":
            raise RequestRejected('reasoning can\'t be turned off on Muse. Use "minimal" or "low"')

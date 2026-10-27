"""Reasoning effort per call: think less on easy calls, full effort on anything that changes something.

Reasoning tokens are billed as output, and caching does nothing for them. Most agent calls are easy,
like reading a tool result or asking the user for an ID. A few are real decisions, like refunding an
order. Guessing which is which before the call is risky, so this doesn't guess.

Every call goes out at `low` effort. If the reply calls a tool that changes something, the harness
throws that reply away and asks again at the agent's own effort (or `high` if set). So every action
that changes something is still decided at full effort. The second call has the same prompt, so it
mostly comes from the cache, and the thrown-away call is cheap because it reasoned less.

A tool counts as read-only if its name starts with one of `read_prefixes`, or is in `read_tools`.
Anything else counts as a write. List your write tools in `write_tools` if names don't follow a pattern.

Some providers keep a separate cache for each effort level. On one we tested, every switch to full
effort only got the shared tools and system prompt from the cache, and paid for the rest of the
conversation again. That ate all the savings. If yours does that, set `escalate=False`. Then the whole
conversation stays at `low`, the cache never switches, and nothing gets asked twice. You lose the full
effort on writes, so check success before you trust it.
"""

from __future__ import annotations

from typing import Any

from harness.policies import conv
from harness.policies.base import Policy, SessionContext
from harness.usage import Protocol

READ_PREFIXES = ("get", "find", "list", "search", "read", "lookup", "fetch", "query", "view", "show", "check", "calculate")


class ReasoningEffort(Policy):
    name = "reasoning"

    def __init__(
        self,
        *,
        low: str = "low",
        high: str | None = None,
        read_prefixes: str | list[str] | tuple[str, ...] = READ_PREFIXES,
        read_tools: str | list[str] | tuple[str, ...] = (),
        write_tools: str | list[str] | tuple[str, ...] | None = None,
        escalate: bool = True,
    ):
        self.low = low
        self.escalate = escalate
        self.high = high
        self.read_prefixes = tuple(p.lower() for p in _names(read_prefixes))
        self.read_tools = set(_names(read_tools))
        self.write_tools = set(_names(write_tools)) if write_tools is not None else None

    def apply(self, protocol: Protocol, request: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        if protocol == "messages":
            return request  # effort works differently there. Leave it alone
        state = ctx.state_for(self.name)
        state["agent_effort"] = get_effort(protocol, request)
        set_effort(protocol, request, self.low)
        return request

    def review(self, protocol: Protocol, request: dict[str, Any], response: dict[str, Any], ctx: SessionContext) -> dict[str, Any] | None:
        if not self.escalate or protocol == "messages" or get_effort(protocol, request) != self.low:
            return None
        writes = [c.name for c in conv.response_tool_calls(protocol, response) if self.is_write(c.name, ctx)]
        if not writes:
            return None
        state = ctx.state_for(self.name)
        state["escalations"] = state.get("escalations", 0) + 1
        effort = self.high or state.get("agent_effort")
        retry = dict(request)
        set_effort(protocol, retry, effort)
        return retry

    def is_write(self, name: str, ctx: SessionContext) -> bool:
        if name in ctx.harness_tools or name in self.read_tools:
            return False
        if self.write_tools is not None:
            return name in self.write_tools
        return not name.lower().startswith(self.read_prefixes)


def _names(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Lists can come in as "a,b,c" from `serve --set`."""
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def get_effort(protocol: Protocol, request: dict[str, Any]) -> str | None:
    if protocol == "responses":
        return (request.get("reasoning") or {}).get("effort")
    return request.get("reasoning_effort")


def set_effort(protocol: Protocol, request: dict[str, Any], effort: str | None) -> None:
    """Set the effort, or remove it when `effort` is None so the provider's default applies."""
    if protocol == "responses":
        reasoning = dict(request.get("reasoning") or {})
        if effort is None:
            reasoning.pop("effort", None)
        else:
            reasoning["effort"] = effort
        if reasoning:
            request["reasoning"] = reasoning
        else:
            request.pop("reasoning", None)
    elif effort is None:
        request.pop("reasoning_effort", None)
    else:
        request["reasoning_effort"] = effort

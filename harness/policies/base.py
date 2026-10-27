"""What every policy looks like.

A policy rewrites a request before it gets sent. Given the same request and the same state it has to
give back the exact same output every time. If it doesn't, it breaks the cache it's supposed to help.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harness.usage import Protocol


@dataclass(frozen=True)
class HarnessTool:
    """A tool the harness runs itself. The agent never sees calls to it."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


@dataclass
class SessionContext:
    """State for one conversation, which is one task trial in a benchmark."""

    task_id: str
    state: dict[str, Any] = field(default_factory=dict)  # each policy's own state, by name
    harness_tools: dict[str, HarnessTool] = field(default_factory=dict)

    def state_for(self, policy: str) -> dict[str, Any]:
        return self.state.setdefault(policy, {})


class Policy(ABC):
    name: str

    def setup(self, ctx: SessionContext) -> None:
        """Runs once per session before the first request, for example to register harness tools."""

    @abstractmethod
    def apply(self, protocol: Protocol, request: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        """Return the rewritten request. `request` is already a copy, so it's fine to change it."""

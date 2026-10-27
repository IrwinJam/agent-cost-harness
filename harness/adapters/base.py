"""Provider adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness.usage import Protocol, Usage


class RequestRejected(ValueError):
    """The harness wouldn't send this request, because the provider would reject it or it would break cost tracking."""


@dataclass
class CallResult:
    provider: str
    protocol: Protocol
    model: str
    request: dict[str, Any]
    response: dict[str, Any]
    usage: Usage
    latency_s: float
    attempts: int
    request_id: str | None = None
    rate_limit: dict[str, str] = field(default_factory=dict)


class Adapter(ABC):
    provider: str

    @abstractmethod
    def call(self, protocol: Protocol, request: dict[str, Any]) -> CallResult: ...

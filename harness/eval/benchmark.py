"""What a benchmark looks like to the runner.

A benchmark runs one task trial against an agent session and says if it passed. Every agent model call
has to go through that session. Anything else it spends money on, like a simulated user, goes in
`sim_cost_usd` so it never counts as agent savings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness.session import Session


@dataclass(frozen=True)
class Outcome:
    success: bool
    reward: float | None = None
    sim_cost_usd: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    name: str

    @abstractmethod
    def task_ids(self) -> list[str]: ...

    @abstractmethod
    def run(self, task_id: str, trial: int, agent: Session) -> Outcome: ...

    def close(self) -> None:
        """Clean up servers or processes. The runner calls this when it's done."""

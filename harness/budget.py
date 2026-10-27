"""Spending limits.

`Budget` is a simple cap on what one running process spends. It starts at zero every time.

`SpendLimit` reads what's already been spent from the trace database, so its limits hold across
restarts. You can set any of these:
- `total_usd`: everything ever saved in this database
- `daily_usd`: everything since midnight, local time
- `per_conversation_usd`: one conversation (one trace run)

Both check before each call. A call that's already on its way can still go over by its own cost.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.trace import TraceStore


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, limit_usd: float, spent_usd: float = 0.0):
        if limit_usd <= 0:
            raise ValueError("limit_usd must be > 0")
        self.limit_usd = limit_usd
        self._spent = spent_usd
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        return self._spent

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self._spent)

    def check(self) -> None:
        """Stop before a call once the limit is hit."""
        if self._spent >= self.limit_usd:
            raise BudgetExceeded(f"spent ${self._spent:.4f} of ${self.limit_usd:.2f}")

    def charge(self, usd: float) -> None:
        with self._lock:
            self._spent += usd


@dataclass(frozen=True)
class SpendLimit:
    total_usd: float | None = None
    daily_usd: float | None = None
    per_conversation_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("total_usd", "daily_usd", "per_conversation_usd"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0")

    @property
    def is_set(self) -> bool:
        return any(v is not None for v in (self.total_usd, self.daily_usd, self.per_conversation_usd))

    def check(self, store: TraceStore, run_id: str | None = None, now: dt.datetime | None = None) -> None:
        """Raise `BudgetExceeded` if any limit is already used up."""
        if self.total_usd is not None:
            spent = store.spent()
            if spent >= self.total_usd:
                raise BudgetExceeded(f"total limit reached: spent ${spent:.4f} of ${self.total_usd:.2f}")
        if self.daily_usd is not None:
            spent = store.spent(since=start_of_day(now))
            if spent >= self.daily_usd:
                raise BudgetExceeded(f"daily limit reached: spent ${spent:.4f} of ${self.daily_usd:.2f} today")
        if self.per_conversation_usd is not None and run_id is not None:
            spent = store.spent(run_id=run_id)
            if spent >= self.per_conversation_usd:
                raise BudgetExceeded(
                    f"conversation limit reached: spent ${spent:.4f} of ${self.per_conversation_usd:.2f} in this conversation"
                )


def start_of_day(now: dt.datetime | None = None) -> dt.datetime:
    """Midnight today in local time."""
    now = (now or dt.datetime.now()).astimezone()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)

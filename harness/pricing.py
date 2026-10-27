"""Price table stored by date, and the cost math."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from harness.usage import Usage

PER_TOKEN = 1_000_000


@dataclass(frozen=True)
class Price:
    provider: str
    model: str
    effective_from: dt.date
    input: float
    cached_input: float
    output: float
    verified: bool


@dataclass(frozen=True)
class Cost:
    uncached_input: float
    cached_input: float
    output: float
    price: Price

    @property
    def total(self) -> float:
        return self.uncached_input + self.cached_input + self.output

    @property
    def cache_savings(self) -> float:
        """How much the cached tokens saved compared to paying full input price for them."""
        if self.price.cached_input == 0:
            return 0.0
        full = self.cached_input * self.price.input / self.price.cached_input
        return full - self.cached_input


class PriceTable:
    def __init__(self, data: dict[str, Any]):
        self._prices: dict[tuple[str, str], list[Price]] = {}
        for provider, spec in data.items():
            for model, rows in spec["models"].items():
                prices = [
                    Price(
                        provider=provider,
                        model=model,
                        effective_from=_as_date(row["effective_from"]),
                        input=float(row["input"]),
                        cached_input=float(row["cached_input"]),
                        output=float(row["output"]),
                        verified=bool(row.get("verified", False)),
                    )
                    for row in rows
                ]
                self._prices[(provider, model)] = sorted(prices, key=lambda p: p.effective_from)

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceTable:
        if path is None:
            text = resources.files("harness").joinpath("prices.yaml").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        return cls(yaml.safe_load(text))

    def lookup(self, provider: str, model: str, on: dt.date | None = None) -> Price:
        on = on or dt.date.today()
        rows = self._prices.get((provider, _base_model(model, provider, self._prices)))
        if not rows:
            raise KeyError(f"no price for {provider}/{model}")
        valid = [p for p in rows if p.effective_from <= on]
        if not valid:
            raise KeyError(f"no price for {provider}/{model} effective on {on}")
        return valid[-1]

    def cost(self, provider: str, model: str, usage: Usage, on: dt.date | None = None) -> Cost:
        price = self.lookup(provider, model, on)
        return Cost(
            uncached_input=usage.uncached_input_tokens * price.input / PER_TOKEN,
            cached_input=usage.cached_input_tokens * price.cached_input / PER_TOKEN,
            output=usage.output_tokens * price.output / PER_TOKEN,
            price=price,
        )


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


_SNAPSHOT_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")


def _base_model(model: str, provider: str, prices: dict[tuple[str, str], list[Price]]) -> str:
    """Map a dated model name like `example-model-2026-09-01` back to its priced model.

    Only a date at the end gets stripped. Something like `-contributor` is a different price tier and
    needs its own row, so it never gets priced as the base model by accident.
    """
    if (provider, model) in prices:
        return model
    return _SNAPSHOT_SUFFIX.sub("", model)

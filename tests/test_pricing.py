import datetime as dt

import pytest

from harness.pricing import PriceTable
from harness.usage import Usage

TABLE = {
    "example": {
        "models": {
            "example-model": [
                {"effective_from": "2026-07-01", "input": 1.0, "cached_input": 0.1, "output": 4.0},
                {"effective_from": "2026-12-01", "input": 2.0, "cached_input": 0.2, "output": 8.0, "verified": True},
            ],
            "example-model-batch": [
                {"effective_from": "2026-07-01", "input": 0.1, "cached_input": 0.002, "output": 0.2},
            ],
        }
    }
}


def usage(inp=1_000_000, cached=0, out=0, reasoning=0):
    return Usage(inp, cached, out, reasoning, "responses")


def test_cost_splits_cached_and_uncached_input():
    table = PriceTable(TABLE)
    cost = table.cost("example", "example-model", usage(inp=1_000_000, cached=600_000, out=100_000), dt.date(2026, 10, 26))
    assert cost.uncached_input == pytest.approx(0.4)
    assert cost.cached_input == pytest.approx(0.06)
    assert cost.output == pytest.approx(0.4)
    assert cost.total == pytest.approx(0.86)
    assert cost.cache_savings == pytest.approx(0.54)


def test_reasoning_billed_inside_output():
    table = PriceTable(TABLE)
    with_reasoning = table.cost("example", "example-model", usage(inp=0, out=1000, reasoning=900), dt.date(2026, 10, 26))
    without = table.cost("example", "example-model", usage(inp=0, out=1000, reasoning=0), dt.date(2026, 10, 26))
    assert with_reasoning.total == without.total


def test_price_versioned_by_date():
    table = PriceTable(TABLE)
    assert table.lookup("example", "example-model", dt.date(2026, 11, 30)).input == 1.0
    assert table.lookup("example", "example-model", dt.date(2026, 12, 1)).input == 2.0
    with pytest.raises(KeyError):
        table.lookup("example", "example-model", dt.date(2026, 1, 1))


def test_dated_snapshot_maps_to_family():
    table = PriceTable(TABLE)
    assert table.lookup("example", "example-model-2026-09-01", dt.date(2026, 10, 26)).model == "example-model"
    assert table.lookup("example", "example-model-20260901", dt.date(2026, 10, 26)).model == "example-model"


def test_unknown_tier_suffix_is_not_priced_as_base():
    table = PriceTable(TABLE)
    assert table.lookup("example", "example-model-batch", dt.date(2026, 10, 26)).input == 0.1
    with pytest.raises(KeyError):
        table.lookup("example", "example-model-full", dt.date(2026, 10, 26))


def test_bundled_price_table_loads():
    table = PriceTable.load()
    price = table.lookup("example", "example-model-large", dt.date(2026, 10, 26))
    assert price.output > price.input > price.cached_input
    assert price.verified is False

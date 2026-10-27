"""Show what's been spent.

    python -m harness spend [--db runs/serve.sqlite] [--max-total 20] [--max-per-day 5]

Prints the total, today's spend, and the most expensive conversations. Pass the same limits you give
`serve` to also see how much is left.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness.budget import start_of_day
from harness.trace import TraceStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness spend", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="runs/serve.sqlite", help="trace database")
    parser.add_argument("--max-total", type=float, help="show what's left of this total limit")
    parser.add_argument("--max-per-day", type=float, help="show what's left of this daily limit")
    parser.add_argument("--top", type=int, default=5, help="how many conversations to list")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"no trace database at {args.db}", file=sys.stderr)
        return 1
    with TraceStore(args.db) as store:
        total = store.spent()
        today = store.spent(since=start_of_day())
        top = store.spend_by_run(args.top)

    print(f"total: ${total:.4f}" + _left(total, args.max_total))
    print(f"today: ${today:.4f}" + _left(today, args.max_per_day))
    if top:
        print("\nmost expensive conversations:")
        for row in top:
            print(f"  ${row['cost_usd']:.4f}  {row['calls']:>4} calls  {row['label'] or row['id']}")
    return 0


def _left(spent: float, limit: float | None) -> str:
    if not limit:
        return ""
    return f" of ${limit:g}  (${max(0.0, limit - spent):.4f} left)" + ("  LIMIT REACHED" if spent >= limit else "")


if __name__ == "__main__":
    raise SystemExit(main())

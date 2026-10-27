"""Command line: ``python -m harness.profiler [--db PATH] [--run ID | --list] [--format md|json]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness.profiler.report import build_profile
from harness.trace import TraceStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.profiler", description=__doc__)
    parser.add_argument("--db", default="runs/traces.sqlite", help="trace database (default: runs/traces.sqlite)")
    parser.add_argument("--run", help="run id to profile (default: the most recent run)")
    parser.add_argument("--list", action="store_true", help="list runs and exit")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    parser.add_argument("--top", type=int, default=10, help="number of sources to show")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"no trace database at {args.db}", file=sys.stderr)
        return 1
    with TraceStore(args.db) as store:
        runs = store.runs()
        if args.list:
            for r in runs:
                print(f"{r['id']}  {r['created_at']}  {r['label'] or ''}")
            return 0
        if not runs:
            print(f"no runs in {args.db}", file=sys.stderr)
            return 1
        run_id = args.run or runs[-1]["id"]
        profile = build_profile(store, run_id)
        print(profile.to_markdown(args.top) if args.format == "md" else profile.to_json(args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

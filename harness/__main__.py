"""`python -m harness <command>`: serve | spend | ab | profile | eval."""

from __future__ import annotations

import sys

USAGE = """usage: python -m harness <command> [options]

commands:
  serve     run the harness as a local OpenAI-compatible endpoint in front of your provider
  spend     show what's been spent, in total, today and per conversation
  ab        compare cost with and without policies after serve --ab
  profile   show where a traced run's money went (same as python -m harness.profiler)
  eval      pre-registered evaluation: validate | estimate | run | certify
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "serve":
        from harness.serve import main as run
    elif command == "spend":
        from harness.spend import main as run
    elif command == "ab":
        from harness.ab import main as run
    elif command == "profile":
        from harness.profiler.__main__ import main as run
    elif command == "eval":
        from harness.eval.__main__ import main as run
    else:
        print(f"unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())

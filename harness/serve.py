"""Run the harness as a local server that looks like the OpenAI API, in front of your provider.

    python -m harness serve --base-url https://api.example.com/v1 --policies cache,handles,mask [--max-per-day 5]

Point any agent at `http://127.0.0.1:<port>/s/<conversation-key>/v1` as its OpenAI base URL. Each key
gets its own session and its own trace, ready for `python -m harness profile`. If there's no key in the
path, the session gets picked by how the conversation starts.

It only listens on 127.0.0.1 unless you set HARNESS_HOST (the Docker image does). Your provider key stays in this process (read from `--api-key-env`), so
clients can send any placeholder key. `--simulate` uses the fake provider instead.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any

from harness.ab import CONTROL, assign
from harness.adapters.base import Adapter
from harness.budget import Budget, SpendLimit
from harness.eval.protocol import POLICIES
from harness.eval.proxy import ProxyServer
from harness.policies import Policy
from harness.pricing import PriceTable
from harness.session import Session
from harness.trace import TraceStore

ALIASES = {"cache": "cache_layout", "handles": "output_handles", "mask": "masking", "reason": "reasoning"}
# Cache layout goes first. Output handles go before masking so masked placeholders can point at stored outputs.
ORDER = ["cache_layout", "output_handles", "masking", "reasoning"]


class ServeError(ValueError):
    pass


def parse_policies(spec: str, settings: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Turn `"cache,handles,mask"` and `["mask.keep_last=6"]` into an ordered list of (name, settings)."""
    names = []
    for raw in filter(None, (s.strip() for s in spec.split(","))):
        name = ALIASES.get(raw, raw)
        if name not in POLICIES:
            raise ServeError(f"unknown policy {raw!r}; choose from {sorted(ALIASES)} or {sorted(POLICIES)}")
        if name not in names:
            names.append(name)
    params: dict[str, dict[str, Any]] = {n: {} for n in names}
    for item in settings:
        key, sep, value = item.partition("=")
        policy, dot, field = key.partition(".")
        policy = ALIASES.get(policy, policy)
        if not sep or not dot or policy not in params:
            raise ServeError(f"--set expects <enabled policy>.<field>=<value>, got {item!r}")
        params[policy][field] = _value(value)
    chain = [(n, params[n]) for n in sorted(names, key=ORDER.index)]
    for name, kwargs in chain:  # fail now, not on the first request
        try:
            POLICIES[name](**kwargs)
        except TypeError as exc:
            raise ServeError(f"{name}: {exc}") from exc
    return chain


def _value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(text.lower(), text)


@dataclass
class Server:
    proxy: ProxyServer
    store: TraceStore
    budget: Budget | None
    chain: list[tuple[str, dict[str, Any]]]

    def stop(self) -> None:
        self.proxy.stop()
        self.store.close()


def build_server(
    adapter: Adapter,
    *,
    chain: list[tuple[str, dict[str, Any]]],
    db: str = "runs/serve.sqlite",
    port: int = 8787,
    budget_usd: float | None = None,
    limit: SpendLimit | None = None,
    prices: PriceTable | None = None,
    label: str = "serve",
    ab: float | None = None,
    host: str = "127.0.0.1",
) -> Server:
    store = TraceStore(db)
    budget = Budget(budget_usd) if budget_usd else None
    prices = prices or PriceTable.load()

    def new_session(key: str) -> Session:
        group = assign(key, ab) if ab is not None else None
        active = [] if group == CONTROL else chain
        policies: list[Policy] = [POLICIES[name](**kwargs) for name, kwargs in active]
        meta: dict[str, Any] = {"serve": True, "key": key, "policies": [n for n, _ in active]}
        if group is not None:
            meta["ab"] = group
        run_id = store.start_run(f"{label}/{group}/{key}" if group else f"{label}/{key}", meta)
        return Session(adapter, task_id=key, policies=policies, store=store, run_id=run_id, prices=prices, budget=budget, limit=limit)

    return Server(ProxyServer(new_session, host=host, port=port).start(), store, budget, chain)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", default="cache,handles,mask", help="comma-separated: cache, handles, mask, reason (or full names). Use '' for none")
    parser.add_argument("--set", action="append", default=[], metavar="POLICY.FIELD=VALUE", help="policy setting, e.g. mask.keep_last=6 (repeatable)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--db", default="runs/serve.sqlite", help="trace database")
    parser.add_argument("--budget", type=float, help="cap for this server run only, in USD. Resets when you restart")
    parser.add_argument("--max-total", type=float, help="cap on everything ever spent in this --db, in USD. Survives restarts")
    parser.add_argument("--max-per-day", type=float, help="cap on what's spent since midnight, in USD")
    parser.add_argument("--max-per-conversation", type=float, help="cap on one conversation key, in USD")
    parser.add_argument("--base-url", help="your provider's OpenAI-compatible base URL, like https://api.example.com/v1")
    parser.add_argument("--api-key-env", default="HARNESS_API_KEY", help="env var that holds your provider key")
    parser.add_argument("--provider", default="example", help="provider name to look up in prices.yaml")
    parser.add_argument("--ab", nargs="?", type=float, const=0.5, metavar="SHARE", help="A/B test: send this share of conversations (default 0.5) through the policies and the rest without. Compare with python -m harness ab")
    parser.add_argument("--muse", action="store_true", help="use Meta's Model API for Muse (sets the base URL, provider and Muse's request rules)")
    parser.add_argument("--simulate", action="store_true", help="answer with the offline simulator instead of a real provider")
    args = parser.parse_args(argv)

    try:
        chain = parse_policies(args.policies, args.set)
        if args.ab is not None and not (0 < args.ab < 1):
            raise ServeError("--ab share must be between 0 and 1")
        if args.ab is not None and not chain:
            raise ServeError("--ab needs at least one policy to compare against")
        limit = SpendLimit(args.max_total, args.max_per_day, args.max_per_conversation)
        if args.simulate:
            from harness.sim import SimulatedAdapter

            adapter: Adapter = SimulatedAdapter(_simulated_reply)
        else:
            if args.muse:
                from harness.adapters import MuseAdapter

                adapter = MuseAdapter(args.base_url or "https://api.meta.ai/v1", api_key_env=args.api_key_env)
            elif not args.base_url:
                raise ServeError("--base-url is required unless you use --muse or --simulate")
            else:
                from harness.adapters import OpenAICompatAdapter

                adapter = OpenAICompatAdapter(args.base_url, api_key_env=args.api_key_env, provider=args.provider)
    except (ServeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    server = build_server(adapter, chain=chain, db=args.db, port=args.port, budget_usd=args.budget, limit=limit, ab=args.ab, host=os.environ.get("HARNESS_HOST", "127.0.0.1"))
    names = ", ".join(n for n, _ in chain) or "none"
    print(f"harness serving on {server.proxy.base_url}  (policies: {names}{'; SIMULATED' if args.simulate else ''})")
    print(f"  OpenAI base URL per conversation: {server.proxy.base_url}/s/<key>/v1")
    print(f"  limits: {_describe(limit, args.budget)}")
    if args.ab is not None:
        print(f"  A/B test: {args.ab:.0%} of conversations get the policies  ->  python -m harness ab --db {args.db}")
    print(f"  traces: {args.db}  ->  python -m harness.profiler --db {args.db} --list")
    print("  Ctrl+C to stop", flush=True)  # flush so anything reading through a pipe sees it right away

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        while not stop.wait(0.5):
            pass
    finally:
        spent = f", ${server.budget.spent_usd:.4f} spent" if server.budget else ""
        server.stop()
        print(f"stopped{spent}")
    return 0


def _describe(limit: SpendLimit, budget: float | None) -> str:
    parts = []
    if budget:
        parts.append(f"${budget:g} this run")
    if limit.total_usd:
        parts.append(f"${limit.total_usd:g} total")
    if limit.daily_usd:
        parts.append(f"${limit.daily_usd:g} per day")
    if limit.per_conversation_usd:
        parts.append(f"${limit.per_conversation_usd:g} per conversation")
    return ", ".join(parts) or "none (see python -m harness spend)"


def _simulated_reply(protocol: str, request: dict[str, Any]) -> dict[str, Any]:
    if protocol == "responses":
        return {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "simulated reply"}]}]}
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "simulated reply"}, "finish_reason": "stop"}]}


if __name__ == "__main__":
    raise SystemExit(main())

"""A live page for an A/B test: cost per conversation in each half, updating as traffic comes in.

    python -m harness dashboard --db runs/serve.sqlite      # watch a real `serve --ab`
    python -m harness dashboard --demo                      # fake traffic, no key, nothing spent

Open http://127.0.0.1:8788. The page asks for fresh numbers every few seconds, so leave it open next to
your agent. `--demo` runs the A/B server with the offline simulator and sends it fake conversations
whose system prompt has a ticking clock, so the control half keeps missing the cache. Its numbers show
how the page works, not what you'll save.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from harness.ab import CONTROL, MIN_PER_GROUP, TREATMENT, Group, compare
from harness.trace import TraceStore

STATIC = Path(__file__).parent / "static"
FILES = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}


def snapshot(store: TraceStore, *, simulated: bool = False, reps: int = 2000) -> dict[str, Any]:
    """Everything the page shows, as JSON. Conversations are in the order they started."""
    groups = {name: Group(name, [], 0, 0, 0) for name in (CONTROL, TREATMENT)}
    for run in store.runs():
        name = json.loads(run["meta"] or "{}").get("ab")
        if name not in groups:
            continue
        s = store.summary(run["id"])
        if not s["calls"]:
            continue
        g = groups[name]
        g.costs.append(s["cost_usd"])
        g.calls += s["calls"]
        g.input_tokens += s["input_tokens"]
        g.cached_input_tokens += s["cached_input_tokens"]
    cmp = compare(groups[CONTROL], groups[TREATMENT], reps=reps)

    def describe(g: Group) -> dict[str, Any]:
        running, total = [], 0.0
        for i, cost in enumerate(g.costs, 1):
            total += cost
            running.append(total / i)
        return {"n": g.n, "calls": g.calls, "mean": g.mean, "total": sum(g.costs), "cache_hit": g.cache_hit_rate, "running_mean": running}

    return {
        "simulated": simulated,
        "min_per_group": MIN_PER_GROUP,
        "groups": {CONTROL: describe(groups[CONTROL]), TREATMENT: describe(groups[TREATMENT])},
        "change": cmp.change,
        "ci": list(cmp.ci) if cmp.ci else None,
        "spent": store.spent(),
    }


def build(db: str, *, port: int = 8788, simulated: bool = False, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    store = TraceStore(db)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/api/ab":
                return self._send(200, "application/json", json.dumps(snapshot(store, simulated=simulated)).encode())
            if path in FILES:
                name, kind = FILES[path]
                return self._send(200, f"{kind}; charset=utf-8", (STATIC / name).read_bytes())
            self._send(404, "text/plain", b"not found")

        def _send(self, status: int, kind: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    server.store = store  # type: ignore[attr-defined]
    return server


DEMO_SYSTEM = "You are a support agent for an online store.\nCurrent time: {clock}\n" + "Follow the store policy exactly. " * 200


def demo_traffic(base_url: str, conversations: int, delay: float, stop: threading.Event) -> None:
    """Send scripted conversations through the A/B server. The clock line changes on every call."""
    tick = 0
    with httpx.Client(timeout=30) as client:
        for i in range(conversations):
            messages = [{"role": "system", "content": ""}, {"role": "user", "content": "Where is my order?"}]
            for turn in range(4):
                if stop.is_set():
                    return
                tick += 1
                messages[0]["content"] = DEMO_SYSTEM.format(clock=f"2026-10-28T14:{tick // 60 % 60:02d}:{tick % 60:02d}Z")
                r = client.post(f"{base_url}/s/demo-{i:03d}/v1/chat/completions", json={"model": "example-model", "messages": messages})
                messages += [r.json()["choices"][0]["message"], {"role": "user", "content": "Okay, what next?"}]
            stop.wait(delay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness dashboard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="runs/serve.sqlite", help="trace database that `serve --ab` writes to")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--demo", action="store_true", help="run a simulated A/B test with fake traffic. No key, nothing spent")
    parser.add_argument("--demo-conversations", type=int, default=80)
    parser.add_argument("--demo-delay", type=float, default=0.4, help="seconds between fake conversations")
    args = parser.parse_args(argv)

    stop = threading.Event()
    ab_server = None
    if args.demo:
        from harness.serve import _simulated_reply, build_server, parse_policies
        from harness.sim import SimulatedAdapter

        args.db = "runs/demo.sqlite"
        Path(args.db).unlink(missing_ok=True)
        ab_server = build_server(SimulatedAdapter(_simulated_reply), chain=parse_policies("cache", []), db=args.db, port=0, ab=0.5)
        threading.Thread(target=demo_traffic, args=(ab_server.proxy.base_url, args.demo_conversations, args.demo_delay, stop), daemon=True).start()
    elif not Path(args.db).exists():
        print(f"no trace database at {args.db}. Start `python -m harness serve --ab` first, or try --demo", file=sys.stderr)
        return 1

    server = build(args.db, port=args.port, simulated=args.demo, host=os.environ.get("HARNESS_HOST", "127.0.0.1"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"dashboard on http://localhost:{server.server_port}  ({'SIMULATED demo' if args.demo else args.db})")
    print("  Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.shutdown()
        server.store.close()  # type: ignore[attr-defined]
        if ab_server is not None:
            ab_server.stop()
    return 0

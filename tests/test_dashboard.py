import threading

import httpx

from harness.ab import CONTROL, TREATMENT
from harness.dashboard import build, demo_traffic, snapshot
from harness.pricing import PriceTable
from harness.serve import _simulated_reply, build_server, parse_policies
from harness.sim import SimulatedAdapter
from harness.trace import TraceStore

PRICES = PriceTable({"example": {"models": {"example-model": [{"effective_from": "2026-01-01", "input": 1.25, "cached_input": 0.15, "output": 4.25}]}}})


def ab_server(tmp_path):
    return build_server(SimulatedAdapter(_simulated_reply), chain=parse_policies("cache", []), db=str(tmp_path / "s.sqlite"), port=0, prices=PRICES, ab=0.5)


def test_snapshot_empty(tmp_path):
    with TraceStore(tmp_path / "s.sqlite") as store:
        snap = snapshot(store)
    assert snap["change"] is None and snap["ci"] is None
    assert snap["groups"][CONTROL]["n"] == 0 and snap["groups"][TREATMENT]["running_mean"] == []


def test_demo_traffic_makes_control_miss_the_cache(tmp_path):
    s = ab_server(tmp_path)
    try:
        demo_traffic(s.proxy.base_url, conversations=12, delay=0, stop=threading.Event())
        snap = snapshot(s.store, simulated=True, reps=500)
    finally:
        s.stop()
    control, policies = snap["groups"][CONTROL], snap["groups"][TREATMENT]
    assert control["n"] + policies["n"] == 12 and control["n"] and policies["n"]
    assert policies["cache_hit"] > control["cache_hit"]
    assert snap["change"] < 0 and snap["ci"][1] < 0
    assert len(policies["running_mean"]) == policies["n"]
    assert snap["simulated"] is True


def test_page_and_api_are_served(tmp_path):
    s = ab_server(tmp_path)
    s.stop()
    server = build(str(tmp_path / "s.sqlite"), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = httpx.get(base + "/")
        assert page.status_code == 200 and "Agent cost A/B" in page.text
        assert httpx.get(base + "/app.js").headers["content-type"].startswith("text/javascript")
        assert httpx.get(base + "/style.css").status_code == 200
        assert httpx.get(base + "/api/ab").json()["groups"][CONTROL]["n"] == 0
        assert httpx.get(base + "/nope").status_code == 404
    finally:
        server.shutdown()
        server.store.close()

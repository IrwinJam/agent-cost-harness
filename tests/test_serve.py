import json
import subprocess
import sys
import threading

import httpx
import pytest

from harness.__main__ import main as harness_main
from harness.adapters import OpenAICompatAdapter
from harness.pricing import PriceTable
from harness.serve import ServeError, build_server, parse_policies
from harness.sim import SimulatedAdapter

PRICES = PriceTable({"example": {"models": {"example-model": [{"effective_from": "2026-01-01", "input": 1.25, "cached_input": 0.15, "output": 4.25}]}}})
BODY = {
    "model": "example-model",
    "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}],
    "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
}


def reply(protocol, request):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}


@pytest.fixture
def server(tmp_path):
    seen = []

    def recording(protocol, request):
        seen.append(request)
        return reply(protocol, request)

    s = build_server(SimulatedAdapter(recording), chain=parse_policies("cache,handles,mask", []), db=str(tmp_path / "s.sqlite"), port=0, prices=PRICES)
    s.seen = seen
    yield s
    s.stop()


# ---------------- policy parsing ----------------


def test_parse_policies_orders_and_aliases():
    assert [n for n, _ in parse_policies("mask,cache,handles", [])] == ["cache_layout", "output_handles", "masking"]
    assert parse_policies("", []) == []
    assert parse_policies("mask,masking", ["mask.keep_last=6", "mask.batch=2"]) == [("masking", {"keep_last": 6, "batch": 2})]
    assert parse_policies("cache", ["cache.set_cache_key=false"]) == [("cache_layout", {"set_cache_key": False})]


@pytest.mark.parametrize(
    "spec,settings",
    [("nope", []), ("cache", ["mask.keep_last=3"]), ("mask", ["keep_last=3"]), ("mask", ["mask.bogus=1"]), ("mask", ["mask.batch=0"])],
)
def test_parse_policies_rejects(spec, settings):
    with pytest.raises((ServeError, ValueError)):
        parse_policies(spec, settings)


# ---------------- serving ----------------


def test_serves_chat_completions_with_policies_and_traces(server):
    url = server.proxy.base_url + "/s/conv-1/v1/chat/completions"
    r = httpx.post(url, json=BODY, headers={"Authorization": "Bearer anything"})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"

    sent = server.seen[-1]
    assert sent["prompt_cache_key"] == "conv-1"  # cache layout
    assert [t["function"]["name"] for t in sent["tools"]] == ["lookup", "fetch_output"]  # output handles

    runs = server.store.runs()
    assert [r["label"] for r in runs] == ["serve/conv-1"]
    assert json.loads(runs[0]["meta"])["policies"] == ["cache_layout", "output_handles", "masking"]
    assert server.store.summary(runs[0]["id"])["calls"] == 1


def test_each_key_is_its_own_session(server):
    for key in ("a", "b", "a"):
        httpx.post(server.proxy.base_url + f"/s/{key}/v1/chat/completions", json=BODY)
    labels = sorted(r["label"] for r in server.store.runs())
    assert labels == ["serve/a", "serve/b"]
    calls = {r["label"]: server.store.summary(r["id"])["calls"] for r in server.store.runs()}
    assert calls == {"serve/a": 2, "serve/b": 1}


def test_responses_protocol_and_unkeyed_route(server):
    body = {"model": "example-model", "instructions": "Be brief.", "input": "hi"}
    server.proxy._factory  # noqa: B018 - server exists
    r = httpx.post(server.proxy.base_url + "/v1/responses", json=body)
    assert r.status_code == 200
    assert server.seen[-1]["prompt_cache_retention"] == "24h"  # Responses-only cache layout setting
    assert server.store.runs()[0]["label"].startswith("serve/h")  # key derived from the opening


def test_budget_returns_429(tmp_path):
    s = build_server(SimulatedAdapter(reply), chain=[], db=str(tmp_path / "b.sqlite"), port=0, budget_usd=1e-9, prices=PRICES)
    try:
        url = s.proxy.base_url + "/s/x/v1/chat/completions"
        assert httpx.post(url, json=BODY).status_code == 200  # first call allowed, then over budget
        r = httpx.post(url, json=BODY)
        assert r.status_code == 429 and r.json()["error"]["type"] == "harness_budget_exceeded"
    finally:
        s.stop()


def test_real_adapter_behind_server(tmp_path):
    seen = {}

    def provider(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"model": "example-model", **reply("chat_completions", {}), "usage": {"prompt_tokens": 20, "completion_tokens": 3}})

    adapter = OpenAICompatAdapter("https://api.example.com/v1", api_key="real-key", client=httpx.Client(transport=httpx.MockTransport(provider)))
    s = build_server(adapter, chain=parse_policies("cache", []), db=str(tmp_path / "m.sqlite"), port=0, prices=PRICES)
    try:
        r = httpx.post(s.proxy.base_url + "/s/k/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer placeholder"})
        assert r.status_code == 200 and r.json()["usage"]["prompt_tokens"] == 20
        bad = httpx.post(s.proxy.base_url + "/s/k/v1/chat/completions", json={**BODY, "stream": True})
        assert bad.status_code == 400  # blocked before any network call
    finally:
        s.stop()
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer real-key"  # the server's key, never the client's placeholder


def test_serve_needs_base_url_without_simulate():
    assert harness_main(["serve"]) == 2


# ---------------- command line ----------------


def test_top_level_cli_dispatch(capsys):
    assert harness_main([]) == 2
    assert harness_main(["--help"]) == 0
    assert "serve" in capsys.readouterr().out
    assert harness_main(["nope"]) == 2
    assert harness_main(["serve", "--policies", "nope", "--simulate"]) == 2


def test_serve_process_starts_and_answers(tmp_path):
    """Start the real command in a subprocess (simulated backend) and talk to it."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "harness", "serve", "--simulate", "--port", "0", "--db", str(tmp_path / "p.sqlite")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    reader = threading.Thread(target=lambda: lines.append(proc.stdout.readline()), daemon=True)
    reader.start()
    try:
        reader.join(timeout=30)
        assert lines, "serve printed nothing within 30s"
        first = lines[0]
        assert first.startswith("harness serving on http://127.0.0.1:"), first
        base = first.split()[3]
        r = httpx.post(base + "/s/demo/v1/chat/completions", json=BODY, timeout=10)
        assert r.json()["choices"][0]["message"]["content"] == "simulated reply"
    finally:
        proc.terminate()
        proc.wait(timeout=10)

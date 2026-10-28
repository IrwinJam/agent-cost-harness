import json

import httpx
import pytest

from harness.adapters import MuseAdapter, RequestRejected
from harness.pricing import PriceTable
from harness.serve import main as serve_main


def adapter(seen):
    def handle(request):
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"model": "muse-spark-1.3", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    return MuseAdapter(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handle)))


def test_sends_to_meta_with_meta_provider():
    seen = []
    result = adapter(seen).call("chat_completions", {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "hi"}], "tool_choice": "auto", "reasoning_effort": "low"})
    assert seen[0][0] == "https://api.meta.ai/v1/chat/completions"
    assert result.provider == "meta"


@pytest.mark.parametrize(
    "protocol,body",
    [
        ("chat_completions", {"tool_choice": "required"}),
        ("chat_completions", {"reasoning_effort": "none"}),
        ("responses", {"reasoning": {"effort": "none"}}),
        ("responses", {"previous_response_id": "resp_1"}),
    ],
)
def test_rejects_what_muse_rejects(protocol, body):
    seen = []
    base = {"model": "muse-spark-1.3", "messages": [], "input": []}
    with pytest.raises(RequestRejected):
        adapter(seen).call(protocol, {**base, **body})
    assert seen == []  # never reached the API


def test_muse_prices_ship_with_the_repo():
    price = PriceTable.load().lookup("meta", "muse-spark-1.3")
    assert (price.input, price.cached_input, price.output) == (1.25, 0.15, 4.25)
    assert price.verified is False


def test_serve_muse_needs_a_key(monkeypatch, capsys):
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    assert serve_main(["--muse"]) == 2
    assert "HARNESS_API_KEY" in capsys.readouterr().err

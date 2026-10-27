"""Local server that looks like the OpenAI API and sends requests through harness sessions.

This lets an existing benchmark or agent use the harness without code changes. Point its OpenAI base
URL here.

`/s/<key>/v1/chat/completions` and `/s/<key>/v1/responses` pick the session by key. Use these when you
can set a base URL per task.

`/v1/chat/completions` and `/v1/responses` pick the session by how the conversation starts. That works
without any setup, but two conversations running at the same time with the same opening would end up
sharing a session.

Only listens on 127.0.0.1. No streaming.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from harness.adapters.base import RequestRejected
from harness.budget import BudgetExceeded
from harness.policies import conv
from harness.session import Session

log = logging.getLogger(__name__)

_ROUTE = re.compile(r"^(?:/s/(?P<key>[A-Za-z0-9_.:-]+))?/v1/(?P<endpoint>chat/completions|responses)$")
_PROTOCOLS = {"chat/completions": "chat_completions", "responses": "responses"}


def opening_key(protocol: str, request: dict[str, Any]) -> str:
    items = conv.items(protocol, dict(request))  # type: ignore[arg-type]
    opening = [request.get("instructions")]
    for item in items:
        opening.append(item)
        if item.get("role") == "user":
            break
    return "h" + hashlib.sha256(json.dumps(opening, sort_keys=True).encode()).hexdigest()[:16]


class ProxyServer:
    def __init__(self, session_factory: Callable[[str], Session], host: str = "127.0.0.1", port: int = 0):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the proxy only binds to localhost")
        self._factory = session_factory
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def url_for(self, key: str) -> str:
        return f"{self.base_url}/s/{key}/v1"

    def register(self, key: str, session: Session) -> None:
        with self._lock:
            self._sessions[key] = session

    def unregister(self, key: str) -> None:
        with self._lock:
            self._sessions.pop(key, None)

    def session(self, key: str) -> Session:
        with self._lock:
            if key not in self._sessions:
                self._sessions[key] = self._factory(key)
            return self._sessions[key]

    def start(self) -> ProxyServer:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> ProxyServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug(fmt, *args)

            def do_POST(self) -> None:  # noqa: N802
                match = _ROUTE.match(self.path.split("?", 1)[0])
                if not match:
                    return self._send(404, {"error": {"message": f"unknown path {self.path}"}})
                protocol = _PROTOCOLS[match["endpoint"]]
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                except ValueError:
                    return self._send(400, {"error": {"message": "body is not JSON"}})
                if body.get("stream"):
                    return self._send(400, {"error": {"message": "streaming is not supported by the harness proxy"}})
                key = match["key"] or opening_key(protocol, body)
                try:
                    response = proxy.session(key).call(protocol, body)
                except RequestRejected as exc:
                    return self._send(400, {"error": {"message": str(exc), "type": "harness_rejected"}})
                except BudgetExceeded as exc:
                    return self._send(429, {"error": {"message": str(exc), "type": "harness_budget_exceeded"}})
                except httpx.HTTPStatusError as exc:
                    return self._send(exc.response.status_code, _json_or_text(exc.response))
                except Exception as exc:
                    log.exception("proxy error")
                    return self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                self._send(200, response)

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


def _json_or_text(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text}}

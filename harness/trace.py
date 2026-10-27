"""Local SQLite store for every call, what it cost, and its prompt pieces.

Traces have full prompts and tool outputs in them. They go in `runs/` by default, which git ignores.
Don't publish trace files without looking through them first.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from harness.adapters.base import CallResult
from harness.pricing import Cost
from harness.segments import Segment, segment

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    label TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    seq INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    protocol TEXT NOT NULL,
    model TEXT NOT NULL,
    request_id TEXT,
    latency_s REAL NOT NULL,
    attempts INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL,
    cost_uncached_input REAL NOT NULL,
    cost_cached_input REAL NOT NULL,
    cost_output REAL NOT NULL,
    price_effective_from TEXT NOT NULL,
    price_input REAL NOT NULL,
    price_cached_input REAL NOT NULL,
    price_output REAL NOT NULL,
    price_verified INTEGER NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE TABLE IF NOT EXISTS segments (
    call_id INTEGER NOT NULL REFERENCES calls(id),
    idx INTEGER NOT NULL,
    origin TEXT NOT NULL,
    ref TEXT,
    hash TEXT NOT NULL,
    chars INTEGER NOT NULL,
    PRIMARY KEY (call_id, idx)
);
CREATE INDEX IF NOT EXISTS calls_run ON calls(run_id, seq);
"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _locked(method):
    """One thread at a time. The proxy saves calls from several request threads into the same connection."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapper


class TraceStore:
    def __init__(self, path: str | Path = "runs/traces.sqlite"):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)

    @property
    def connection(self) -> sqlite3.Connection:
        """The shared connection, for code that keeps its own tables in the same file (like eval). Hold `lock` while using it."""
        return self._db

    @_locked
    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @_locked
    def start_run(self, label: str | None = None, meta: dict[str, Any] | None = None) -> str:
        run_id = uuid.uuid4().hex
        with self._db:
            self._db.execute(
                "INSERT INTO runs (id, created_at, label, meta) VALUES (?, ?, ?, ?)",
                (run_id, _now(), label, json.dumps(meta or {})),
            )
        return run_id

    @_locked
    def record(self, run_id: str, result: CallResult, cost: Cost) -> int:
        segments = segment(result.protocol, result.request)
        u = result.usage
        with self._db:
            seq = self._db.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM calls WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            cursor = self._db.execute(
                """INSERT INTO calls (run_id, seq, created_at, provider, protocol, model, request_id,
                       latency_s, attempts, input_tokens, cached_input_tokens, output_tokens,
                       reasoning_tokens, cost_uncached_input, cost_cached_input, cost_output,
                       price_effective_from, price_input, price_cached_input, price_output,
                       price_verified, request_json, response_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, seq, _now(), result.provider, result.protocol, result.model, result.request_id,
                    result.latency_s, result.attempts, u.input_tokens, u.cached_input_tokens,
                    u.output_tokens, u.reasoning_tokens, cost.uncached_input, cost.cached_input,
                    cost.output, cost.price.effective_from.isoformat(), cost.price.input,
                    cost.price.cached_input, cost.price.output, int(cost.price.verified),
                    json.dumps(result.request), json.dumps(result.response),
                ),
            )
            call_id = cursor.lastrowid
            self._db.executemany(
                "INSERT INTO segments (call_id, idx, origin, ref, hash, chars) VALUES (?, ?, ?, ?, ?, ?)",
                [(call_id, s.index, s.origin, s.ref, s.hash, s.chars) for s in segments],
            )
        return call_id

    @_locked
    def runs(self) -> list[sqlite3.Row]:
        return self._db.execute("SELECT * FROM runs ORDER BY created_at").fetchall()

    @_locked
    def run(self, run_id: str) -> sqlite3.Row | None:
        return self._db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    @_locked
    def calls(self, run_id: str) -> list[sqlite3.Row]:
        return self._db.execute("SELECT * FROM calls WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()

    @_locked
    def segments(self, call_id: int) -> list[Segment]:
        rows = self._db.execute(
            "SELECT idx, origin, ref, hash, chars FROM segments WHERE call_id = ? ORDER BY idx", (call_id,)
        ).fetchall()
        return [Segment(r["idx"], r["origin"], r["ref"], r["hash"], r["chars"]) for r in rows]

    @_locked
    def summary(self, run_id: str) -> dict[str, Any]:
        row = self._db.execute(
            """SELECT COUNT(*) AS calls,
                      COALESCE(SUM(input_tokens), 0) AS input_tokens,
                      COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                      COALESCE(SUM(output_tokens), 0) AS output_tokens,
                      COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                      COALESCE(SUM(cost_uncached_input + cost_cached_input + cost_output), 0) AS cost_usd,
                      COALESCE(MIN(price_verified), 1) AS all_prices_verified
               FROM calls WHERE run_id = ?""",
            (run_id,),
        ).fetchone()
        summary = dict(row)
        summary["cache_hit_rate"] = (
            summary["cached_input_tokens"] / summary["input_tokens"] if summary["input_tokens"] else 0.0
        )
        summary["all_prices_verified"] = bool(summary["all_prices_verified"])
        return summary

    @_locked
    def spent(self, *, run_id: str | None = None, since: dt.datetime | None = None) -> float:
        """Total dollars spent on saved calls. Narrow it to one run, or to calls after `since`."""
        sql = "SELECT COALESCE(SUM(cost_uncached_input + cost_cached_input + cost_output), 0) FROM calls WHERE 1=1"
        args: list[Any] = []
        if run_id is not None:
            sql += " AND run_id = ?"
            args.append(run_id)
        if since is not None:
            sql += " AND julianday(created_at) >= julianday(?)"
            args.append(since.astimezone(dt.timezone.utc).isoformat())
        return float(self._db.execute(sql, args).fetchone()[0])

    @_locked
    def spend_by_run(self, limit: int = 10) -> list[dict[str, Any]]:
        """The most expensive runs, with their label, call count and cost."""
        rows = self._db.execute(
            """SELECT r.id, r.label, COUNT(c.id) AS calls,
                      COALESCE(SUM(c.cost_uncached_input + c.cost_cached_input + c.cost_output), 0) AS cost_usd
               FROM runs r LEFT JOIN calls c ON c.run_id = r.id
               GROUP BY r.id ORDER BY cost_usd DESC, r.created_at LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    @_locked
    def export_jsonl(self, run_id: str, path: str | Path) -> int:
        rows = self.calls(run_id)
        with Path(path).open("w", encoding="utf-8") as f:
            for row in rows:
                record = dict(row)
                record["request"] = json.loads(record.pop("request_json"))
                record["response"] = json.loads(record.pop("response_json"))
                record["segments"] = [s.__dict__ for s in self.segments(row["id"])]
                f.write(json.dumps(record) + "\n")
        return len(rows)

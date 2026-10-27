"""Eval results, saved in the same SQLite file as the traces.

One row per eval, setup, task and trial. Each trial's calls are their own trace run, so any trial can be
profiled by itself.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass
from typing import Any

from harness.trace import TraceStore, _locked

SCHEMA = """
CREATE TABLE IF NOT EXISTS evals (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    protocol_json TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS eval_trials (
    eval_id TEXT NOT NULL REFERENCES evals(id),
    arm TEXT NOT NULL,
    task_id TEXT NOT NULL,
    trial INTEGER NOT NULL,
    run_id TEXT REFERENCES runs(id),
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    success INTEGER,
    reward REAL,
    agent_cost_usd REAL NOT NULL,
    sim_cost_usd REAL NOT NULL,
    error TEXT,
    finished_at TEXT NOT NULL,
    PRIMARY KEY (eval_id, arm, task_id, trial)
);
"""


@dataclass(frozen=True)
class TrialRow:
    arm: str
    task_id: str
    trial: int
    run_id: str | None
    status: str
    success: bool | None
    reward: float | None
    agent_cost_usd: float
    sim_cost_usd: float
    error: str | None


class EvalResults:
    def __init__(self, store: TraceStore):
        self.store = store
        self.lock = store.lock
        self._db = store.connection
        with self.lock:
            self._db.executescript(SCHEMA)

    @_locked
    def start(self, protocol_hash: str, protocol: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
        eval_id = uuid.uuid4().hex[:12]
        with self._db:
            self._db.execute(
                "INSERT INTO evals (id, created_at, protocol_hash, protocol_json, meta) VALUES (?, ?, ?, ?, ?)",
                (eval_id, dt.datetime.now(dt.timezone.utc).isoformat(), protocol_hash, json.dumps(protocol), json.dumps(meta or {})),
            )
        return eval_id

    @_locked
    def eval(self, eval_id: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM evals WHERE id = ?", (eval_id,)).fetchone()
        if row is None:
            raise KeyError(f"no eval {eval_id}")
        out = dict(row)
        out["protocol"] = json.loads(out.pop("protocol_json"))
        out["meta"] = json.loads(out["meta"])
        return out

    @_locked
    def evals(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT id, created_at, protocol_hash FROM evals ORDER BY created_at")]

    @_locked
    def record(self, eval_id: str, row: TrialRow) -> None:
        with self._db:
            self._db.execute(
                """INSERT OR REPLACE INTO eval_trials (eval_id, arm, task_id, trial, run_id, status, success, reward,
                       agent_cost_usd, sim_cost_usd, error, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    eval_id, row.arm, row.task_id, row.trial, row.run_id, row.status,
                    None if row.success is None else int(row.success), row.reward,
                    row.agent_cost_usd, row.sim_cost_usd, row.error, dt.datetime.now(dt.timezone.utc).isoformat(),
                ),
            )

    @_locked
    def trials(self, eval_id: str) -> list[TrialRow]:
        rows = self._db.execute(
            "SELECT * FROM eval_trials WHERE eval_id = ? ORDER BY task_id, trial, arm", (eval_id,)
        ).fetchall()
        return [
            TrialRow(
                r["arm"], r["task_id"], r["trial"], r["run_id"], r["status"],
                None if r["success"] is None else bool(r["success"]), r["reward"],
                r["agent_cost_usd"], r["sim_cost_usd"], r["error"],
            )
            for r in rows
        ]

    @_locked
    def done(self, eval_id: str) -> set[tuple[str, str, int]]:
        """Trials that finished without errors. Errored ones get retried when you resume."""
        rows = self._db.execute(
            "SELECT arm, task_id, trial FROM eval_trials WHERE eval_id = ? AND status = 'ok'", (eval_id,)
        ).fetchall()
        return {(r["arm"], r["task_id"], r["trial"]) for r in rows}

    @_locked
    def spent(self, eval_id: str) -> float:
        """Everything this eval has cost so far, including failed or interrupted trials.

        Agent cost comes from the saved calls, so a trial that errored and got retried still counts. Simulated
        user and judge cost comes from the trial rows.
        """
        agent = self._db.execute(
            """SELECT COALESCE(SUM(c.cost_uncached_input + c.cost_cached_input + c.cost_output), 0)
               FROM calls c JOIN runs r ON r.id = c.run_id
               WHERE json_extract(r.meta, '$.eval_id') = ?""",
            (eval_id,),
        ).fetchone()[0]
        sim = self._db.execute(
            "SELECT COALESCE(SUM(sim_cost_usd), 0) FROM eval_trials WHERE eval_id = ?", (eval_id,)
        ).fetchone()[0]
        return float(agent) + float(sim)

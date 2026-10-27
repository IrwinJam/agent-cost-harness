"""The test plan: load it, check it, fingerprint it, and build each setup's policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.policies import BatchedMasking, CacheStableLayout, OutputHandles, Policy

POLICIES: dict[str, type[Policy]] = {
    "cache_layout": CacheStableLayout,
    "masking": BatchedMasking,
    "output_handles": OutputHandles,
}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Protocol:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> Protocol:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Protocol:
        protocol = cls(data)
        protocol.validate()
        return protocol

    @property
    def hash(self) -> str:
        """Fingerprint of the content. Comments and formatting don't change it."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def trials(self) -> int:
        return int(self.raw["trials"])

    @property
    def arms(self) -> list[str]:
        return list(self.raw["arms"])

    @property
    def baseline(self) -> str:
        return self.raw["baseline"]

    @property
    def primary(self) -> str:
        return self.raw["primary"]

    @property
    def agent(self) -> dict[str, Any]:
        return self.raw["agent"]

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw["analysis"]

    @property
    def budget_usd(self) -> float:
        return float(self.raw["budget_usd"])

    def task_ids(self, available: list[str]) -> list[str]:
        wanted = self.raw["benchmark"].get("tasks", "all")
        if wanted == "all":
            return list(available)
        missing = [t for t in wanted if t not in available]
        if missing:
            raise ProtocolError(f"tasks not in benchmark: {missing}")
        return [str(t) for t in wanted]

    def build_policies(self, arm: str) -> list[Policy]:
        chain = []
        for spec in self.raw["arms"][arm] or []:
            params = dict(spec)
            chain.append(POLICIES[params.pop("policy")](**params))
        return chain

    def validate(self) -> None:
        d = self.raw
        for key in ("name", "benchmark", "agent", "trials", "baseline", "primary", "arms", "analysis", "budget_usd"):
            if key not in d:
                raise ProtocolError(f"protocol is missing {key!r}")
        if int(d["trials"]) < 1:
            raise ProtocolError("trials must be >= 1")
        for role in ("baseline", "primary"):
            if d[role] not in d["arms"]:
                raise ProtocolError(f"{role} arm {d[role]!r} is not defined in arms")
        if d["baseline"] == d["primary"]:
            raise ProtocolError("primary must differ from baseline")
        for arm, chain in d["arms"].items():
            for spec in chain or []:
                if spec.get("policy") not in POLICIES:
                    raise ProtocolError(f"arm {arm!r}: unknown policy {spec.get('policy')!r}")
            try:
                self.build_policies(arm)
            except TypeError as exc:
                raise ProtocolError(f"arm {arm!r}: {exc}") from exc
        a = d["analysis"]
        if not 0 < float(a.get("alpha", 0)) < 0.5:
            raise ProtocolError("analysis.alpha must be in (0, 0.5)")
        if float(a.get("noninferiority_margin_pp", -1)) < 0:
            raise ProtocolError("analysis.noninferiority_margin_pp must be >= 0")
        if int(a.get("bootstrap_reps", 0)) < 1000:
            raise ProtocolError("analysis.bootstrap_reps must be >= 1000")
        if float(d["budget_usd"]) <= 0:
            raise ProtocolError("budget_usd must be > 0")

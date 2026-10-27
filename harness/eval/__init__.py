"""Pre-registered, paired cost/quality evaluation."""

from harness.eval.benchmark import Benchmark, Outcome
from harness.eval.certificate import build_certificate, to_markdown
from harness.eval.protocol import Protocol, ProtocolError
from harness.eval.results import EvalResults, TrialRow
from harness.eval.runner import EvalStopped, run_eval
from harness.eval.stats import compare, summarize_arm

__all__ = [
    "Benchmark",
    "EvalResults",
    "EvalStopped",
    "Outcome",
    "Protocol",
    "ProtocolError",
    "TrialRow",
    "build_certificate",
    "compare",
    "run_eval",
    "summarize_arm",
    "to_markdown",
]

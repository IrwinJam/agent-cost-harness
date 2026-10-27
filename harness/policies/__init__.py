from harness.policies.base import HarnessTool, Policy, SessionContext
from harness.policies.cache_layout import CacheStableLayout
from harness.policies.masking import BatchedMasking
from harness.policies.output_handles import OutputHandles

__all__ = ["BatchedMasking", "CacheStableLayout", "HarnessTool", "OutputHandles", "Policy", "SessionContext"]


def default_chain() -> list[Policy]:
    """Cache layout first, then output handles, then masking, so masked placeholders can point at stored outputs."""
    return [CacheStableLayout(), OutputHandles(), BatchedMasking()]

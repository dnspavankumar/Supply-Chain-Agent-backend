from __future__ import annotations

from typing import Any

try:
    from agno.agent import Agent
except ImportError:  # pragma: no cover
    Agent = Any  # type: ignore[assignment]


def build_audit_agent() -> Agent:
    """Create the Audit agent shell.

    TODO: Implement agent instructions, tools, and runtime behavior.
    """
    raise NotImplementedError("Audit agent logic is not implemented yet.")

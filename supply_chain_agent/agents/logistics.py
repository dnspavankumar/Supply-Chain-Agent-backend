from __future__ import annotations

from typing import Any

try:
    from agno.agent import Agent
except ImportError:  # pragma: no cover
    Agent = Any  # type: ignore[assignment]


def build_logistics_agent() -> Agent:
    """Create the Logistics agent shell.

    TODO: Implement agent instructions, tools, and runtime behavior.
    """
    raise NotImplementedError("Logistics agent logic is not implemented yet.")

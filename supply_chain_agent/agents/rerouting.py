from __future__ import annotations

from typing import Any

try:
    from agno.agent import Agent
except ImportError:  # pragma: no cover
    Agent = Any  # type: ignore[assignment]


def build_rerouting_agent() -> Agent:
    """Create the Rerouting agent shell.

    TODO: Implement agent instructions, tools, and runtime behavior.
    """
    raise NotImplementedError("Rerouting agent logic is not implemented yet.")

from __future__ import annotations

from typing import Any


class ComplianceViolationError(Exception):
    """Raised when pipeline guardrails block execution."""

    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {"detail": message}


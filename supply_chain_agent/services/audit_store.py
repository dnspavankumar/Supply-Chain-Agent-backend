from __future__ import annotations

import asyncio
import json
from pathlib import Path

from supply_chain_agent.models.schemas import AgentDecisionLog


class AuditStore:
    """Append-only JSONL audit log writer."""

    def __init__(self, file_path: str | Path = "supply_chain_agent_audit.jsonl") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, log_entry: AgentDecisionLog) -> None:
        payload = json.dumps(log_entry.model_dump(mode="json"), ensure_ascii=False)
        async with self._lock:
            await asyncio.to_thread(self._append_line, payload)

    def _append_line(self, payload: str) -> None:
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

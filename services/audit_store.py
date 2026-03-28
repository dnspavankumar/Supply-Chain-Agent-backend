from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import ClassVar

from models.schemas import AgentDecisionLog


class AuditStore:
    """Singleton append-only JSONL audit log store."""

    _instance: ClassVar[AuditStore | None] = None
    _instance_lock: ClassVar[Lock] = Lock()

    def __new__(cls, file_path: str | Path = "audit.jsonl") -> AuditStore:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, file_path: str | Path = "audit.jsonl") -> None:
        if getattr(self, "_initialized", False):
            return

        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.touch(exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = True

    async def append(self, entry: AgentDecisionLog) -> None:
        payload = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False)
        async with self._lock:
            await asyncio.to_thread(self._append_line, payload)

    async def read_all(self) -> list[AgentDecisionLog]:
        async with self._lock:
            lines = await asyncio.to_thread(self._read_lines)
        return [self._parse_line(line) for line in lines if line.strip()]

    async def read_by_agent(self, agent_name: str) -> list[AgentDecisionLog]:
        normalized = agent_name.strip().lower()
        entries = await self.read_all()
        return [entry for entry in entries if entry.agent_name.strip().lower() == normalized]

    async def read_by_timerange(self, start: datetime, end: datetime) -> list[AgentDecisionLog]:
        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        if start_utc > end_utc:
            raise ValueError("start must be before or equal to end")

        entries = await self.read_all()
        return [
            entry
            for entry in entries
            if start_utc <= self._to_utc(entry.timestamp) <= end_utc
        ]

    def _append_line(self, payload: str) -> None:
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    def _read_lines(self) -> list[str]:
        if not self.file_path.exists():
            self.file_path.touch(exist_ok=True)
        with self.file_path.open("r", encoding="utf-8") as handle:
            return handle.readlines()

    def _parse_line(self, line: str) -> AgentDecisionLog:
        data = json.loads(line)
        return AgentDecisionLog.model_validate(data)

    def _to_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

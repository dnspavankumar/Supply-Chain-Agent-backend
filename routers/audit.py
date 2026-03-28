from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from agents.audit import AuditAgent
from models.schemas import AgentDecisionLog, AuditQueryRequest, AuditReport
from services.audit_store import AuditStore

router = APIRouter(prefix="/audit", tags=["audit"])
audit_store = AuditStore(file_path="audit.jsonl")
audit_agent = AuditAgent(audit_store=audit_store)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@router.get(
    "/logs",
    response_model=list[AgentDecisionLog],
    summary="List Audit Logs",
    description="Returns audit log entries with optional agent and date-range filtering.",
)
async def get_audit_logs(
    agent_name: str | None = Query(default=None),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
) -> list[AgentDecisionLog]:
    """Fetch audit logs filtered by agent name and/or timestamp window."""
    if date_from and date_to and _to_utc(date_from) > _to_utc(date_to):
        raise HTTPException(status_code=422, detail="date_from must be before or equal to date_to")

    try:
        start = _to_utc(date_from) if date_from else None
        end = _to_utc(date_to) if date_to else None

        if agent_name and date_from and date_to:
            entries = await audit_store.read_by_agent(agent_name)
            return [
                entry
                for entry in entries
                if start <= _to_utc(entry.timestamp) <= end
            ]
        if agent_name:
            entries = await audit_store.read_by_agent(agent_name)
            if start is not None:
                entries = [entry for entry in entries if _to_utc(entry.timestamp) >= start]
            if end is not None:
                entries = [entry for entry in entries if _to_utc(entry.timestamp) <= end]
            return entries
        if date_from and date_to:
            return await audit_store.read_by_timerange(date_from, date_to)

        entries = await audit_store.read_all()
        if start is not None:
            entries = [entry for entry in entries if _to_utc(entry.timestamp) >= start]
        if end is not None:
            entries = [entry for entry in entries if _to_utc(entry.timestamp) <= end]
        return entries
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/logs/export",
    summary="Export Raw Audit Log File",
    description="Downloads the append-only audit.jsonl file as an attachment.",
)
async def export_audit_logs() -> FileResponse:
    """Export the raw newline-delimited JSON audit log file."""
    await audit_store.read_all()
    return FileResponse(
        path=audit_store.file_path,
        media_type="application/x-ndjson",
        filename="audit.jsonl",
    )


@router.post(
    "/query",
    response_model=AuditReport,
    summary="Query Audit Report",
    description="Runs AuditAgent over filtered logs and returns a human-readable compliance report.",
)
async def query_audit_report(payload: AuditQueryRequest) -> AuditReport:
    """Generate an audit report from natural-language query and optional filters."""
    try:
        return await audit_agent.query(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

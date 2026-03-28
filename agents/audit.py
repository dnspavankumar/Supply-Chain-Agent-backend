from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timezone
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, ValidationError

from config import get_settings
from models.schemas import AgentDecisionLog, AuditQueryRequest, AuditReport
from services.audit_store import AuditStore

try:
    from agno.run import RunStatus
except Exception:  # pragma: no cover
    RunStatus = None  # type: ignore[assignment]


AUDIT_SYSTEM_PROMPT = """
You are AuditAgent for a supply-chain operations platform.

You receive:
- a natural-language audit query
- optional filters
- a list of audit decision logs

Your task:
1) Summarize what decisions were made in plain English.
2) Identify compliance issues from guardrail status, rationale, and confidence.
3) Provide practical recommendations.
4) Keep recommendations short and actionable.

Output contract:
- Return JSON only.
- Return valid JSON matching this schema exactly:
  {
    "summary": string,
    "decisions_reviewed": int,
    "compliance_issues": [string],
    "recommendations": [string]
  }
- No markdown, no additional keys.
""".strip()


class AuditAgent:
    def __init__(self, audit_store: AuditStore | None = None) -> None:
        settings = get_settings()
        self._audit_store = audit_store or AuditStore(file_path="audit.jsonl")

        self._schema_param_name: str | None = None
        self._run_schema_param_name: str | None = None

        model = Gemini(
            id=settings.gemini_model_id,
            api_key=settings.google_api_key or None,
        )
        self._agent = self._build_agent(model)

    def _build_agent(self, model: Gemini) -> Agent:
        init_params = set(inspect.signature(Agent.__init__).parameters)
        kwargs: dict[str, Any] = {
            "name": "AuditAgent",
            "model": model,
            "system_message": AUDIT_SYSTEM_PROMPT,
        }
        if "markdown" in init_params:
            kwargs["markdown"] = False
        if "structured_outputs" in init_params:
            kwargs["structured_outputs"] = True
        if "parse_response" in init_params:
            kwargs["parse_response"] = True
        if "use_json_mode" in init_params:
            kwargs["use_json_mode"] = False

        if "response_model" in init_params:
            kwargs["response_model"] = AuditReport
            self._schema_param_name = "response_model"
        elif "output_schema" in init_params:
            kwargs["output_schema"] = AuditReport
            self._schema_param_name = "output_schema"

        run_params = set(inspect.signature(Agent.run).parameters)
        if "response_model" in run_params:
            self._run_schema_param_name = "response_model"
        elif "output_schema" in run_params:
            self._run_schema_param_name = "output_schema"

        return Agent(**kwargs)

    async def query(self, request: AuditQueryRequest) -> AuditReport:
        entries = await self._load_entries(request)
        report = await self._generate_report(request, entries)
        await self._write_audit_log(request, report, len(entries))
        return report

    async def _load_entries(self, request: AuditQueryRequest) -> list[AgentDecisionLog]:
        if request.agent_name and request.date_from and request.date_to:
            entries = await self._audit_store.read_by_agent(request.agent_name)
            return self._filter_by_dates(entries, request.date_from, request.date_to)
        if request.agent_name:
            entries = await self._audit_store.read_by_agent(request.agent_name)
            return self._filter_partial_dates(entries, request.date_from, request.date_to)
        if request.date_from and request.date_to:
            return await self._audit_store.read_by_timerange(request.date_from, request.date_to)

        entries = await self._audit_store.read_all()
        return self._filter_partial_dates(entries, request.date_from, request.date_to)

    async def _generate_report(
        self,
        request: AuditQueryRequest,
        entries: list[AgentDecisionLog],
    ) -> AuditReport:
        max_entries = 250
        truncated = len(entries) > max_entries
        payload_entries = entries[-max_entries:] if truncated else entries

        prompt_payload = {
            "query": request.natural_language_query,
            "filters": {
                "agent_name": request.agent_name,
                "date_from": request.date_from.isoformat() if request.date_from else None,
                "date_to": request.date_to.isoformat() if request.date_to else None,
            },
            "entries_count_total": len(entries),
            "entries_count_provided_to_llm": len(payload_entries),
            "entries_truncated": truncated,
            "entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "agent_name": entry.agent_name,
                    "decision": entry.decision,
                    "rationale": entry.rationale,
                    "guardrail_status": entry.guardrail_status,
                    "guardrails_checked": entry.guardrails_checked,
                    "confidence": entry.confidence,
                }
                for entry in payload_entries
            ],
        }
        prompt = (
            "Generate an audit report from this payload:\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}\n"
            "Return only valid AuditReport JSON."
        )

        run_kwargs: dict[str, Any] = {}
        if self._schema_param_name is None and self._run_schema_param_name is not None:
            run_kwargs[self._run_schema_param_name] = AuditReport

        try:
            run_output = await asyncio.to_thread(self._agent.run, prompt, **run_kwargs)
            self._raise_if_run_error(run_output)
            report = self._coerce_report(self._extract_content(run_output))
            if report.decisions_reviewed == 0:
                return report.model_copy(update={"decisions_reviewed": len(entries)})
            return report
        except Exception:
            return self._fallback_report(request, entries)

    def _filter_by_dates(
        self,
        entries: list[AgentDecisionLog],
        date_from: datetime,
        date_to: datetime,
    ) -> list[AgentDecisionLog]:
        start = self._to_utc(date_from)
        end = self._to_utc(date_to)
        if start > end:
            raise ValueError("date_from must be before or equal to date_to")

        return [
            entry
            for entry in entries
            if start <= self._to_utc(entry.timestamp) <= end
        ]

    def _filter_partial_dates(
        self,
        entries: list[AgentDecisionLog],
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> list[AgentDecisionLog]:
        filtered = entries
        if date_from is not None:
            start = self._to_utc(date_from)
            filtered = [entry for entry in filtered if self._to_utc(entry.timestamp) >= start]
        if date_to is not None:
            end = self._to_utc(date_to)
            filtered = [entry for entry in filtered if self._to_utc(entry.timestamp) <= end]
        return filtered

    def _to_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _raise_if_run_error(self, run_output: Any) -> None:
        if not hasattr(run_output, "status"):
            return
        status = getattr(run_output, "status")
        if RunStatus is not None and status == RunStatus.error:
            message = str(getattr(run_output, "content", "") or "Unknown AuditAgent error.")
            raise RuntimeError(f"AuditAgent run failed: {message}")
        if "ERROR" in str(status).upper():
            message = str(getattr(run_output, "content", "") or "Unknown AuditAgent error.")
            raise RuntimeError(f"AuditAgent run failed: {message}")

    def _extract_content(self, run_output: Any) -> Any:
        if hasattr(run_output, "content"):
            return getattr(run_output, "content")
        return run_output

    def _coerce_report(self, content: Any) -> AuditReport:
        if isinstance(content, AuditReport):
            return content
        if isinstance(content, BaseModel):
            return AuditReport.model_validate(content.model_dump(mode="json"))
        if isinstance(content, dict):
            return AuditReport.model_validate(content)
        if isinstance(content, str):
            return self._parse_report_text(content)
        raise ValueError(f"Unsupported AuditAgent response type: {type(content)!r}")

    def _parse_report_text(self, text: str) -> AuditReport:
        stripped = text.strip()
        try:
            return AuditReport.model_validate_json(stripped)
        except ValidationError:
            payload = self._parse_json_object(stripped)
            return AuditReport.model_validate(payload)

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        candidate = text
        if candidate.startswith("```"):
            lines = [line for line in candidate.splitlines() if not line.strip().startswith("```")]
            candidate = "\n".join(lines).strip()

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("AuditAgent response is not valid JSON object output.")

    def _fallback_report(
        self,
        request: AuditQueryRequest,
        entries: list[AgentDecisionLog],
    ) -> AuditReport:
        fail_count = sum(1 for entry in entries if entry.guardrail_status == "FAIL")
        warn_count = sum(1 for entry in entries if entry.guardrail_status == "WARN")
        low_conf_count = sum(1 for entry in entries if entry.confidence < 0.6)

        issues: list[str] = []
        if fail_count > 0:
            issues.append(f"{fail_count} decisions are marked FAIL.")
        if warn_count > 0:
            issues.append(f"{warn_count} decisions are marked WARN.")
        if low_conf_count > 0:
            issues.append(f"{low_conf_count} decisions have confidence below 0.60.")
        if not issues:
            issues.append("No explicit compliance issues found in the filtered logs.")

        recommendations: list[str] = []
        if fail_count > 0:
            recommendations.append("Prioritize human review for FAIL decisions and document mitigations.")
        if warn_count > 0:
            recommendations.append("Review WARN decisions to confirm guardrail exceptions are justified.")
        if low_conf_count > 0:
            recommendations.append("Re-run low-confidence decisions with additional context or stricter checks.")
        if not recommendations:
            recommendations.append("Continue periodic audit sampling to maintain compliance posture.")

        summary = (
            f"Processed {len(entries)} audit decision(s) for query "
            f"'{request.natural_language_query}'."
        )
        return AuditReport(
            summary=summary,
            decisions_reviewed=len(entries),
            compliance_issues=issues,
            recommendations=recommendations,
        )

    async def _write_audit_log(
        self,
        request: AuditQueryRequest,
        report: AuditReport,
        entries_count: int,
    ) -> None:
        guardrail_status = "PASS"
        if report.compliance_issues and not (
            len(report.compliance_issues) == 1
            and report.compliance_issues[0] == "No explicit compliance issues found in the filtered logs."
        ):
            guardrail_status = "WARN"

        confidence = 0.8 if entries_count > 0 else 0.65
        log_entry = AgentDecisionLog(
            timestamp=datetime.now(timezone.utc),
            agent_name="AuditAgent",
            decision=f"Reviewed {entries_count} audit log entries",
            rationale=(
                f"Query: {request.natural_language_query}. "
                f"Summary: {report.summary}"
            ),
            guardrails_checked=[
                "audit_query_filtering",
                "decision_log_review",
                "compliance_issue_detection",
            ],
            guardrail_status=guardrail_status,  # type: ignore[arg-type]
            confidence=confidence,
        )
        await self._audit_store.append(log_entry)


def build_audit_agent() -> AuditAgent:
    return AuditAgent()

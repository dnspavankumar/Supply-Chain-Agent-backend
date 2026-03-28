from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from agno.run import RunStatus
from pydantic import BaseModel, ValidationError

from supply_chain_agent.config import get_settings
from supply_chain_agent.models.schemas import (
    AgentDecisionLog,
    PackagingRequest,
    PackagingResult,
)
from supply_chain_agent.services.audit_store import AuditStore

PACKAGING_SYSTEM_PROMPT = """
You are PackagingAgent for a supply-chain AI backend.

Task:
1) Recommend packaging materials for each goods item.
2) Estimate packaging cost per item in USD.
3) Enforce guardrails before finalizing recommendations.

Guardrails:
- Fragile items: if is_fragile=true, bubble wrap must be included as a layer.
  Single-wall cardboard only packaging is not allowed.
- Hazmat items: if is_hazmat=true, recommend hazmat-rated packaging only.
- Cost budget guardrail:
  Assume goods value proxy = weight_kg * 100 USD.
  If estimated packaging cost exceeds 15% of proxy value, add a cost flag.

Output contract:
- Return JSON that strictly matches the PackagingResult schema.
- items length must match number of goods.
- Use goods_index to map each recommendation to the input index.
- Set guardrail_status for each item as PASS, WARN, or FAIL.
- Include guardrail_flags when guardrails are triggered.
- Keep agent_confidence between 0 and 1.
- Return JSON only. No markdown, no commentary.
""".strip()


class PackagingAgent:
    def __init__(self, audit_store: AuditStore | None = None) -> None:
        settings = get_settings()
        self._audit_store = audit_store or AuditStore(file_path="audit.jsonl")
        self._agent = Agent(
            name="PackagingAgent",
            model=Gemini(
                id=settings.agno_model,
                api_key=settings.google_api_key or None,
            ),
            system_message=PACKAGING_SYSTEM_PROMPT,
            output_schema=PackagingResult,
            structured_outputs=True,
            parse_response=True,
            markdown=False,
            use_json_mode=False,
        )

    async def recommend(self, request: PackagingRequest) -> PackagingResult:
        prompt = self._build_prompt(request)
        try:
            run_output = await asyncio.to_thread(
                self._agent.run,
                prompt,
                output_schema=PackagingResult,
            )
        except Exception as exc:
            raise RuntimeError(f"PackagingAgent run failed: {exc}") from exc

        if run_output.status == RunStatus.error:
            error_message = str(run_output.content or "Unknown PackagingAgent failure.")
            raise RuntimeError(f"PackagingAgent run failed: {error_message}")

        result = self._coerce_packaging_result(run_output.content)
        await self._write_audit_log(request, result)
        return result

    def _build_prompt(self, request: PackagingRequest) -> str:
        payload = request.model_dump(mode="json")
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return (
            "Generate packaging recommendations for this request payload.\n"
            "Input JSON:\n"
            f"{payload_json}\n"
            "Remember to enforce guardrails and return only valid PackagingResult JSON."
        )

    def _coerce_packaging_result(self, content: Any) -> PackagingResult:
        if isinstance(content, PackagingResult):
            return content
        if isinstance(content, dict):
            return self._validate_packaging_result(content)
        if isinstance(content, BaseModel):
            return self._validate_packaging_result(content.model_dump(mode="json"))
        if isinstance(content, str):
            try:
                return PackagingResult.model_validate_json(content)
            except ValidationError:
                parsed = self._parse_json_object(content)
                return self._validate_packaging_result(parsed)
        raise ValueError(f"Unsupported PackagingAgent response type: {type(content)!r}")

    def _validate_packaging_result(self, payload: dict[str, Any]) -> PackagingResult:
        try:
            return PackagingResult.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"PackagingAgent returned invalid PackagingResult: {exc}") from exc

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("PackagingAgent response is not valid JSON object output.")

    async def _write_audit_log(
        self,
        request: PackagingRequest,
        result: PackagingResult,
    ) -> None:
        item_statuses = [item.guardrail_status for item in result.items]
        overall_status = "PASS"
        if "FAIL" in item_statuses:
            overall_status = "FAIL"
        elif "WARN" in item_statuses:
            overall_status = "WARN"

        item_flags = sorted({flag for item in result.items for flag in item.guardrail_flags})
        compliance_flags = sorted(set(request.compliance_flags))
        all_flags = sorted(set(item_flags + compliance_flags))

        log_entry = AgentDecisionLog(
            timestamp=datetime.now(timezone.utc),
            agent_name="PackagingAgent",
            decision=f"Generated packaging recommendations for {len(result.items)} items",
            rationale=(
                f"Total packaging cost: ${result.total_cost_usd:.2f}. "
                f"Flags: {all_flags if all_flags else ['none']}."
            ),
            guardrails_checked=[
                "fragile_requires_bubble_wrap",
                "hazmat_requires_hazmat_rated_packaging",
                "cost_budget_15_percent_of_proxy_value",
            ],
            guardrail_status=overall_status,
            confidence=result.agent_confidence,
        )
        await self._audit_store.append(log_entry)


def build_packaging_agent() -> PackagingAgent:
    return PackagingAgent()

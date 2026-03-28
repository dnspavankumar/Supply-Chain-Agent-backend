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
from models.schemas import AgentDecisionLog, PackagingRequest, PackagingResult
from services.audit_store import AuditStore

try:
    from agno.run import RunStatus
except Exception:  # pragma: no cover
    RunStatus = None  # type: ignore[assignment]


PACKAGING_SYSTEM_PROMPT = """
You are PackagingAgent for a supply-chain backend.

Your job:
1) Recommend packaging materials for each goods item.
2) Estimate packaging cost per item in USD.
3) Apply guardrails BEFORE final recommendations.

Guardrails:
- Fragile guardrail:
  If is_fragile=true, bubble wrap must be included as a layer.
  Single-wall cardboard as the only cardboard layer is rejected.
- Hazmat guardrail:
  If is_hazmat=true, recommend hazmat-rated packaging materials only.
- Cost budget guardrail:
  goods_value_proxy_usd = weight_kg * 100.
  If packaging cost > 15% of goods_value_proxy_usd, add a "cost_limit" flag.

Status rules:
- PASS: all guardrails satisfied, no warnings.
- WARN: recommendations satisfy hard safety rules but include a non-blocking warning
  like "cost_limit".
- FAIL: any hard guardrail is violated.

Output requirements:
- Return JSON only.
- Return JSON that strictly matches PackagingResult schema:
  {
    "items": [
      {
        "goods_index": int,
        "recommended_materials": [str],
        "estimated_cost_usd": float,
        "guardrail_flags": [str],
        "guardrail_status": "PASS" | "WARN" | "FAIL"
      }
    ],
    "total_cost_usd": float,
    "agent_confidence": float
  }
- items length must equal number of goods.
- goods_index must map to input order (0-based index).
- agent_confidence must be between 0 and 1.
Do not include markdown, prose, or keys outside this schema.
""".strip()


class PackagingAgent:
    def __init__(self, audit_store: AuditStore | None = None) -> None:
        settings = get_settings()
        self._audit_store = audit_store or AuditStore(file_path="audit.jsonl")
        api_key = (settings.google_api_key or "").strip()
        if settings.require_google_api_key and not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required in strict mode. "
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env before starting backend."
            )

        self._schema_param_name: str | None = None
        self._run_schema_param_name: str | None = None

        model = Gemini(
            id=settings.gemini_model_id,
            api_key=api_key or None,
        )
        self._agent = self._build_agent(model)

    def _build_agent(self, model: Gemini) -> Agent:
        init_params = set(inspect.signature(Agent.__init__).parameters)

        kwargs: dict[str, Any] = {
            "name": "PackagingAgent",
            "model": model,
            "system_message": PACKAGING_SYSTEM_PROMPT,
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
            kwargs["response_model"] = PackagingResult
            self._schema_param_name = "response_model"
        elif "output_schema" in init_params:
            kwargs["output_schema"] = PackagingResult
            self._schema_param_name = "output_schema"

        run_params = set(inspect.signature(Agent.run).parameters)
        if "response_model" in run_params:
            self._run_schema_param_name = "response_model"
        elif "output_schema" in run_params:
            self._run_schema_param_name = "output_schema"

        return Agent(**kwargs)

    async def recommend(self, request: PackagingRequest) -> PackagingResult:
        prompt = self._build_prompt(request)

        run_kwargs: dict[str, Any] = {}
        if self._schema_param_name is None and self._run_schema_param_name is not None:
            run_kwargs[self._run_schema_param_name] = PackagingResult

        try:
            run_output = await asyncio.to_thread(self._agent.run, prompt, **run_kwargs)
        except TypeError:
            run_output = await asyncio.to_thread(self._agent.run, prompt)
        except Exception as exc:
            raise RuntimeError(f"PackagingAgent run failed: {exc}") from exc

        try:
            self._raise_if_run_error(run_output)
            result = self._coerce_packaging_result(self._extract_content(run_output))
        except Exception as exc:
            raise RuntimeError(f"PackagingAgent returned invalid output: {exc}") from exc
        await self._write_audit_log(request=request, result=result)
        return result

    def _raise_if_run_error(self, run_output: Any) -> None:
        if not hasattr(run_output, "status"):
            return

        status = getattr(run_output, "status")
        if RunStatus is not None and status == RunStatus.error:
            message = str(getattr(run_output, "content", "") or "Unknown PackagingAgent error.")
            raise RuntimeError(f"PackagingAgent run failed: {message}")

        status_text = str(status).upper()
        if "ERROR" in status_text:
            message = str(getattr(run_output, "content", "") or "Unknown PackagingAgent error.")
            raise RuntimeError(f"PackagingAgent run failed: {message}")

    def _extract_content(self, run_output: Any) -> Any:
        if hasattr(run_output, "content"):
            return getattr(run_output, "content")
        return run_output

    def _build_prompt(self, request: PackagingRequest) -> str:
        payload_json = json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2)
        return (
            "Generate packaging recommendations from this PackagingRequest JSON:\n"
            f"{payload_json}\n"
            "Apply guardrails and return JSON only matching PackagingResult."
        )

    def _coerce_packaging_result(self, content: Any) -> PackagingResult:
        if isinstance(content, PackagingResult):
            return content
        if isinstance(content, dict):
            return self._validate_packaging_result(content)
        if isinstance(content, BaseModel):
            return self._validate_packaging_result(content.model_dump(mode="json"))
        if isinstance(content, str):
            return self._parse_and_validate_text(content)
        raise ValueError(f"Unsupported PackagingAgent response type: {type(content)!r}")

    def _parse_and_validate_text(self, text: str) -> PackagingResult:
        stripped = text.strip()

        try:
            return PackagingResult.model_validate_json(stripped)
        except ValidationError:
            payload = self._parse_json_object(stripped)
            return self._validate_packaging_result(payload)

    def _validate_packaging_result(self, payload: dict[str, Any]) -> PackagingResult:
        try:
            return PackagingResult.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"PackagingAgent returned invalid PackagingResult: {exc}") from exc

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

        raise ValueError("PackagingAgent response is not valid JSON object output.")

    async def _write_audit_log(self, request: PackagingRequest, result: PackagingResult) -> None:
        statuses = [item.guardrail_status for item in result.items]
        overall_status = "PASS"
        if "FAIL" in statuses:
            overall_status = "FAIL"
        elif "WARN" in statuses:
            overall_status = "WARN"

        item_flags = {flag for item in result.items for flag in item.guardrail_flags}
        compliance_flags = set(request.compliance_flags)
        for check in request.compliance_results:
            compliance_flags.update(check.flags)

        log_entry = AgentDecisionLog(
            timestamp=datetime.now(timezone.utc),
            agent_name="PackagingAgent",
            decision=f"Generated packaging recommendations for {len(result.items)} items",
            rationale=(
                f"Estimated total packaging cost: ${result.total_cost_usd:.2f}. "
                f"Flags: {sorted(item_flags | compliance_flags) or ['none']}."
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

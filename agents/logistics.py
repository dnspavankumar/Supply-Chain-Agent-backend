from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timezone
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field, ValidationError

from config import get_settings
from models.schemas import (
    AgentDecisionLog,
    Item,
    LogisticsRequest,
    LogisticsResult,
    VehicleInput,
    VehiclePlan,
)
from services.audit_store import AuditStore
from services.bin_packing import best_bin_packing_result

try:
    from agno.run import RunStatus
except Exception:  # pragma: no cover
    RunStatus = None  # type: ignore[assignment]


class LogisticsNarrative(BaseModel):
    recommendation: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


LOGISTICS_SYSTEM_PROMPT = """
You are LogisticsAgent, focused on explaining deterministic 3D bin packing outcomes.

You will receive a JSON context that includes:
- number of items
- vehicle plans with space and weight utilization
- overflow items
- guardrail flags

Task:
1) Explain the plan in plain language.
2) Mention if shipment split is required.
3) Call out important guardrail risks if present.
4) Keep recommendation concise and operational.

Output contract:
- Return JSON only.
- Match this exact schema:
  {
    "recommendation": string,
    "confidence": float  // 0..1
  }
- No markdown, no additional keys.
""".strip()


class LogisticsAgent:
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
            "name": "LogisticsAgent",
            "model": model,
            "system_message": LOGISTICS_SYSTEM_PROMPT,
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
            kwargs["response_model"] = LogisticsNarrative
            self._schema_param_name = "response_model"
        elif "output_schema" in init_params:
            kwargs["output_schema"] = LogisticsNarrative
            self._schema_param_name = "output_schema"

        run_params = set(inspect.signature(Agent.run).parameters)
        if "response_model" in run_params:
            self._run_schema_param_name = "response_model"
        elif "output_schema" in run_params:
            self._run_schema_param_name = "output_schema"

        return Agent(**kwargs)

    async def optimize(self, request: LogisticsRequest) -> LogisticsResult:
        items = [goods.to_item(index) for index, goods in enumerate(request.goods)]
        initial_flags = list(request.compliance_flags)
        if request.packaging_result is not None:
            for packaged in request.packaging_result.items:
                initial_flags.extend(packaged.guardrail_flags)
                if packaged.guardrail_status == "FAIL":
                    initial_flags.append("packaging_guardrail_fail")

        plans, overflow_items, guardrail_flags = self._build_vehicle_plans(
            items=items,
            vehicles=request.vehicles,
            initial_flags=initial_flags,
        )

        context_payload = {
            "items_count": len(items),
            "vehicle_plans": [plan.model_dump(mode="json") for plan in plans],
            "total_vehicles_needed": len(plans),
            "overflow_items": overflow_items,
            "guardrail_flags": guardrail_flags,
        }

        narrative = await self._get_narrative(context_payload)
        result = LogisticsResult(
            vehicle_plans=plans,
            total_vehicles_needed=len(plans),
            overflow_items=overflow_items,
            guardrail_flags=guardrail_flags,
            recommendation=narrative.recommendation,
        )
        await self._write_audit_log(request=request, result=result, confidence=narrative.confidence)
        return result

    def _build_vehicle_plans(
        self,
        items: list[Item],
        vehicles: list[VehicleInput],
        initial_flags: list[str],
    ) -> tuple[list[VehiclePlan], list[str], list[str]]:
        remaining = list(items)
        plans: list[VehiclePlan] = []

        for vehicle in vehicles:
            if not remaining:
                break
            cargo_bin = vehicle.to_bin()
            best_result = best_bin_packing_result(remaining, cargo_bin)
            if not best_result.packed_items:
                continue

            plans.append(
                VehiclePlan(
                    vehicle_id=vehicle.vehicle_id,
                    packed_items=best_result.packed_items,
                    utilization_pct=best_result.space_utilization_pct,
                    weight_utilization_pct=best_result.weight_utilization_pct,
                    positions=best_result.positions,
                    cargo_dimensions_cm={
                        "l": cargo_bin.length,
                        "w": cargo_bin.width,
                        "h": cargo_bin.height,
                    },
                    strategy_used=best_result.strategy,
                )
            )

            packed_lookup = set(best_result.packed_items)
            remaining = [item for item in remaining if item.item_id not in packed_lookup]

        template_vehicle = max(
            vehicles,
            key=lambda vehicle: (
                vehicle.capacity_cbm,
                vehicle.max_weight_kg,
            ),
        )
        split_index = 1
        template_bin = template_vehicle.to_bin()
        while remaining:
            best_result = best_bin_packing_result(remaining, template_bin)
            if not best_result.packed_items:
                break

            plans.append(
                VehiclePlan(
                    vehicle_id=f"{template_vehicle.vehicle_id}_split_{split_index}",
                    packed_items=best_result.packed_items,
                    utilization_pct=best_result.space_utilization_pct,
                    weight_utilization_pct=best_result.weight_utilization_pct,
                    positions=best_result.positions,
                    cargo_dimensions_cm={
                        "l": template_bin.length,
                        "w": template_bin.width,
                        "h": template_bin.height,
                    },
                    strategy_used=best_result.strategy,
                )
            )

            packed_lookup = set(best_result.packed_items)
            remaining = [item for item in remaining if item.item_id not in packed_lookup]
            split_index += 1

        overflow_items = [item.item_id for item in remaining]
        guardrail_flags = set(initial_flags)

        max_vehicle_weight = max(vehicle.max_weight_kg for vehicle in vehicles)
        if any(item.weight_kg > max_vehicle_weight for item in items):
            guardrail_flags.add("weight_limit_exceeded")
        if overflow_items:
            guardrail_flags.add("capacity_exceeded")
        if len(plans) > 1:
            guardrail_flags.add("split_shipment_required")

        return plans, overflow_items, sorted(guardrail_flags)

    async def _get_narrative(self, context_payload: dict[str, Any]) -> LogisticsNarrative:
        prompt = (
            "Generate logistics recommendation from this deterministic planning context:\n"
            f"{json.dumps(context_payload, ensure_ascii=False, indent=2)}\n"
            "Return only valid LogisticsNarrative JSON."
        )

        run_kwargs: dict[str, Any] = {}
        if self._schema_param_name is None and self._run_schema_param_name is not None:
            run_kwargs[self._run_schema_param_name] = LogisticsNarrative

        try:
            run_output = await asyncio.to_thread(self._agent.run, prompt, **run_kwargs)
            self._raise_if_run_error(run_output)
            narrative = self._coerce_narrative(self._extract_content(run_output))
            if not narrative.recommendation.strip():
                raise ValueError("Empty recommendation from LogisticsAgent.")
            return narrative
        except Exception as exc:
            raise RuntimeError(f"LogisticsAgent narrative generation failed: {exc}") from exc

    def _raise_if_run_error(self, run_output: Any) -> None:
        if not hasattr(run_output, "status"):
            return
        status = getattr(run_output, "status")
        if RunStatus is not None and status == RunStatus.error:
            message = str(getattr(run_output, "content", "") or "Unknown LogisticsAgent error.")
            raise RuntimeError(f"LogisticsAgent run failed: {message}")
        if "ERROR" in str(status).upper():
            message = str(getattr(run_output, "content", "") or "Unknown LogisticsAgent error.")
            raise RuntimeError(f"LogisticsAgent run failed: {message}")

    def _extract_content(self, run_output: Any) -> Any:
        if hasattr(run_output, "content"):
            return getattr(run_output, "content")
        return run_output

    def _coerce_narrative(self, content: Any) -> LogisticsNarrative:
        if isinstance(content, LogisticsNarrative):
            return content
        if isinstance(content, BaseModel):
            return LogisticsNarrative.model_validate(content.model_dump(mode="json"))
        if isinstance(content, dict):
            return LogisticsNarrative.model_validate(content)
        if isinstance(content, str):
            return self._parse_narrative_text(content)
        raise ValueError(f"Unsupported LogisticsAgent response type: {type(content)!r}")

    def _parse_narrative_text(self, text: str) -> LogisticsNarrative:
        stripped = text.strip()
        try:
            return LogisticsNarrative.model_validate_json(stripped)
        except ValidationError:
            payload = self._parse_json_object(stripped)
            return LogisticsNarrative.model_validate(payload)

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

        raise ValueError("LogisticsAgent recommendation is not valid JSON object output.")

    async def _write_audit_log(
        self,
        request: LogisticsRequest,
        result: LogisticsResult,
        confidence: float,
    ) -> None:
        guardrail_status = "PASS"
        if result.overflow_items:
            guardrail_status = "FAIL"
        elif result.guardrail_flags:
            guardrail_status = "WARN"

        log_entry = AgentDecisionLog(
            timestamp=datetime.now(timezone.utc),
            agent_name="LogisticsAgent",
            decision=(
                f"Built logistics plan for {len(request.goods)} goods across "
                f"{result.total_vehicles_needed} vehicle(s)"
            ),
            rationale=(
                f"Overflow items: {result.overflow_items or ['none']}. "
                f"Guardrail flags: {result.guardrail_flags or ['none']}."
            ),
            guardrails_checked=[
                "vehicle_cargo_capacity",
                "vehicle_max_weight",
                "overflow_detection",
                "split_shipment_decision",
            ],
            guardrail_status=guardrail_status,
            confidence=confidence,
        )
        await self._audit_store.append(log_entry)


def build_logistics_agent() -> LogisticsAgent:
    return LogisticsAgent()

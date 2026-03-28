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
from models.schemas import AgentDecisionLog, ReroutingRequest, ReroutingResult, RouteInput, ScoredRoute
from services.audit_store import AuditStore
from services.compliance import ComplianceEngine
from services.map_routing import MapRoutingService, RouteMetrics
from services.route_scorer import compute_route_score

try:
    from agno.run import RunStatus
except Exception:  # pragma: no cover
    RunStatus = None  # type: ignore[assignment]


CURFEW_START_HOUR = 22
CURFEW_END_HOUR = 6


class ReroutingNarrative(BaseModel):
    rationale: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


REROUTING_SYSTEM_PROMPT = """
You are ReroutingAgent for a supply-chain control tower.

Given scored route options and compliance outcomes:
1) Explain why the selected route is safest and most practical.
2) If no route is selected, explain why escalation is required.
3) Mention disruption type and risk controls.

Output contract:
- Return JSON only.
- Match exactly:
  {
    "rationale": string,
    "confidence": float
  }
- confidence must be between 0 and 1.
- No markdown or extra keys.
""".strip()


class ReroutingAgent:
    def __init__(
        self,
        audit_store: AuditStore | None = None,
        compliance_engine: ComplianceEngine | None = None,
        map_routing_service: MapRoutingService | None = None,
    ) -> None:
        settings = get_settings()
        self._audit_store = audit_store or AuditStore(file_path="audit.jsonl")
        self._compliance_engine = compliance_engine or ComplianceEngine()
        self._map_routing_service = map_routing_service or MapRoutingService()

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
            "name": "ReroutingAgent",
            "model": model,
            "system_message": REROUTING_SYSTEM_PROMPT,
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
            kwargs["response_model"] = ReroutingNarrative
            self._schema_param_name = "response_model"
        elif "output_schema" in init_params:
            kwargs["output_schema"] = ReroutingNarrative
            self._schema_param_name = "output_schema"

        run_params = set(inspect.signature(Agent.run).parameters)
        if "response_model" in run_params:
            self._run_schema_param_name = "response_model"
        elif "output_schema" in run_params:
            self._run_schema_param_name = "output_schema"

        return Agent(**kwargs)

    async def reroute(self, request: ReroutingRequest) -> ReroutingResult:
        map_metrics = await self._fetch_metrics(request.available_alternatives)
        scored_routes, selected = self._score_and_filter_routes(request, map_metrics)

        status = "REROUTED" if selected is not None else "ESCALATE"
        selected_route = selected.route if selected is not None else None
        selected_score = selected.score if selected is not None else None

        narrative = await self._generate_rationale(
            request=request,
            status=status,
            selected=selected,
            scored_routes=scored_routes,
        )

        result = ReroutingResult(
            ticket_id=request.ticket_id,
            selected_route=selected_route,
            route_score=selected_score,
            status=status,  # type: ignore[arg-type]
            all_scored_routes=scored_routes,
            rationale=narrative.rationale,
        )
        await self._write_audit_log(request, result, narrative.confidence)
        return result

    async def _fetch_metrics(self, routes: list[RouteInput]) -> list[RouteMetrics | None]:
        tasks = [self._map_routing_service.get_route_metrics(route) for route in routes]
        return await asyncio.gather(*tasks)

    def _score_and_filter_routes(
        self,
        request: ReroutingRequest,
        map_metrics: list[RouteMetrics | None],
    ) -> tuple[list[ScoredRoute], ScoredRoute | None]:
        durations = [metric.duration_min for metric in map_metrics if metric is not None]
        duration_min = min(durations) if durations else None
        duration_max = max(durations) if durations else None

        scored_routes: list[ScoredRoute] = []
        compliant_candidates: list[ScoredRoute] = []

        for route, metrics in zip(request.available_alternatives, map_metrics):
            weather_adjusted, safety_adjusted = self._adjust_scores_for_map_metrics(
                request.weather_score,
                request.road_safety_score,
                metrics,
                duration_min,
                duration_max,
            )
            score = compute_route_score(weather_adjusted, safety_adjusted)

            disqualifications: list[str] = []
            restricted_check = self._compliance_engine.check_restricted_zones(route)
            if restricted_check.status == "FAIL":
                disqualifications.append("restricted_zone_overlap")

            eta = route.estimated_arrival_time or request.estimated_arrival_time
            if self._is_curfew_violation(eta):
                disqualifications.append("curfew_violation")

            candidate = ScoredRoute(
                route=route,
                score=score,
                disqualification_reason=", ".join(disqualifications) if disqualifications else None,
                map_distance_km=metrics.distance_km if metrics is not None else None,
                map_duration_min=metrics.duration_min if metrics is not None else None,
                map_source=metrics.source if metrics is not None else None,
            )
            scored_routes.append(candidate)
            if not disqualifications:
                compliant_candidates.append(candidate)

        selected = max(compliant_candidates, key=lambda route: route.score) if compliant_candidates else None
        return scored_routes, selected

    def _adjust_scores_for_map_metrics(
        self,
        base_weather: float,
        base_safety: float,
        metrics: RouteMetrics | None,
        duration_min: float | None,
        duration_max: float | None,
    ) -> tuple[float, float]:
        weather = max(0.0, min(1.0, base_weather))
        safety = max(0.0, min(1.0, base_safety))
        if metrics is None or duration_min is None or duration_max is None:
            return weather, safety

        spread = duration_max - duration_min
        if spread <= 1e-9:
            return weather, safety

        duration_factor = (metrics.duration_min - duration_min) / spread
        weather_adjusted = max(0.0, min(1.0, weather - (0.10 * duration_factor)))
        safety_adjusted = max(0.0, min(1.0, safety - (0.20 * duration_factor)))
        return weather_adjusted, safety_adjusted

    def _is_curfew_violation(self, eta: datetime | None) -> bool:
        if eta is None:
            return False

        hour = eta.hour
        return hour >= CURFEW_START_HOUR or hour < CURFEW_END_HOUR

    async def _generate_rationale(
        self,
        request: ReroutingRequest,
        status: str,
        selected: ScoredRoute | None,
        scored_routes: list[ScoredRoute],
    ) -> ReroutingNarrative:
        context_payload = {
            "ticket_id": request.ticket_id,
            "disruption_type": request.disruption_type,
            "disruption_location": request.disruption_location,
            "status": status,
            "selected_route": selected.model_dump(mode="json") if selected is not None else None,
            "all_scored_routes": [route.model_dump(mode="json") for route in scored_routes],
        }
        prompt = (
            "Generate rerouting rationale from this context:\n"
            f"{json.dumps(context_payload, ensure_ascii=False, indent=2)}\n"
            "Return only valid ReroutingNarrative JSON."
        )

        run_kwargs: dict[str, Any] = {}
        if self._schema_param_name is None and self._run_schema_param_name is not None:
            run_kwargs[self._run_schema_param_name] = ReroutingNarrative

        try:
            run_output = await asyncio.to_thread(self._agent.run, prompt, **run_kwargs)
            self._raise_if_run_error(run_output)
            narrative = self._coerce_narrative(self._extract_content(run_output))
            if not narrative.rationale.strip():
                raise ValueError("ReroutingAgent returned empty rationale.")
            return narrative
        except Exception:
            return ReroutingNarrative(
                rationale=self._fallback_rationale(request, status, selected, scored_routes),
                confidence=0.55,
            )

    def _raise_if_run_error(self, run_output: Any) -> None:
        if not hasattr(run_output, "status"):
            return
        status = getattr(run_output, "status")
        if RunStatus is not None and status == RunStatus.error:
            message = str(getattr(run_output, "content", "") or "Unknown ReroutingAgent error.")
            raise RuntimeError(f"ReroutingAgent run failed: {message}")
        if "ERROR" in str(status).upper():
            message = str(getattr(run_output, "content", "") or "Unknown ReroutingAgent error.")
            raise RuntimeError(f"ReroutingAgent run failed: {message}")

    def _extract_content(self, run_output: Any) -> Any:
        if hasattr(run_output, "content"):
            return getattr(run_output, "content")
        return run_output

    def _coerce_narrative(self, content: Any) -> ReroutingNarrative:
        if isinstance(content, ReroutingNarrative):
            return content
        if isinstance(content, BaseModel):
            return ReroutingNarrative.model_validate(content.model_dump(mode="json"))
        if isinstance(content, dict):
            return ReroutingNarrative.model_validate(content)
        if isinstance(content, str):
            return self._parse_narrative_text(content)
        raise ValueError(f"Unsupported ReroutingAgent response type: {type(content)!r}")

    def _parse_narrative_text(self, text: str) -> ReroutingNarrative:
        stripped = text.strip()
        try:
            return ReroutingNarrative.model_validate_json(stripped)
        except ValidationError:
            payload = self._parse_json_object(stripped)
            return ReroutingNarrative.model_validate(payload)

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

        raise ValueError("ReroutingAgent rationale is not valid JSON object output.")

    def _fallback_rationale(
        self,
        request: ReroutingRequest,
        status: str,
        selected: ScoredRoute | None,
        scored_routes: list[ScoredRoute],
    ) -> str:
        if status == "REROUTED" and selected is not None:
            return (
                f"Ticket {request.ticket_id}: selected the highest-scoring compliant route "
                f"(score={selected.score:.3f}) after filtering restricted-zone and curfew violations."
            )
        disqualified = [route for route in scored_routes if route.disqualification_reason]
        return (
            f"Ticket {request.ticket_id}: no compliant alternative route remained after guardrail checks. "
            f"Escalating for human review; {len(disqualified)} route(s) were disqualified."
        )

    async def _write_audit_log(
        self,
        request: ReroutingRequest,
        result: ReroutingResult,
        confidence: float,
    ) -> None:
        guardrail_status = "PASS" if result.status == "REROUTED" else "FAIL"
        if result.status == "REROUTED" and any(
            route.disqualification_reason for route in result.all_scored_routes
        ):
            guardrail_status = "WARN"

        log_entry = AgentDecisionLog(
            timestamp=datetime.now(timezone.utc),
            agent_name="ReroutingAgent",
            decision=(
                f"Ticket {request.ticket_id} {result.status.lower()} for disruption "
                f"{request.disruption_type}"
            ),
            rationale=result.rationale,
            guardrails_checked=[
                "restricted_zone_filter",
                "curfew_filter_22_to_06",
                "weighted_route_score_alpha_beta",
            ],
            guardrail_status=guardrail_status,  # type: ignore[arg-type]
            confidence=confidence,
        )
        await self._audit_store.append(log_entry)


def build_rerouting_agent() -> ReroutingAgent:
    return ReroutingAgent()

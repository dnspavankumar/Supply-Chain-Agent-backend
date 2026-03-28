from __future__ import annotations

import asyncio
import math
import inspect
import json
from datetime import datetime, timedelta, timezone
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
MAX_ROUTE_OPTIONS = 4


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
        candidate_routes, map_metrics = await self._resolve_route_options(request)
        scored_routes, selected = self._score_and_filter_routes(
            request=request,
            candidate_routes=candidate_routes,
            map_metrics=map_metrics,
        )

        status = "REROUTED" if selected is not None else "ESCALATE"
        try:
            narrative = await self._generate_rationale(
                request=request,
                status=status,
                selected=selected,
                scored_routes=scored_routes,
            )
        except Exception as exc:
            selected = self._fallback_to_next_shortest_route(scored_routes, selected)
            status = "REROUTED" if selected is not None else "ESCALATE"
            narrative = ReroutingNarrative(
                rationale=self._build_failover_rationale(
                    request=request,
                    selected=selected,
                    scored_routes=scored_routes,
                    llm_error=str(exc),
                ),
                confidence=0.35,
            )

        selected_route = selected.route if selected is not None else None
        selected_score = selected.score if selected is not None else None

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

    async def _resolve_route_options(
        self,
        request: ReroutingRequest,
    ) -> tuple[list[RouteInput], list[RouteMetrics | None]]:
        explicit_routes = list(request.available_alternatives)[:MAX_ROUTE_OPTIONS]
        if explicit_routes:
            return explicit_routes, await self._fetch_metrics(explicit_routes)

        generated_metrics = await self._map_routing_service.get_route_alternatives(
            request.original_route,
            max_alternatives=MAX_ROUTE_OPTIONS,
        )
        if generated_metrics:
            generated_routes = self._build_generated_routes(
                request=request,
                route_count=len(generated_metrics),
            )
            return generated_routes, list(generated_metrics)

        fallback_route = RouteInput(
            origin=request.original_route.origin,
            destination=request.original_route.destination,
            waypoints=list(request.original_route.waypoints),
            estimated_arrival_time=(
                request.estimated_arrival_time or request.original_route.estimated_arrival_time
            ),
        )
        fallback_metrics = await self._fetch_metrics([fallback_route])
        return [fallback_route], fallback_metrics

    def _build_generated_routes(
        self,
        request: ReroutingRequest,
        route_count: int,
    ) -> list[RouteInput]:
        base_eta = request.estimated_arrival_time or request.original_route.estimated_arrival_time
        generated: list[RouteInput] = []

        for idx in range(route_count):
            eta = base_eta + timedelta(minutes=idx * 10) if base_eta is not None else None
            generated.append(
                RouteInput(
                    origin=request.original_route.origin,
                    destination=request.original_route.destination,
                    waypoints=list(request.original_route.waypoints),
                    estimated_arrival_time=eta,
                )
            )
        return generated

    async def _fetch_metrics(self, routes: list[RouteInput]) -> list[RouteMetrics | None]:
        tasks = [self._map_routing_service.get_route_metrics(route) for route in routes]
        return await asyncio.gather(*tasks)

    def _score_and_filter_routes(
        self,
        request: ReroutingRequest,
        candidate_routes: list[RouteInput],
        map_metrics: list[RouteMetrics | None],
    ) -> tuple[list[ScoredRoute], ScoredRoute | None]:
        durations = [metric.duration_min for metric in map_metrics if metric is not None]
        duration_min = min(durations) if durations else None
        duration_max = max(durations) if durations else None
        eta_values = [
            (route.estimated_arrival_time or request.estimated_arrival_time).timestamp()
            for route in candidate_routes
            if (route.estimated_arrival_time or request.estimated_arrival_time) is not None
        ]
        eta_min = min(eta_values) if eta_values else None
        eta_max = max(eta_values) if eta_values else None

        scored_routes: list[ScoredRoute] = []
        compliant_candidates: list[ScoredRoute] = []

        for route, metrics in zip(candidate_routes, map_metrics):
            weather_adjusted, safety_adjusted = self._adjust_scores_for_map_metrics(
                request.weather_score,
                request.road_safety_score,
                metrics,
                duration_min,
                duration_max,
            )
            score = compute_route_score(weather_adjusted, safety_adjusted)
            score = self._apply_fallback_eta_adjustment(
                score=score,
                route_eta=(route.estimated_arrival_time or request.estimated_arrival_time),
                eta_min=eta_min,
                eta_max=eta_max,
                metrics_present=metrics is not None,
            )
            score = self._apply_waypoint_penalty(score, route.waypoints)

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
                map_geometry=metrics.geometry_coords if metrics is not None else [],
            )
            scored_routes.append(candidate)
            if not disqualifications:
                compliant_candidates.append(candidate)

        selected = max(compliant_candidates, key=lambda route: route.score) if compliant_candidates else None
        return scored_routes, selected

    def _rank_compliant_by_shortest(self, scored_routes: list[ScoredRoute]) -> list[ScoredRoute]:
        compliant = [route for route in scored_routes if not route.disqualification_reason]
        return sorted(
            compliant,
            key=lambda route: (
                route.map_duration_min if route.map_duration_min is not None else math.inf,
                route.map_distance_km if route.map_distance_km is not None else math.inf,
                -route.score,
            ),
        )

    def _fallback_to_next_shortest_route(
        self,
        scored_routes: list[ScoredRoute],
        current_selected: ScoredRoute | None,
    ) -> ScoredRoute | None:
        ranked = self._rank_compliant_by_shortest(scored_routes)
        if not ranked:
            return None
        if current_selected is None:
            return ranked[0]

        current_signature = self._route_signature(current_selected.route)
        for candidate in ranked:
            if self._route_signature(candidate.route) != current_signature:
                return candidate
        return ranked[0]

    def _route_signature(self, route: RouteInput) -> str:
        waypoints = ",".join(route.waypoints or [])
        return f"{route.origin}|{waypoints}|{route.destination}"

    def _build_failover_rationale(
        self,
        request: ReroutingRequest,
        selected: ScoredRoute | None,
        scored_routes: list[ScoredRoute],
        llm_error: str,
    ) -> str:
        compact_error = " ".join(llm_error.split())
        if len(compact_error) > 180:
            compact_error = f"{compact_error[:177]}..."

        ranked = self._rank_compliant_by_shortest(scored_routes)
        if selected is not None:
            rank = 1
            selected_signature = self._route_signature(selected.route)
            for index, candidate in enumerate(ranked, start=1):
                if self._route_signature(candidate.route) == selected_signature:
                    rank = index
                    break
            return (
                f"LLM rationale unavailable ({compact_error}). "
                f"Applied deterministic failover and selected shortest compliant candidate rank {rank} "
                f"with score {selected.score:.3f}."
            )

        return (
            f"LLM rationale unavailable ({compact_error}). "
            f"No compliant alternatives remained after guardrail checks; escalation required."
        )

    def _apply_fallback_eta_adjustment(
        self,
        score: float,
        route_eta: datetime | None,
        eta_min: float | None,
        eta_max: float | None,
        metrics_present: bool,
    ) -> float:
        if metrics_present:
            return score
        if route_eta is None or eta_min is None or eta_max is None:
            return score

        spread = eta_max - eta_min
        if spread <= 1e-9:
            return score

        eta_factor = (route_eta.timestamp() - eta_min) / spread
        adjusted = score - (0.15 * eta_factor)
        return max(0.0, min(1.0, adjusted))

    def _apply_waypoint_penalty(self, score: float, waypoints: list[str]) -> float:
        penalty = min(len(waypoints), 4) * 0.02
        adjusted = score - penalty
        return max(0.0, min(1.0, adjusted))

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
        except Exception as exc:
            raise RuntimeError(f"ReroutingAgent rationale generation failed: {exc}") from exc

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

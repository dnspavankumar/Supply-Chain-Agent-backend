from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response

from agents.logistics import LogisticsAgent
from agents.packaging import PackagingAgent
from models.schemas import (
    AgentDecisionLog,
    LogisticsRequest,
    PackagingRequest,
    PipelineResult,
    ShipmentRequest,
)
from services.audit_store import AuditStore
from services.compliance import ComplianceEngine
from services.errors import ComplianceViolationError
from services.route_scorer import compute_route_score

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

audit_store = AuditStore(file_path="audit.jsonl")
compliance_engine = ComplianceEngine()
packaging_agent = PackagingAgent(audit_store=audit_store)
logistics_agent = LogisticsAgent(audit_store=audit_store)

pipeline_cache: dict[str, PipelineResult] = {}
pipeline_cache_lock = asyncio.Lock()


def _collect_compliance_flags(compliance_results: list) -> list[str]:
    flags: set[str] = set()
    for result in compliance_results:
        if result.status != "PASS":
            flags.add(result.check_name)
        for flag in result.flags:
            flags.add(flag)
    return sorted(flags)


async def _collect_new_audit_entries(baseline_count: int) -> list[AgentDecisionLog]:
    all_entries = await audit_store.read_all()
    if baseline_count < 0 or baseline_count > len(all_entries):
        return all_entries
    return all_entries[baseline_count:]


async def _cache_pipeline_result(run_id: str, result: PipelineResult) -> None:
    async with pipeline_cache_lock:
        pipeline_cache[run_id] = result


@router.post(
    "/run",
    response_model=PipelineResult,
    summary="Run End-to-End Shipment Pipeline",
    description=(
        "Runs compliance, packaging, logistics, and initial route scoring in sequence. "
        "Returns a combined pipeline result and stores it in in-memory run cache."
    ),
    responses={
        422: {"description": "Compliance or packaging guardrail violation."},
    },
)
async def run_pipeline(request: ShipmentRequest, response: Response) -> PipelineResult:
    """Execute the full shipment pipeline and return unified results."""
    run_id = str(uuid4())
    response.headers["X-Run-Id"] = run_id
    baseline_audit_count = len(await audit_store.read_all())
    warnings: list[str] = []

    compliance_results = compliance_engine.run_all(request)
    if any(result.status == "FAIL" for result in compliance_results):
        failed_result = PipelineResult(
            compliance_results=compliance_results,
            packaging_result=None,
            logistics_result=None,
            initial_route_score=None,
            pipeline_status="FAILED",
            warnings=["Compliance checks failed. Pipeline execution blocked."],
            audit_entries=await _collect_new_audit_entries(baseline_audit_count),
        )
        await _cache_pipeline_result(run_id, failed_result)
        raise ComplianceViolationError(
            "Compliance checks failed",
            payload={
                "run_id": run_id,
                "compliance_results": [item.model_dump(mode="json") for item in compliance_results],
                "pipeline_result": failed_result.model_dump(mode="json"),
            },
        )

    compliance_flags = _collect_compliance_flags(compliance_results)
    packaging_result = await packaging_agent.recommend(
        PackagingRequest(
            goods=request.goods,
            compliance_results=compliance_results,
            compliance_flags=compliance_flags,
        )
    )
    if any(item.guardrail_status == "FAIL" for item in packaging_result.items):
        failed_result = PipelineResult(
            compliance_results=compliance_results,
            packaging_result=packaging_result,
            logistics_result=None,
            initial_route_score=None,
            pipeline_status="FAILED",
            warnings=["Packaging guardrail failure detected. Pipeline execution blocked."],
            audit_entries=await _collect_new_audit_entries(baseline_audit_count),
        )
        await _cache_pipeline_result(run_id, failed_result)
        raise ComplianceViolationError(
            "Packaging guardrail failure",
            payload={
                "run_id": run_id,
                "packaging_result": packaging_result.model_dump(mode="json"),
                "pipeline_result": failed_result.model_dump(mode="json"),
            },
        )

    logistics_result = await logistics_agent.optimize(
        LogisticsRequest(
            goods=request.goods,
            vehicles=request.vehicles,
            compliance_flags=compliance_flags,
            packaging_result=packaging_result,
        )
    )

    pipeline_status = "SUCCESS"
    if logistics_result.overflow_items:
        warnings.append("Some items could not be packed. Consider splitting shipment.")
        pipeline_status = "PARTIAL"

    weather_score = request.weather_score
    road_safety_score = request.road_safety_score
    if weather_score is None or road_safety_score is None:
        weather_score = 0.5 if weather_score is None else weather_score
        road_safety_score = 0.5 if road_safety_score is None else road_safety_score
        warnings.append("External API unavailable. Using conservative default scores.")

    initial_route_score = compute_route_score(
        weather_score=weather_score,
        road_safety_score=road_safety_score,
    )

    result = PipelineResult(
        compliance_results=compliance_results,
        packaging_result=packaging_result,
        logistics_result=logistics_result,
        initial_route_score=initial_route_score,
        pipeline_status=pipeline_status,  # type: ignore[arg-type]
        warnings=warnings,
        audit_entries=await _collect_new_audit_entries(baseline_audit_count),
    )
    await _cache_pipeline_result(run_id, result)
    return result


@router.get(
    "/status/{run_id}",
    response_model=PipelineResult,
    summary="Get Cached Pipeline Result",
    description="Retrieves a previously executed pipeline run result from in-memory cache.",
)
async def get_pipeline_status(run_id: str) -> PipelineResult:
    """Return pipeline result by run id if available."""
    async with pipeline_cache_lock:
        result = pipeline_cache.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Pipeline run_id not found")
    return result


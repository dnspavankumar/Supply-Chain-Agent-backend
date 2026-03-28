from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agents.logistics import LogisticsAgent
from agents.packaging import PackagingAgent
from agents.rerouting import ReroutingAgent
from models.schemas import (
    LogisticsRequest,
    LogisticsResult,
    PackagingRequest,
    PackagingResult,
    ReroutingRequest,
    ReroutingResult,
)

router = APIRouter(prefix="/agents", tags=["agents"])
packaging_agent = PackagingAgent()
logistics_agent = LogisticsAgent()
rerouting_agent = ReroutingAgent()


@router.post(
    "/packaging",
    response_model=PackagingResult,
    summary="Run Packaging Agent",
    description="Generates packaging recommendations and guardrail flags for shipment goods.",
)
async def run_packaging_agent(payload: PackagingRequest) -> PackagingResult:
    """Run the packaging recommendation agent."""
    try:
        return await packaging_agent.recommend(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/logistics",
    response_model=LogisticsResult,
    summary="Run Logistics Agent",
    description="Builds deterministic load plans and operational recommendation for available vehicles.",
)
async def run_logistics_agent(payload: LogisticsRequest) -> LogisticsResult:
    """Run logistics optimization and return vehicle plans."""
    try:
        return await logistics_agent.optimize(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/reroute",
    response_model=ReroutingResult,
    summary="Run Rerouting Agent",
    description="Evaluates alternative routes under disruption and selects the best compliant path.",
)
async def run_rerouting_agent(payload: ReroutingRequest) -> ReroutingResult:
    """Run rerouting decision logic for disruption tickets."""
    try:
        return await rerouting_agent.reroute(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agents.logistics import LogisticsAgent
from agents.packaging import PackagingAgent
from agents.rerouting import ReroutingAgent
from models.schemas import (
    LogisticsPreviewResult,
    LogisticsRequest,
    LogisticsResult,
    PackingAlgorithmResult,
    PackagingRequest,
    PackagingResult,
    ReroutingRequest,
    ReroutingResult,
    VehiclePackingPreview,
)
from services.bin_packing import deepest_bottom_left, extreme_point_rule, guillotine_heuristic

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
    "/logistics/preview",
    response_model=LogisticsPreviewResult,
    summary="Preview 3D Packing Layouts",
    description=(
        "Runs Guillotine, Extreme Point, and Deepest Bottom-Left packing algorithms "
        "for each input vehicle and returns placements plus the best strategy."
    ),
)
async def preview_logistics_layout(payload: LogisticsRequest) -> LogisticsPreviewResult:
    """Return algorithm-wise box placements for frontend 3D rendering."""
    try:
        items = [goods.to_item(index) for index, goods in enumerate(payload.goods)]
        previews: list[VehiclePackingPreview] = []

        for vehicle in payload.vehicles:
            cargo_bin = vehicle.to_bin()
            results = [
                guillotine_heuristic(items, cargo_bin),
                extreme_point_rule(items, cargo_bin),
                deepest_bottom_left(items, cargo_bin),
            ]
            best_result = max(
                results,
                key=lambda result: (
                    result.space_utilization_pct,
                    result.weight_utilization_pct,
                    len(result.packed_items),
                ),
            )
            previews.append(
                VehiclePackingPreview(
                    vehicle_id=vehicle.vehicle_id,
                    cargo_dimensions_cm={
                        "l": cargo_bin.length,
                        "w": cargo_bin.width,
                        "h": cargo_bin.height,
                    },
                    recommended_strategy=best_result.strategy,  # type: ignore[arg-type]
                    recommended_space_utilization_pct=best_result.space_utilization_pct,
                    recommended_weight_utilization_pct=best_result.weight_utilization_pct,
                    algorithm_results=[
                        PackingAlgorithmResult(
                            strategy=result.strategy,  # type: ignore[arg-type]
                            packed_items=result.packed_items,
                            unpacked_items=result.unpacked_items,
                            space_utilization_pct=result.space_utilization_pct,
                            weight_utilization_pct=result.weight_utilization_pct,
                            positions=result.positions,
                        )
                        for result in results
                    ],
                )
            )

        return LogisticsPreviewResult(vehicles=previews)
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

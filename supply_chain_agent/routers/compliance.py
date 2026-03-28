from __future__ import annotations

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from supply_chain_agent.models.schemas import ComplianceResult, ShipmentRequest
from supply_chain_agent.services.compliance import ComplianceEngine

router = APIRouter(prefix="/compliance", tags=["compliance"])
engine = ComplianceEngine()


@router.post(
    "/check",
    response_model=list[ComplianceResult],
    responses={422: {"model": list[ComplianceResult]}},
)
async def compliance_check(payload: ShipmentRequest):
    results = engine.run_all(payload)
    if any(result.status == "FAIL" for result in results):
        return JSONResponse(status_code=422, content=jsonable_encoder(results))
    return results

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from supply_chain_agent.agents.packaging import PackagingAgent
from supply_chain_agent.models.schemas import PackagingRequest, PackagingResult

router = APIRouter(prefix="/agents", tags=["agents"])
packaging_agent = PackagingAgent()


@router.post("/packaging", response_model=PackagingResult)
async def run_packaging_agent(payload: PackagingRequest) -> PackagingResult:
    try:
        return await packaging_agent.recommend(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

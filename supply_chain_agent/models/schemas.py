from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GoodsInput(BaseModel):
    goods_type: str = Field(..., min_length=1)
    dimensions: dict[str, float] = Field(..., description="Dimensions in cm: l, w, h")
    weight_kg: float = Field(..., gt=0)
    is_fragile: bool
    is_hazmat: bool

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: dict[str, float]) -> dict[str, float]:
        required_keys = {"l", "w", "h"}
        provided_keys = set(value.keys())
        if provided_keys != required_keys:
            raise ValueError("dimensions must contain exactly: l, w, h")

        for key in required_keys:
            if float(value[key]) <= 0:
                raise ValueError(f"dimensions.{key} must be greater than 0")
            value[key] = float(value[key])

        return value


class VehicleInput(BaseModel):
    vehicle_id: str = Field(..., min_length=1)
    max_weight_kg: float = Field(..., gt=0)
    capacity_cbm: float = Field(..., gt=0)
    vehicle_type: str = Field(..., min_length=1)


class RouteInput(BaseModel):
    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    waypoints: list[str] = Field(default_factory=list)


class ShipmentRequest(BaseModel):
    goods: list[GoodsInput] = Field(..., min_length=1)
    vehicles: list[VehicleInput] = Field(..., min_length=1)
    route: RouteInput


class AgentDecisionLog(BaseModel):
    timestamp: datetime
    agent_name: str = Field(..., min_length=1)
    decision: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    guardrails_checked: list[str] = Field(default_factory=list)
    guardrail_status: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ComplianceResult(BaseModel):
    check_name: str = Field(..., min_length=1)
    status: Literal["PASS", "WARN", "FAIL"]
    reason: str = Field(..., min_length=1)
    affected_items: list[str] = Field(default_factory=list)


class PackagingRequest(BaseModel):
    goods: list[GoodsInput] = Field(..., min_length=1)
    compliance_results: list[ComplianceResult] = Field(default_factory=list)
    compliance_flags: list[str] = Field(default_factory=list)


class PackagedItem(BaseModel):
    goods_index: int = Field(..., ge=0)
    recommended_materials: list[str] = Field(..., min_length=1)
    estimated_cost_usd: float = Field(..., ge=0)
    guardrail_flags: list[str] = Field(default_factory=list)
    guardrail_status: Literal["PASS", "WARN", "FAIL"]


class PackagingResult(BaseModel):
    items: list[PackagedItem] = Field(..., min_length=1)
    total_cost_usd: float = Field(..., ge=0)
    agent_confidence: float = Field(..., ge=0.0, le=1.0)

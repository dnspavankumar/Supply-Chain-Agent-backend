from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _validate_lwh_map(value: dict[str, float], field_name: str) -> dict[str, float]:
    required_keys = {"l", "w", "h"}
    provided_keys = set(value.keys())
    if provided_keys != required_keys:
        raise ValueError(f"{field_name} must contain exactly: l, w, h")

    validated: dict[str, float] = {}
    for key in required_keys:
        numeric = float(value[key])
        if numeric <= 0:
            raise ValueError(f"{field_name}.{key} must be greater than 0")
        validated[key] = numeric
    return validated


class Item(BaseModel):
    item_id: str = Field(..., min_length=1)
    length: float = Field(..., gt=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)


class Bin(BaseModel):
    length: float = Field(..., gt=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    max_weight_kg: float = Field(..., gt=0)


class PositionedItem(BaseModel):
    item_id: str = Field(..., min_length=1)
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    z: float = Field(..., ge=0)
    length: float = Field(..., gt=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class BinPackingResult(BaseModel):
    packed_items: list[str] = Field(default_factory=list)
    unpacked_items: list[str] = Field(default_factory=list)
    space_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    weight_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    positions: list[dict[str, float | str]] = Field(default_factory=list)
    strategy: Literal[
        "guillotine_heuristic",
        "extreme_point_rule",
        "deepest_bottom_left",
        "none",
    ] = "none"


class GoodsInput(BaseModel):
    item_id: str | None = Field(default=None, min_length=1)
    goods_type: str = Field(..., min_length=1)
    dimensions: dict[str, float] = Field(..., description="Dimensions in cm: l, w, h")
    weight_kg: float = Field(..., gt=0)
    is_fragile: bool
    is_hazmat: bool

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_lwh_map(value, "dimensions")

    def to_item(self, index: int) -> Item:
        dims = self.dimensions
        item_id = self.item_id or f"item_{index}"
        return Item(
            item_id=item_id,
            length=dims["l"],
            width=dims["w"],
            height=dims["h"],
            weight_kg=self.weight_kg,
        )


class VehicleInput(BaseModel):
    vehicle_id: str = Field(..., min_length=1)
    max_weight_kg: float = Field(..., gt=0)
    capacity_cbm: float = Field(..., gt=0)
    vehicle_type: str = Field(..., min_length=1)
    cargo_dimensions_cm: dict[str, float] | None = Field(
        default=None,
        description="Optional explicit cargo dimensions in cm: l, w, h",
    )

    @field_validator("cargo_dimensions_cm")
    @classmethod
    def validate_cargo_dimensions(
        cls,
        value: dict[str, float] | None,
    ) -> dict[str, float] | None:
        if value is None:
            return None
        return _validate_lwh_map(value, "cargo_dimensions_cm")

    def to_bin(self) -> Bin:
        if self.cargo_dimensions_cm is None:
            volume_cm3 = self.capacity_cbm * 1_000_000.0
            side = volume_cm3 ** (1.0 / 3.0)
            dims = {"l": side, "w": side, "h": side}
        else:
            dims = self.cargo_dimensions_cm

        return Bin(
            length=dims["l"],
            width=dims["w"],
            height=dims["h"],
            max_weight_kg=self.max_weight_kg,
        )


class RouteInput(BaseModel):
    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    waypoints: list[str] = Field(default_factory=list)
    estimated_arrival_time: datetime | None = None


class ShipmentRequest(BaseModel):
    goods: list[GoodsInput] = Field(..., min_length=1)
    vehicles: list[VehicleInput] = Field(..., min_length=1)
    route: RouteInput
    weather_score: float | None = Field(default=None, ge=0.0, le=1.0)
    road_safety_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ComplianceResult(BaseModel):
    check_name: str = Field(..., min_length=1)
    status: Literal["PASS", "WARN", "FAIL"]
    reason: str = Field(..., min_length=1)
    affected_items: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class PackagingRequest(BaseModel):
    goods: list[GoodsInput] = Field(..., min_length=1)
    compliance_results: list[ComplianceResult] = Field(default_factory=list)
    compliance_flags: list[str] = Field(default_factory=list)


class PackagedItem(BaseModel):
    goods_index: int = Field(..., ge=0)
    recommended_materials: list[str] = Field(..., min_length=1)
    estimated_cost_usd: float = Field(..., ge=0.0)
    guardrail_flags: list[str] = Field(default_factory=list)
    guardrail_status: Literal["PASS", "WARN", "FAIL"]


class PackagingResult(BaseModel):
    items: list[PackagedItem] = Field(..., min_length=1)
    total_cost_usd: float = Field(..., ge=0.0)
    agent_confidence: float = Field(..., ge=0.0, le=1.0)


class LogisticsRequest(BaseModel):
    goods: list[GoodsInput] = Field(..., min_length=1)
    vehicles: list[VehicleInput] = Field(..., min_length=1)
    compliance_flags: list[str] = Field(default_factory=list)
    packaging_result: PackagingResult | None = None


class PackingAlgorithmResult(BaseModel):
    strategy: Literal[
        "guillotine_heuristic",
        "extreme_point_rule",
        "deepest_bottom_left",
    ]
    packed_items: list[str] = Field(default_factory=list)
    unpacked_items: list[str] = Field(default_factory=list)
    space_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    weight_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    positions: list[dict[str, float | str]] = Field(default_factory=list)


class VehiclePackingPreview(BaseModel):
    vehicle_id: str = Field(..., min_length=1)
    cargo_dimensions_cm: dict[str, float] = Field(..., description="l, w, h")
    recommended_strategy: Literal[
        "guillotine_heuristic",
        "extreme_point_rule",
        "deepest_bottom_left",
    ]
    recommended_space_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    recommended_weight_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    algorithm_results: list[PackingAlgorithmResult] = Field(default_factory=list)

    @field_validator("cargo_dimensions_cm")
    @classmethod
    def validate_preview_dimensions(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_lwh_map(value, "cargo_dimensions_cm")


class LogisticsPreviewResult(BaseModel):
    vehicles: list[VehiclePackingPreview] = Field(default_factory=list)


class VehiclePlan(BaseModel):
    vehicle_id: str = Field(..., min_length=1)
    packed_items: list[str] = Field(default_factory=list)
    utilization_pct: float = Field(..., ge=0.0, le=100.0)
    weight_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    positions: list[dict[str, float | str]] = Field(default_factory=list)
    cargo_dimensions_cm: dict[str, float] = Field(..., description="l, w, h")
    strategy_used: Literal[
        "guillotine_heuristic",
        "extreme_point_rule",
        "deepest_bottom_left",
        "none",
    ] = "none"

    @field_validator("cargo_dimensions_cm")
    @classmethod
    def validate_vehicle_plan_dimensions(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_lwh_map(value, "cargo_dimensions_cm")


class LogisticsResult(BaseModel):
    vehicle_plans: list[VehiclePlan] = Field(default_factory=list)
    total_vehicles_needed: int = Field(..., ge=0)
    overflow_items: list[str] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    recommendation: str = Field(..., min_length=1)


class ReroutingRequest(BaseModel):
    ticket_id: str = Field(..., min_length=1)
    original_route: RouteInput
    disruption_type: Literal[
        "weather",
        "road_closure",
        "accident",
        "curfew",
        "restricted_zone",
    ]
    disruption_location: str = Field(..., min_length=1)
    available_alternatives: list[RouteInput] = Field(default_factory=list)
    weather_score: float = Field(..., ge=0.0, le=1.0)
    road_safety_score: float = Field(..., ge=0.0, le=1.0)
    estimated_arrival_time: datetime | None = None


class ScoredRoute(BaseModel):
    route: RouteInput
    score: float
    disqualification_reason: str | None = None
    map_distance_km: float | None = None
    map_duration_min: float | None = None
    map_source: str | None = None
    map_geometry: list[tuple[float, float]] = Field(default_factory=list)


class ReroutingResult(BaseModel):
    ticket_id: str = Field(..., min_length=1)
    selected_route: RouteInput | None = None
    route_score: float | None = None
    status: Literal["REROUTED", "ESCALATE"]
    all_scored_routes: list[ScoredRoute] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)


class AuditQueryRequest(BaseModel):
    natural_language_query: str = Field(..., min_length=1)
    agent_name: str | None = Field(default=None, min_length=1)
    date_from: datetime | None = None
    date_to: datetime | None = None


class AuditReport(BaseModel):
    summary: str = Field(..., min_length=1)
    decisions_reviewed: int = Field(..., ge=0)
    compliance_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    compliance_results: list[ComplianceResult] = Field(default_factory=list)
    packaging_result: PackagingResult | None = None
    logistics_result: LogisticsResult | None = None
    initial_route_score: float | None = None
    pipeline_status: Literal["SUCCESS", "PARTIAL", "FAILED"]
    warnings: list[str] = Field(default_factory=list)
    audit_entries: list[AgentDecisionLog] = Field(default_factory=list)


class AgentDecisionLog(BaseModel):
    timestamp: datetime
    agent_name: str = Field(..., min_length=1)
    decision: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    guardrails_checked: list[str] = Field(default_factory=list)
    guardrail_status: Literal["PASS", "WARN", "FAIL"]
    confidence: float = Field(..., ge=0.0, le=1.0)


PipelineResult.model_rebuild()

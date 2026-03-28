from __future__ import annotations

from models.schemas import ComplianceResult, GoodsInput, RouteInput, ShipmentRequest, VehicleInput


class ComplianceEngine:
    DEFAULT_RESTRICTED_ZONES: list[str] = [
        "Military Base Alpha",
        "Customs Blacksite 9",
        "No-Transit Chemical Corridor",
    ]

    def check_restricted_zones(
        self,
        route: RouteInput,
        restricted_zones: list[str] | None = None,
    ) -> ComplianceResult:
        zones = restricted_zones or self.DEFAULT_RESTRICTED_ZONES
        route_points = [route.origin, *route.waypoints, route.destination]
        route_points_lower = {point.strip().lower() for point in route_points}

        matches = [zone for zone in zones if zone.strip().lower() in route_points_lower]
        if matches:
            return ComplianceResult(
                check_name="restricted_zones",
                status="FAIL",
                reason="Route intersects one or more restricted zones.",
                affected_items=matches,
                flags=["restricted_zone_overlap"],
            )

        return ComplianceResult(
            check_name="restricted_zones",
            status="PASS",
            reason="Route does not intersect restricted zones.",
            affected_items=[],
            flags=[],
        )

    def check_weight_limits(
        self,
        goods: list[GoodsInput],
        vehicles: list[VehicleInput],
    ) -> ComplianceResult:
        total_goods_weight = sum(item.weight_kg for item in goods)
        total_vehicle_capacity = sum(vehicle.max_weight_kg for vehicle in vehicles)

        if total_goods_weight > total_vehicle_capacity:
            return ComplianceResult(
                check_name="weight_limits",
                status="FAIL",
                reason=(
                    "Total goods weight "
                    f"({total_goods_weight:.2f} kg) exceeds fleet max capacity "
                    f"({total_vehicle_capacity:.2f} kg)."
                ),
                affected_items=["total_goods_weight", "total_vehicle_capacity"],
                flags=["weight_limit_exceeded"],
            )

        return ComplianceResult(
            check_name="weight_limits",
            status="PASS",
            reason=(
                "Total goods weight "
                f"({total_goods_weight:.2f} kg) is within fleet max capacity "
                f"({total_vehicle_capacity:.2f} kg)."
            ),
            affected_items=[],
            flags=[],
        )

    def check_hazmat(self, goods: list[GoodsInput]) -> ComplianceResult:
        hazmat_items = [
            f"{idx}:{item.item_id or item.goods_type}"
            for idx, item in enumerate(goods)
            if item.is_hazmat
        ]
        if hazmat_items:
            return ComplianceResult(
                check_name="hazmat",
                status="WARN",
                reason="Hazmat goods detected. Downstream handling guardrails are required.",
                affected_items=hazmat_items,
                flags=["hazmat_present"],
            )

        return ComplianceResult(
            check_name="hazmat",
            status="PASS",
            reason="No hazmat goods detected.",
            affected_items=[],
            flags=[],
        )

    def run_all(self, request: ShipmentRequest) -> list[ComplianceResult]:
        return [
            self.check_weight_limits(request.goods, request.vehicles),
            self.check_hazmat(request.goods),
            self.check_restricted_zones(request.route),
        ]

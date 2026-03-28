from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from config import get_settings
from models.schemas import RouteInput


@dataclass
class RouteMetrics:
    distance_km: float
    duration_min: float
    source: str


class MapRoutingService:
    """Mapbox-backed route enrichment service.

    Uses Geocoding API to resolve route points and Directions API
    (driving-traffic profile) to obtain route distance and duration.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._token = settings.mapbox_access_token.strip()
        self._geocode_url = settings.mapbox_geocoding_base_url.rstrip("/")
        self._directions_base = settings.mapbox_directions_base_url.rstrip("/")
        self._timeout = settings.map_api_timeout_seconds
        self._coord_cache: dict[str, tuple[float, float]] = {}
        self._cache_lock = asyncio.Lock()

    def has_api_credentials(self) -> bool:
        return bool(self._token)

    async def get_route_metrics(self, route: RouteInput) -> RouteMetrics | None:
        if not self.has_api_credentials():
            return None

        try:
            points = [route.origin, *route.waypoints, route.destination]
            coords = await self._resolve_points(points)
            if len(coords) < 2:
                return None
            metrics = await self._request_directions(coords)
            return metrics
        except Exception:
            return None

    async def _resolve_points(self, points: list[str]) -> list[tuple[float, float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resolved: list[tuple[float, float]] = []
            for point in points:
                normalized = point.strip()
                if not normalized:
                    continue

                async with self._cache_lock:
                    cached = self._coord_cache.get(normalized.lower())
                if cached is not None:
                    resolved.append(cached)
                    continue

                coord = await self._geocode(client, normalized)
                if coord is None:
                    return []

                async with self._cache_lock:
                    self._coord_cache[normalized.lower()] = coord
                resolved.append(coord)
            return resolved

    async def _geocode(
        self,
        client: httpx.AsyncClient,
        text_query: str,
    ) -> tuple[float, float] | None:
        params = {
            "q": text_query,
            "limit": 1,
            "access_token": self._token,
        }
        response = await client.get(self._geocode_url, params=params)
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features") or []
        if not features:
            return None

        geometry = features[0].get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) != 2:
            return None

        lon, lat = float(coordinates[0]), float(coordinates[1])
        return lon, lat

    async def _request_directions(
        self,
        coords: list[tuple[float, float]],
    ) -> RouteMetrics | None:
        if len(coords) < 2:
            return None

        coordinates_part = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)
        url = f"{self._directions_base}/mapbox/driving-traffic/{coordinates_part}"
        params = {
            "alternatives": "false",
            "overview": "false",
            "access_token": self._token,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()

        routes = payload.get("routes") or []
        if not routes:
            return None

        primary = routes[0]
        distance_m = float(primary.get("distance") or 0.0)
        duration_s = float(primary.get("duration") or 0.0)
        if distance_m <= 0 or duration_s <= 0:
            return None

        return RouteMetrics(
            distance_km=round(distance_m / 1000.0, 3),
            duration_min=round(duration_s / 60.0, 3),
            source="mapbox_directions_driving_traffic",
        )


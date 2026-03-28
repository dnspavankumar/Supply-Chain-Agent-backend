from __future__ import annotations

import asyncio
import math
import re
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
    geometry_coords: list[tuple[float, float]]


class MapRoutingService:
    """Route enrichment service with Mapbox and OSRM fallback.

    Priority:
    1) Mapbox Geocoding + Directions when MAPBOX_ACCESS_TOKEN is set.
    2) Nominatim + OSRM public routing when token is absent.
    """

    OSRM_DIRECTIONS_URL = "https://router.project-osrm.org/route/v1/driving"
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    DIRECT_COORDINATE_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")
    AVERAGE_SPEED_KMH = 52.0
    FALLBACK_COORDS = {
        "new york": (40.7128, -74.0060),
        "los angeles": (34.0522, -118.2437),
        "chicago": (41.8781, -87.6298),
        "miami": (25.7617, -80.1918),
        "seattle": (47.6062, -122.3321),
        "boston": (42.3601, -71.0589),
        "mumbai": (19.0760, 72.8777),
        "hyderabad": (17.3850, 78.4867),
        "lingampally": (17.4857, 78.3242),
        "narsapur": (17.6761, 78.1012),
        "dubai": (25.2048, 55.2708),
        "hamburg": (53.5511, 9.9937),
        "frankfurt": (50.1109, 8.6821),
        "amsterdam": (52.3676, 4.9041),
        "rotterdam": (51.9244, 4.4777),
    }

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
        alternatives = await self.get_route_alternatives(route, max_alternatives=1)
        return alternatives[0] if alternatives else None

    async def get_route_alternatives(
        self,
        route: RouteInput,
        max_alternatives: int = 3,
    ) -> list[RouteMetrics]:
        points = [route.origin, *route.waypoints, route.destination]
        coords = await self._resolve_points(points)
        if len(coords) < 2:
            return []

        try:
            if self.has_api_credentials():
                payload = await self._request_mapbox_directions(coords, max_alternatives=max_alternatives)
                source = "mapbox_directions_driving_traffic"
            else:
                payload = await self._request_osrm_directions(coords, max_alternatives=max_alternatives)
                source = "osrm_driving"

            parsed = self._parse_routes(payload, source=source, max_alternatives=max_alternatives)
            if parsed:
                return parsed
        except Exception:
            pass

        return [self._build_fallback_metrics(coords=coords, source="geodesic_fallback")]

    async def _resolve_points(self, points: list[str]) -> list[tuple[float, float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resolved: list[tuple[float, float]] = []
            for point in points:
                normalized = point.strip()
                if not normalized:
                    continue

                direct = self._parse_direct_coordinates(normalized)
                if direct is not None:
                    resolved.append(direct)
                    continue

                async with self._cache_lock:
                    cached = self._coord_cache.get(normalized.lower())
                if cached is not None:
                    resolved.append(cached)
                    continue

                try:
                    if self.has_api_credentials():
                        coord = await self._geocode_mapbox(client, normalized)
                    else:
                        coord = await self._geocode_nominatim(client, normalized)
                except Exception:
                    coord = None

                if coord is None:
                    coord = self._fallback_coordinate(normalized)

                async with self._cache_lock:
                    self._coord_cache[normalized.lower()] = coord
                resolved.append(coord)
            return resolved

    async def _geocode_mapbox(
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

    async def _geocode_nominatim(
        self,
        client: httpx.AsyncClient,
        text_query: str,
    ) -> tuple[float, float] | None:
        params = {
            "q": text_query,
            "format": "jsonv2",
            "limit": 1,
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "SupplyChainAgent/1.0",
        }
        response = await client.get(self.NOMINATIM_URL, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            return None

        first = payload[0]
        lat = float(first.get("lat"))
        lon = float(first.get("lon"))
        return lon, lat

    async def _request_mapbox_directions(
        self,
        coords: list[tuple[float, float]],
        max_alternatives: int,
    ) -> dict[str, Any]:
        coordinates_part = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)
        url = f"{self._directions_base}/mapbox/driving-traffic/{coordinates_part}"
        params = {
            "alternatives": "true" if max_alternatives > 1 else "false",
            "overview": "full",
            "geometries": "geojson",
            "access_token": self._token,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def _request_osrm_directions(
        self,
        coords: list[tuple[float, float]],
        max_alternatives: int,
    ) -> dict[str, Any]:
        coordinates_part = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)
        url = f"{self.OSRM_DIRECTIONS_URL}/{coordinates_part}"
        params = {
            "alternatives": "true" if max_alternatives > 1 else "false",
            "overview": "full",
            "geometries": "geojson",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def _parse_direct_coordinates(self, text: str) -> tuple[float, float] | None:
        match = self.DIRECT_COORDINATE_PATTERN.match(text)
        if match is None:
            return None

        lat = float(match.group(1))
        lon = float(match.group(2))
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            return None
        return lat, lon

    def _fallback_coordinate(self, text_query: str) -> tuple[float, float]:
        normalized = text_query.strip().lower()
        known = self.FALLBACK_COORDS.get(normalized)
        if known is not None:
            return known

        hash_value = 0
        for char in normalized:
            hash_value = (hash_value * 31 + ord(char)) & 0xFFFFFFFF

        lat = -55.0 + float(hash_value % 11000) / 100.0
        lon = -170.0 + float((hash_value // 113) % 34000) / 100.0
        return (lat, lon)

    def _build_fallback_metrics(
        self,
        coords: list[tuple[float, float]],
        source: str,
    ) -> RouteMetrics:
        segment_distances = [
            self._haversine_km(start, end)
            for start, end in zip(coords, coords[1:])
        ]
        total_distance = sum(segment_distances)
        duration_min = (total_distance / self.AVERAGE_SPEED_KMH) * 60.0 if total_distance > 0 else 0.0
        geometry = self._densify_geometry(coords)

        return RouteMetrics(
            distance_km=round(total_distance, 3),
            duration_min=round(max(duration_min, 1.0), 3),
            source=source,
            geometry_coords=geometry,
        )

    def _densify_geometry(self, coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(coords) < 2:
            return coords

        output: list[tuple[float, float]] = [coords[0]]
        for start, end in zip(coords, coords[1:]):
            points = 8
            for step in range(1, points + 1):
                ratio = step / points
                lat = start[0] + (end[0] - start[0]) * ratio
                lon = start[1] + (end[1] - start[1]) * ratio
                output.append((lat, lon))
        return output

    def _haversine_km(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        lat1, lon1 = start
        lat2, lon2 = end
        r = 6371.0

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def _parse_routes(
        self,
        payload: dict[str, Any],
        source: str,
        max_alternatives: int,
    ) -> list[RouteMetrics]:
        routes = payload.get("routes") or []
        metrics: list[RouteMetrics] = []

        for route in routes[: max(1, max_alternatives)]:
            distance_m = float(route.get("distance") or 0.0)
            duration_s = float(route.get("duration") or 0.0)
            geometry = route.get("geometry") or {}
            coords = geometry.get("coordinates") or []
            geometry_coords: list[tuple[float, float]] = []
            for point in coords:
                if not isinstance(point, list) or len(point) < 2:
                    continue
                lon, lat = float(point[0]), float(point[1])
                geometry_coords.append((lat, lon))

            if distance_m <= 0 or duration_s <= 0:
                continue

            metrics.append(
                RouteMetrics(
                    distance_km=round(distance_m / 1000.0, 3),
                    duration_min=round(duration_s / 60.0, 3),
                    source=source,
                    geometry_coords=geometry_coords,
                )
            )

        return metrics

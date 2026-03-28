from __future__ import annotations

from config import get_settings


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_route_score(
    weather_score: float,
    road_safety_score: float,
    *,
    alpha: float | None = None,
    beta: float | None = None,
) -> float:
    settings = get_settings()
    alpha_value = float(settings.route_score_alpha if alpha is None else alpha)
    beta_value = float(settings.route_score_beta if beta is None else beta)

    weather = _clamp_01(weather_score)
    safety = _clamp_01(road_safety_score)
    score = (alpha_value * weather) + (beta_value * safety)
    return round(score, 6)


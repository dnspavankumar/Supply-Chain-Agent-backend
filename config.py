from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Supply Chain AI Agent", validation_alias="APP_NAME")
    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model_id: str = Field(
        default="gemini-2.0-flash-001",
        validation_alias=AliasChoices("GEMINI_MODEL_ID", "AGNO_MODEL"),
    )
    require_google_api_key: bool = Field(
        default=True,
        validation_alias="REQUIRE_GOOGLE_API_KEY",
    )
    route_score_alpha: float = Field(default=0.4, validation_alias="ROUTE_SCORE_ALPHA")
    route_score_beta: float = Field(default=0.6, validation_alias="ROUTE_SCORE_BETA")
    mapbox_access_token: str = Field(default="", validation_alias="MAPBOX_ACCESS_TOKEN")
    mapbox_directions_base_url: str = Field(
        default="https://api.mapbox.com/directions/v5",
        validation_alias="MAPBOX_DIRECTIONS_BASE_URL",
    )
    mapbox_geocoding_base_url: str = Field(
        default="https://api.mapbox.com/search/geocode/v6/forward",
        validation_alias="MAPBOX_GEOCODING_BASE_URL",
    )
    map_api_timeout_seconds: float = Field(default=8.0, validation_alias="MAP_API_TIMEOUT_SECONDS")
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        validation_alias="CORS_ALLOW_ORIGINS",
    )

    @field_validator("route_score_alpha", "route_score_beta")
    @classmethod
    def validate_route_weights(cls, value: float) -> float:
        numeric = float(value)
        if numeric < 0:
            raise ValueError("route score weights must be >= 0")
        return numeric

    @field_validator("map_api_timeout_seconds")
    @classmethod
    def validate_map_timeout(cls, value: float) -> float:
        numeric = float(value)
        if numeric <= 0:
            raise ValueError("map_api_timeout_seconds must be > 0")
        return numeric

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def parse_cors_allow_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ["*"]
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()

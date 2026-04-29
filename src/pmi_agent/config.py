"""Application configuration."""

from functools import lru_cache
import os

from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    BaseSettings = BaseModel  # type: ignore[assignment]
    SettingsConfigDict = None  # type: ignore[assignment]


class AppConfig(BaseSettings):
    """Runtime settings loaded from environment variables."""

    polymarket_base_url: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Base URL reserved for a future Polymarket integration.",
    )
    request_timeout_seconds: float = Field(default=10.0, ge=0.1)
    max_search_results: int = Field(default=20, ge=1, le=100)
    min_relevance_score: float = Field(default=0.15, ge=0.0, le=1.0)

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_prefix="PMI_",
            extra="ignore",
        )


@lru_cache
def get_config() -> AppConfig:
    """Return cached application configuration."""

    if SettingsConfigDict is not None:
        return AppConfig()

    return AppConfig(
        polymarket_base_url=os.getenv(
            "PMI_POLYMARKET_BASE_URL",
            "https://gamma-api.polymarket.com",
        ),
        request_timeout_seconds=float(os.getenv("PMI_REQUEST_TIMEOUT_SECONDS", "10")),
        max_search_results=int(os.getenv("PMI_MAX_SEARCH_RESULTS", "20")),
        min_relevance_score=float(os.getenv("PMI_MIN_RELEVANCE_SCORE", "0.15")),
    )

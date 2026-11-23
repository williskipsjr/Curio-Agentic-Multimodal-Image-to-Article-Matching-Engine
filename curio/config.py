"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Curio application settings."""

    # Application
    app_name: str = "Curio"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://curio:curio_dev_password@localhost:5432/curio"

    @property
    def async_database_url(self) -> str:
        """Ensure the URL uses the asyncpg driver."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    # Gemini API
    gemini_api_key: str = ""

    # Image APIs
    unsplash_access_key: str = ""
    pexels_api_key: str = ""

    # Cost tracking
    daily_budget_usd: float = 0.0
    model_cost_tracking: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

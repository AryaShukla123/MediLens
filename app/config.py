"""
Central place for all app configuration.
Reads values from the .env file (via pydantic-settings) so nothing
sensitive is hardcoded anywhere else in the codebase.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App ---
    secret_key: str
    environment: str = "development"

    # --- Database ---
    database_url: str

    # --- Auth ---
    access_token_expire_minutes: int = 60

    # --- Chatbot ---
    gemini_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Import this single instance anywhere you need config values:
#   from app.config import settings
#   settings.database_url
settings = Settings()
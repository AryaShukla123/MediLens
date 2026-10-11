from pydantic import field_validator
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
    # Gemini model used by the chatbot. Google retires models regularly
    # (gemini-2.5-flash already returns 404), so this is an env var:
    # if the chatbot ever starts failing with "model not found", change
    # GEMINI_MODEL in .env instead of editing code.
    gemini_model: str = "gemini-3.5-flash"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("database_url")
    @classmethod
    def normalise_database_url(cls, value: str) -> str:
        # Some hosts (Heroku, Railway, Supabase) hand out "postgres://..." URLs.
        # SQLAlchemy 1.4+ only understands "postgresql://", so translate it
        # rather than make every deployment fix it by hand.
        value = value.strip()
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://"):]
        return value


settings = Settings()
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


settings = Settings()
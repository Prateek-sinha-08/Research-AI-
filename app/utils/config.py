# app/utils/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field

class Settings(BaseSettings):
    # === Project Metadata ===
    PROJECT_NAME: str = "AutoResearch AI"
    DEBUG: bool = True

    # === Mistral AI ===
    MISTRAL_API_KEY: str

    # === Instructor Model (local or HF path) ===
    INSTRUCTOR_MODEL_PATH: str = "hkunlp/instructor-xl"

    # === PostgreSQL ===
    POSTGRES_USER: str 
    POSTGRES_PASSWORD: str 
    POSTGRES_DB: str 
    DB_HOST: str 
    DB_PORT: str 

    # === Redis ===
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"

    # === Computed Fields ===
    @computed_field
    @property
    def DB_URL(self) -> str:
        return f"postgres://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"

    @computed_field
    @property
    def REDIS_BROKER_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Export the settings instance
settings = Settings()

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Platform API"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    backend_cors_origins: str = "http://localhost:5173"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    vector_size: int = 1536

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        return "/" + value.strip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

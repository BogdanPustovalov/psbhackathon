from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT_DIR / ".env"
load_dotenv(ENV_FILE, override=False)


class Settings(BaseSettings):
    app_name: str = "RAG Platform API"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    backend_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    data_hack_output_dir: str = str(ROOT_DIR / "data_hack_output")

    chroma_persist_directory: str = "./data/chroma"
    chroma_collection: str = "documents"

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_model_path: str = ""
    embedding_cache_dir: str = ""
    embedding_device: str = "cpu"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-6-luna"
    temperature: float = 0.0
    max_tokens: int = 1000
    top_k: int = 50
    rrf_top_k: int = 50
    final_context_top_k: int = 15
    rrf_semantic_weight: float = 0.4
    rrf_bm25_weight: float = 0.6

    reranker_enabled: bool = True
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_cache_dir: str = ""
    reranker_device: str = "cpu"
    reranker_batch_size: int = 2
    reranker_max_length: int = 512
    reranker_candidates: int = 50
    reranker_top_n: int = 50

    model_config = SettingsConfigDict(env_file=ENV_FILE, case_sensitive=False, extra="ignore")

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

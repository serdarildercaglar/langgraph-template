"""
Configuration management with Pydantic Settings.
Supports environment-based config with validation.
"""

from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEV = "development"
    STAGING = "staging"
    PROD = "production"


class VLLMSettings(BaseSettings):
    """vLLM OpenAI-compatible API settings."""

    model_config = SettingsConfigDict(env_prefix="VLLM_")

    base_url: str = Field(default="http://localhost:8000/v1", description="vLLM API base URL")
    api_key: SecretStr = Field(default=SecretStr("EMPTY"), description="API key (usually EMPTY for vLLM)")
    model_name: str = Field(default="default-model", description="Model name deployed on vLLM")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class EmbeddingSettings(BaseSettings):
    """vLLM Embedding service settings."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    base_url: str = Field(default="http://localhost:8001/v1", description="Embedding API base URL")
    api_key: SecretStr = Field(default=SecretStr("EMPTY"))
    model_name: str = Field(default="bge-large-en-v1.5")
    batch_size: int = Field(default=32, gt=0, description="Batch size for async processing")
    max_concurrent: int = Field(default=5, gt=0, description="Max concurrent embedding requests")
    dimension: int = Field(default=1024, gt=0, description="Embedding dimension")


class MilvusSettings(BaseSettings):
    """Milvus vector database settings."""

    model_config = SettingsConfigDict(env_prefix="MILVUS_")

    host: str = Field(default="localhost")
    port: int = Field(default=19530)
    user: str = Field(default="")
    password: SecretStr = Field(default=SecretStr(""))
    database: str = Field(default="default")
    collection_name: str = Field(default="documents")
    batch_size: int = Field(default=100, gt=0, description="Batch size for async operations")
    max_concurrent: int = Field(default=3, gt=0)
    timeout: float = Field(default=30.0, gt=0)

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"


class LangfuseSettings(BaseSettings):
    """Langfuse observability settings."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")

    enabled: bool = Field(default=True)
    public_key: str = Field(default="")
    secret_key: SecretStr = Field(default=SecretStr(""))
    host: str = Field(default="https://cloud.langfuse.com")

    @field_validator("enabled", mode="before")
    @classmethod
    def validate_enabled(cls, v: Any, info: Any) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")


class CheckpointSettings(BaseSettings):
    """LangGraph checkpoint settings."""

    model_config = SettingsConfigDict(env_prefix="CHECKPOINT_")

    backend: str = Field(default="sqlite", description="sqlite, postgres, memory")
    sqlite_path: str = Field(default="./checkpoints.db")
    postgres_uri: str = Field(default="")


class APISettings(BaseSettings):
    """FastAPI settings."""

    model_config = SettingsConfigDict(env_prefix="API_")

    title: str = Field(default="LangGraph Agentic API")
    version: str = Field(default="0.1.0")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    debug: bool = Field(default=False)
    cors_origins: list[str] = Field(default=["*"])


class Settings(BaseSettings):
    """Root settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Field(default=Environment.DEV)

    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    api: APISettings = Field(default_factory=APISettings)

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PROD


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()

"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "CallClear"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    storage_dir: Path = Path("data")
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "callclear"
    default_sample_rate: int = 16000
    telephony_sample_rate: int = 8000
    asr_model: str = "base"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_language: str = "hi"
    asr_beam_size: int = 5
    hf_dataset: str = "google/fleurs"
    hf_config: str = "hi_in"
    benchmark_split: str = "test"
    benchmark_limit: int = 200
    packet_loss_ms: int = 60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def raw_dir(self) -> Path:
        """Return the raw-audio storage directory."""
        return self.storage_dir / "raw"

    @property
    def degraded_dir(self) -> Path:
        """Return the degraded-audio storage directory."""
        return self.storage_dir / "degraded"

    @property
    def enhanced_dir(self) -> Path:
        """Return the enhanced-audio storage directory."""
        return self.storage_dir / "enhanced"


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()

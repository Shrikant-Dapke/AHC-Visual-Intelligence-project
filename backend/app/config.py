"""Central configuration. Single source of truth for incident labels.

DO NOT invent the 12 class names here. They must come from dataset metadata
(labels.txt / classes.txt / dataset README). Until that file is supplied,
INCIDENT_CLASSES stays empty and inference returns an honest "unknown" result.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Placeholder — replace with the 12 real labels from the dataset, in order.
# Example (DO NOT use until confirmed):
# INCIDENT_CLASSES = ["collision", ...]  # must be exactly 12 entries
INCIDENT_CLASSES: list[str] = []

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AHC Visual Intelligence"
    api_prefix: str = "/api"
    upload_dir: Path = UPLOAD_DIR
    max_upload_mb: int = 200
    sample_fps: float = 1.0
    cors_origins: str = "http://localhost:3000"
    # Future: MODEL_NAME=yolov8n, MODEL_BACKEND=heuristic|yolo|clip
    model_backend: str = "heuristic"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings

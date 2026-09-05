"""Central configuration. Single source of truth for incident labels.

DO NOT invent the 12 class names here. They must come from dataset metadata
(labels.txt / classes.txt / dataset README). Until that file is supplied,
INCIDENT_CLASSES stays empty and inference returns an honest "unknown" result.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
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
    # Task 3: unified CLIP+YOLO analysis (all overrideable via env).
    clip_model_dir: Path = BASE_DIR / "models" / "clip_linear_v1"
    evidence_dir: Path = BASE_DIR / "data" / "evidence"
    yolo_enabled: bool = True
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    yolo_imgsz: int = 640
    yolo_sample_fps: float = 2.0
    yolo_tracker: str = "iou"  # "iou" | "bytetrack"
    yolo_max_frames: int = 600

    @field_validator("clip_model_dir", "evidence_dir", "upload_dir", mode="after")
    @classmethod
    def _resolve_against_repo(cls, v: Path) -> Path:
        """Relative env-supplied paths resolve against the repo root, so the
        app behaves identically regardless of the process working directory."""
        return v if v.is_absolute() else BASE_DIR / v


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.evidence_dir.mkdir(parents=True, exist_ok=True)
    return settings

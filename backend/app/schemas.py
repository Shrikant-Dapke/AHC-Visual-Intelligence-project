"""Pydantic schemas. Mirrors shared/types.ts — keep both in sync."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "ahc-visual-intelligence-backend"
    model_backend: str = "heuristic"
    labels_configured: bool = False


class EventSpan(BaseModel):
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    label: str
    score: float = Field(ge=0.0, le=1.0)


class AnalyzeResult(BaseModel):
    job_id: str
    incident_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    events: list[EventSpan] = []
    explanation: str
    thumbnail_urls: list[str] = []
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0


class ErrorResponse(BaseModel):
    detail: str

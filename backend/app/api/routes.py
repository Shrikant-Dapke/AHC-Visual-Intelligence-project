"""API routes. Keep contract stable; inference stays behind get_inference()."""

from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import INCIDENT_CLASSES, get_settings
from app.inference.predict import get_inference
from app.schemas import AnalyzeResult, EventSpan, HealthResponse
from app.services.explain import build_explanation
from app.services.video import cleanup, probe_video, sample_frames, save_upload

router = APIRouter()

# In-memory job store (no DB for demo). {job_id: AnalyzeResult}
JOBS: dict[str, AnalyzeResult] = {}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        model_backend=settings.model_backend,
        labels_configured=bool(INCIDENT_CLASSES),
    )


@router.post("/analyze", response_model=AnalyzeResult)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResult:
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 uploads are supported.")
    try:
        saved = await save_upload(
            file.read, file.filename, settings.upload_dir, settings.max_upload_mb
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = uuid4().hex[:12]
    try:
        meta = probe_video(saved)
        frames = sample_frames(meta, sample_fps=settings.sample_fps)
        inference = get_inference(settings.model_backend)
        output = inference.predict(frames, meta.duration_sec)
        explanation = build_explanation(output.incident_class, output.confidence, output, meta)
        result = AnalyzeResult(
            job_id=job_id,
            incident_class=output.incident_class,
            confidence=output.confidence,
            events=[
                EventSpan(t_start=e.t_start, t_end=e.t_end, label=e.label, score=e.score)
                for e in output.events
            ],
            explanation=explanation,
            thumbnail_urls=[],
            duration_sec=round(meta.duration_sec, 2),
            fps=round(meta.fps, 2),
            width=meta.width,
            height=meta.height,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        cleanup(saved)

    JOBS[job_id] = result
    return result


@router.get("/results/{job_id}", response_model=AnalyzeResult)
def get_result(job_id: str) -> AnalyzeResult:
    result = JOBS.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return result

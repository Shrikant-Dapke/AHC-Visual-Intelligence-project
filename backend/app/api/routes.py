"""API routes. Keep contract stable; inference stays behind get_inference().

Task 3: POST /api/analyze now runs the unified pipeline (CLIP primary +
YOLO supporting evidence) via app.services.unified. All original fields keep
their meaning; new fields are additive. Evidence JPEGs are served by the
minimal safe GET /api/evidence/{job_id}/{kind} route below.
"""

import re
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import INCIDENT_CLASSES, get_settings
from app.inference.clip import clip_status
from app.schemas import AnalyzeResult, HealthResponse
from app.services.unified import run_unified_analysis, yolo_status
from app.services.video import cleanup, save_upload

router = APIRouter()

# In-memory job store (no DB for demo). {job_id: AnalyzeResult}
JOBS: dict[str, AnalyzeResult] = {}

# Evidence filenames are fixed; job_id is server-generated hex. Both are
# whitelisted so no request can escape the evidence directory.
_EVIDENCE_KINDS = {"pre": "pre.jpg", "peak": "peak.jpg", "post": "post.jpg"}
_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    clip = clip_status(settings.clip_model_dir)
    yolo = yolo_status()
    return HealthResponse(
        model_backend=settings.model_backend,
        labels_configured=bool(INCIDENT_CLASSES),
        clip_available=bool(clip["available"]),
        clip_reason=None if clip["available"] else str(clip.get("reason")),
        yolo_available=bool(yolo["available"]),
        yolo_reason=None if yolo["available"] else str(yolo.get("reason")),
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
        result = run_unified_analysis(saved, job_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        cleanup(saved)  # uploaded mp4 deleted; evidence JPEGs persist

    JOBS[job_id] = result
    return result


@router.get("/results/{job_id}", response_model=AnalyzeResult)
def get_result(job_id: str) -> AnalyzeResult:
    result = JOBS.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return result


@router.get("/evidence/{job_id}/{kind}")
def get_evidence(job_id: str, kind: str):
    """Serve one evidence frame (pre/peak/post) for a job.

    Minimal safe media route: kind is whitelisted, job_id is restricted to a
    server-generated charset, and the resolved path is verified to stay inside
    the evidence root (path-traversal proof). Filesystem paths are never
    exposed; clients use the `url` fields from the analyze response.
    """
    settings = get_settings()
    filename = _EVIDENCE_KINDS.get(kind)
    if filename is None or not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Evidence not found.")
    root = settings.evidence_dir.resolve()
    full = (root / job_id / filename).resolve()
    if full.parent != root / job_id or not full.is_file():
        raise HTTPException(status_code=404, detail="Evidence not found.")
    return FileResponse(str(full), media_type="image/jpeg")

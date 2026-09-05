"""Unified video analysis (Task 3): CLIP primary + YOLO supporting evidence.

Flow per uploaded MP4:
    probe -> heuristic motion baseline (existing, unchanged)
    CLIP incident inference (primary; honest fallback when untrained)
    YOLO object analysis (supporting evidence; NEVER overwrites CLIP)
    unified AnalyzeResult (+ evidence JPEGs served via /api/evidence/...)

Performance rules (CPU-only dev box):
- ONE shared YOLO detector for the process (lazy singleton); a FRESH tracker
  per request (trackers hold per-video state).
- YOLO runs at settings.yolo_sample_fps (default 2.0), capped frames.
- CLIP embeds exactly 8 frames (frozen policy); encoder+classifier singletons.
- Frames stream one at a time; uploads are deleted after analysis while the
  small evidence JPEGs persist under settings.evidence_dir/<job_id>/.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

from app.config import get_settings
from app.inference.clip import ClipUnavailable, predict_clip
from app.inference.predict import get_inference
from app.schemas import (
    AnalyzeResult,
    EvidenceItem,
    EventSpan,
    ObjectCount,
    ObjectTimeline,
    ObjectTrack,
    TrackPoint,
)
from app.services.explain import build_explanation, describe_objects
from app.services.object_analysis import (
    IncidentInput,
    analyze_video_objects,
    write_analysis_json,
)
from app.services.video import VideoMeta, probe_video, sample_frames

REPO_ROOT = Path(__file__).resolve().parents[3]

_YOLO_LOCK = threading.Lock()
_detector = None
_detector_key: tuple | None = None


def _resolve_yolo_model(model_name: str) -> str:
    """Prefer the already-downloaded repo-root weights so a backend CWD of
    `backend/` does not trigger a redundant download."""
    from app.inference.detector import resolve_model_path

    return resolve_model_path(model_name)


def yolo_status() -> dict:
    """Cheap availability probe. Never loads weights."""
    settings = get_settings()
    if not settings.yolo_enabled:
        return {"available": False, "reason": "disabled"}
    if importlib.util.find_spec("ultralytics") is None:
        return {"available": False, "reason": "deps_missing",
                "detail": "ultralytics not installed"}
    return {"available": True, "reason": None,
            "model": _resolve_yolo_model(settings.yolo_model)}


def _get_detector():
    """Process-shared YOLO detector. Tracker stays per-request (stateful)."""
    global _detector, _detector_key
    settings = get_settings()
    key = (settings.yolo_model, settings.yolo_conf, settings.yolo_imgsz)
    if _detector is None or _detector_key != key:
        from app.inference.detector import YoloDetector

        det = YoloDetector(
            model=_resolve_yolo_model(settings.yolo_model),
            conf=settings.yolo_conf,
            imgsz=settings.yolo_imgsz,
        )
        det.load()
        _detector = det
        _detector_key = key
    return _detector


def _new_tracker():
    from app.inference.tracker import make_tracker

    settings = get_settings()
    if settings.yolo_tracker == "bytetrack":
        # Fresh instance per request: ByteTrack holds per-video MOT state and
        # ultralytics persist-state is not safely resettable between videos.
        tr = make_tracker(
            "bytetrack",
            model_name=_resolve_yolo_model(settings.yolo_model),
            conf=settings.yolo_conf, imgsz=settings.yolo_imgsz)
        tr.load()
        return tr
    return make_tracker("iou")


def _evidence_url(job_id: str, kind: str) -> str:
    return f"{get_settings().api_prefix}/evidence/{job_id}/{kind}"


def run_unified_analysis(saved: Path, job_id: str) -> AnalyzeResult:
    """Full pipeline for one saved upload. Raises ValueError on bad video."""
    settings = get_settings()
    warnings: list[str] = []

    # --- 1. probe + heuristic baseline (existing behavior, unchanged) ---
    meta: VideoMeta = probe_video(saved)
    frames = sample_frames(meta, sample_fps=settings.sample_fps)
    inference = get_inference(settings.model_backend)
    output = inference.predict(frames, meta.duration_sec)

    # --- 2. CLIP primary incident classification ---
    incident_class, confidence, source = output.incident_class, output.confidence, "heuristic"
    if output.incident_class == "unknown":
        source = "none"
    try:
        pred = predict_clip(saved, settings.clip_model_dir)
        incident_class, confidence, source = (
            pred.class_name, pred.confidence, "clip")
    except ClipUnavailable as e:
        warnings.append(f"CLIP unavailable ({e.reason}); using motion baseline.")
    except Exception as e:  # never let CLIP break the whole analysis
        warnings.append(
            f"CLIP error ({type(e).__name__}); using motion baseline.")

    # --- 3. YOLO supporting evidence (never overwrites the incident) ---
    obj: dict | None = None
    yolo_err: str | None = None
    if settings.yolo_enabled:
        try:
            with _YOLO_LOCK:  # one shared detector; serialize CPU inference
                obj = analyze_video_objects(
                    saved,
                    _get_detector(),
                    _new_tracker(),
                    sample_fps=settings.yolo_sample_fps,
                    max_frames=settings.yolo_max_frames,
                    evidence_dir=settings.evidence_dir,
                    evidence_name=job_id,
                    incident=IncidentInput(incident_class, confidence, source),
                    progress=False,
                )
            write_analysis_json(obj, settings.evidence_dir / job_id)
        except Exception as e:
            yolo_err = f"{type(e).__name__}: {e}"
            warnings.append(f"Object analysis unavailable ({yolo_err}).")
            obj = None
    else:
        yolo_err = "disabled via settings"
        warnings.append("Object analysis disabled via settings.")

    # --- 4. merge into the (backward-compatible) response ---
    explanation = build_explanation(
        incident_class, confidence, output, meta)
    if source == "clip":
        # build_explanation() attributes the label to the motion heuristic;
        # correct the record when the trained classifier supplied the verdict.
        explanation = (
            f"Incident label '{incident_class}' with confidence "
            f"{confidence:.2f} comes from the trained CLIP incident "
            f"classifier (frozen ViT-B/32 + logistic regression). "
            + explanation
        )
    thumb_urls: list[str] = []
    timeline = None
    objects: list[ObjectCount] = []
    tracks: list[ObjectTrack] = []
    evidence: list[EvidenceItem] = []
    if obj is not None:
        tl = obj["timeline"]
        timeline = ObjectTimeline(start=tl["start"], peak=tl["peak"], end=tl["end"])
        for o in obj["objects"]:
            objects.append(ObjectCount(**{"class": o["class"], "count": o["count"],
                                          "track_ids": o["track_ids"]}))
        for t in obj["tracks"]:
            tracks.append(ObjectTrack(**{
                "class": t["class"], "track_id": t["track_id"],
                "first_seen": t["first_seen"], "last_seen": t["last_seen"],
                "duration": t["duration"],
                "hits": t["hits"], "displacement_px": t["displacement_px"],
                "trajectory": [
                    TrackPoint(t=p["t"], frame=p["frame"],
                               cx=p["cx"], cy=p["cy"])
                    for p in t["trajectory"]]}))
        for ev in obj["evidence"]:
            evidence.append(EvidenceItem(
                kind=ev["kind"], timestamp=ev["timestamp"],
                frame_index=ev["frame_index"], path=ev["path"],
                url=_evidence_url(job_id, ev["kind"])))
            thumb_urls.append(_evidence_url(job_id, ev["kind"]))
        explanation = (explanation + " " + describe_objects(
            obj["objects"], len(obj["tracks"]), obj["frames_analyzed"],
            obj["sample_fps"], obj["timeline"], obj["detector"],
            yolo_err=None))
    else:
        explanation = (explanation + " " + describe_objects(
            [], 0, 0, settings.yolo_sample_fps,
            {"start": 0.0, "peak": 0.0, "end": 0.0},
            {"model": settings.yolo_model, "conf": settings.yolo_conf},
            yolo_err=yolo_err))

    return AnalyzeResult(
        job_id=job_id,
        incident_class=incident_class,
        confidence=confidence,
        incident_source=source,
        events=[EventSpan(t_start=e.t_start, t_end=e.t_end,
                          label=e.label, score=e.score)
                for e in output.events],
        explanation=explanation,
        thumbnail_urls=thumb_urls,
        duration_sec=round(meta.duration_sec, 2),
        fps=round(meta.fps, 2),
        width=meta.width,
        height=meta.height,
        timeline=timeline,
        objects=objects,
        tracks=tracks,
        evidence=evidence,
        warnings=warnings,
    )

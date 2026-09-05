"""CLIP incident classifier service (Task 3).

The PRIMARY incident classifier. Reuses the frozen preprocessing and the
`ClipEncoder` from `scripts/benchmark_embedding_models.py` -- no duplicated
inference logic. Importing that module is safe: it has no training code path
(importing never trains; training lives in `train_clip_classifier.py`, which
is never imported here).

Honesty rules:
- Model files are checked BEFORE any heavy import/load. While full training
  is still running (`models/clip_linear_v1/classifier.joblib` absent), every
  call raises `ClipUnavailable(reason="model_files_missing")` -- fast, no
  350 MB encoder load, no fake predictions.
- Callers fall back to the heuristic mapping and report
  `incident_source="heuristic"` (or `"none"` when unknown).
- YOLO object evidence must NEVER overwrite the CLIP verdict; the merge
  happens in `app.services.unified`.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = REPO_ROOT / "scripts"

_CLASSIFIER_FILE = "classifier.joblib"
_LABELS_FILE = "labels.json"

_lock = threading.Lock()
_encoder = None
_classifier = None
_classifier_dir: Path | None = None


class ClipUnavailable(Exception):
    """Raised whenever CLIP cannot predict. `reason` is machine-readable."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


@dataclass
class ClipPrediction:
    class_name: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)
    source: str = "clip"


def _ensure_scripts_on_path() -> None:
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))


def _missing_deps() -> list[str]:
    """Cheap importability probe (no heavy imports executed)."""
    return [m for m in ("joblib", "sklearn", "transformers", "torch")
            if importlib.util.find_spec(m) is None]


def _check_model_files(model_dir: Path) -> None:
    missing = [f for f in (_CLASSIFIER_FILE, _LABELS_FILE)
               if not (model_dir / f).is_file()]
    if missing:
        raise ClipUnavailable(
            "model_files_missing",
            f"CLIP classifier not trained yet (missing {', '.join(missing)} "
            f"in {model_dir}). Full training still running or not started.",
        )


def clip_status(model_dir: Path | str) -> dict:
    """Lightweight availability probe. Never loads weights (no 350 MB hit)."""
    model_dir = Path(model_dir)
    if not (model_dir / _CLASSIFIER_FILE).is_file() \
            or not (model_dir / _LABELS_FILE).is_file():
        return {"available": False, "reason": "model_files_missing",
                "model_dir": str(model_dir), "classes": []}
    deps = _missing_deps()
    if deps:
        return {"available": False, "reason": "deps_missing",
                "model_dir": str(model_dir), "classes": [],
                "detail": f"missing packages: {', '.join(deps)}"}
    try:
        _ensure_scripts_on_path()
        import json

        import joblib

        labels = json.loads((model_dir / _LABELS_FILE).read_text())
        clf = joblib.load(model_dir / _CLASSIFIER_FILE)
        classes = [str(c) for c in getattr(clf, "classes_", labels.values())]
        return {"available": True, "reason": None,
                "model_dir": str(model_dir), "classes": classes}
    except Exception as e:  # corrupt/partial artifacts (e.g. mid-write) -> unavailable
        return {"available": False, "reason": "model_unreadable",
                "model_dir": str(model_dir), "classes": [],
                "detail": f"{type(e).__name__}: {e}"}


def _get_encoder():
    global _encoder
    if _encoder is None:
        _ensure_scripts_on_path()
        try:
            from benchmark_embedding_models import ClipEncoder
        except ImportError as e:
            raise ClipUnavailable(
                "deps_missing",
                f"cannot import frozen CLIP encoder: {e}") from e
        try:
            enc = ClipEncoder()
            enc.load()
        except Exception as e:
            raise ClipUnavailable(
                "encoder_failed",
                f"CLIP encoder failed to load: {type(e).__name__}: {e}") from e
        _encoder = enc
    return _encoder


def _get_classifier(model_dir: Path):
    global _classifier, _classifier_dir
    if _classifier is None or _classifier_dir != model_dir:
        try:
            import joblib
        except ImportError as e:
            raise ClipUnavailable(
                "deps_missing", "joblib/scikit-learn not installed") from e
        try:
            _classifier = joblib.load(model_dir / _CLASSIFIER_FILE)
            _classifier_dir = model_dir
        except Exception as e:
            raise ClipUnavailable(
                "model_unreadable",
                f"cannot load classifier: {type(e).__name__}: {e}") from e
    return _classifier


def predict_clip(
    video_path: Path | str,
    model_dir: Path | str,
    top_k: int = 3,
) -> ClipPrediction:
    """Run the trained CLIP linear classifier on one video file.

    Same frozen preprocessing as training and `predict_video.py`: 8
    time-sampled letterboxed frames, mean-pooled CLIP embedding.
    Raises `ClipUnavailable` instead of ever fabricating a label.
    """
    import numpy as np

    video_path = Path(video_path)
    model_dir = Path(model_dir)
    if not video_path.is_file():
        raise ClipUnavailable("inference_failed",
                              f"video not found: {video_path}")
    _check_model_files(model_dir)  # fast path: no heavy work when untrained
    if _missing_deps():
        raise ClipUnavailable(
            "deps_missing",
            f"missing packages: {', '.join(_missing_deps())}")

    with _lock:  # serialize CPU-heavy inference; one shared encoder/classifier
        clf = _get_classifier(model_dir)
        enc = _get_encoder()
        _ensure_scripts_on_path()
        from benchmark_embedding_models import probe_duration, read_rgb_frames

        dur = probe_duration(video_path)
        if not dur:
            raise ClipUnavailable("inference_failed",
                                  "could not decode video (no duration)")
        frames = read_rgb_frames(video_path, dur)
        if frames is None:
            raise ClipUnavailable("inference_failed",
                                  "could not sample frames")
        try:
            x = enc.embed(frames).mean(axis=0, keepdims=True).astype("float32")
            proba = clf.predict_proba(x)[0]
        except Exception as e:
            raise ClipUnavailable(
                "inference_failed",
                f"classifier inference failed: {type(e).__name__}: {e}") from e

    order = np.argsort(proba)[::-1]
    top = int(order[0])
    classes = [str(c) for c in clf.classes_]
    return ClipPrediction(
        class_name=classes[top],
        confidence=round(float(proba[top]), 4),
        top_k=[(classes[int(i)], round(float(proba[int(i)]), 4))
               for i in order[: max(1, top_k)]],
        source="clip",
    )

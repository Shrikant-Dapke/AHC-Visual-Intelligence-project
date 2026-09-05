"""YOLO object detection layer (Task 2, Phase 2).

Pretrained YOLO only -- never trained from scratch here. CPU-compatible,
incremental (one frame at a time, never the whole video in RAM).

Design for future extension:
    Detector (Protocol)
    ├── YoloDetector        (ultralytics YOLOv8n default; this file)
    ├── GroundingDinoDetector  (FUTURE: open-vocabulary detection)
    └── SamRefiner           (FUTURE: mask refinement on top of boxes)

Usage:
    det = YoloDetector(model="yolov8n.pt", conf=0.35, imgsz=640)
    det.load()                      # downloads ~6 MB weights on first run
    for frame_bgr in frames:        # caller owns the VideoCapture loop
        dets = det.detect(frame_bgr, frame_index=i, timestamp=t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_model_path(model: str) -> str:
    """Resolve a bare weights name (e.g. "yolov8n.pt") against the repo root.

    Ultralytics resolves relative names against the process CWD, so a bare
    name used from `backend/` would trigger a redundant multi-minute
    re-download. An already-present repo-root copy is preferred; otherwise
    the name passes through and ultralytics downloads it once to the CWD.
    """
    p = Path(model)
    if p.is_file() or p.is_absolute():
        return model
    cached = REPO_ROOT / p.name
    return str(cached) if cached.is_file() else model


# COCO classes worth keeping for road/incident scenes. Everything else the
# pretrained model emits is dropped to keep evidence focused and artifacts small.
ROAD_CLASSES = frozenset({
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
})


@dataclass
class Detection:
    """One bounding box on one sampled frame."""

    class_name: str
    confidence: float
    # xyxy in original-frame pixels: [x1, y1, x2, y2]
    bbox: list[float]
    frame_index: int
    timestamp: float
    track_id: int | None = None  # filled in by the tracker layer


@dataclass
class DetectorConfig:
    model: str = "yolov8n.pt"
    conf: float = 0.35
    imgsz: int = 640
    classes: frozenset = field(default_factory=lambda: ROAD_CLASSES)


class Detector(Protocol):
    """Future detectors (Grounding DINO, ...) implement this interface."""

    def load(self) -> None: ...
    def detect(
        self, frame_bgr, frame_index: int, timestamp: float
    ) -> list[Detection]: ...
    def close(self) -> None: ...


class YoloDetector:
    """Thin CPU wrapper around an ultralytics YOLO model.

    - Lazy import: `ultralytics` (+ torch) is only required when this class
      is instantiated, so the rest of the backend keeps working without it.
    - `predict()` per frame (NOT `track()`): ID association lives in
      `tracker.py` so detection stays swappable (DINO/SAM later).
    """

    name = "yolo"

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.35,
        imgsz: int = 640,
        classes: frozenset | None = None,
    ) -> None:
        self.config = DetectorConfig(
            model=resolve_model_path(model), conf=conf, imgsz=imgsz,
            classes=classes or ROAD_CLASSES,
        )
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Install it with:\n"
                "    pip install -r backend/requirements.txt\n"
                "(backend venv python) -- it pulls the CPU torch build."
            ) from e
        # Weights resolve via resolve_model_path() (repo-root copy preferred);
        # only a genuinely first run downloads (~6 MB) once to the CWD.
        self._model = YOLO(self.config.model)
        # Force CPU even on machines that report CUDA; the dev box is CPU-only
        # and the demo must behave identically everywhere.
        try:
            self._model.to("cpu")
        except Exception:
            pass  # older ultralytics already defaults to CPU

    @property
    def class_names(self) -> dict[int, str]:
        if self._model is None:
            raise RuntimeError("Detector.load() must be called before inference.")
        return dict(self._model.names)

    def detect(self, frame_bgr, frame_index: int, timestamp: float) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("Detector.load() must be called before inference.")
        results = self._model.predict(
            frame_bgr,
            conf=self.config.conf,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        if not results:
            return []
        names = self._model.names
        out: list[Detection] = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = str(names[cls_id])
            if label not in self.config.classes:
                continue
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            out.append(Detection(
                class_name=label,
                confidence=round(float(box.conf[0]), 4),
                bbox=[round(v, 1) for v in xyxy],
                frame_index=frame_index,
                timestamp=round(float(timestamp), 3),
            ))
        return out

    def close(self) -> None:
        self._model = None

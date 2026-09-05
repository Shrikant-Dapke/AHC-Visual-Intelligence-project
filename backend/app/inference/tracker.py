"""Lightweight object tracking layer (Task 2, Phase 3).

Default: `IoUTracker` -- greedy max-IoU association per class. Zero extra
dependencies, deterministic, CPU-trivial. Good enough for a hackathon
prototype's persistent-object counts, trajectories and
appearance/disappearance evidence.

Future trackers implement the `Tracker` protocol:
    Tracker (Protocol)
    ├── IoUTracker        (default; this file)
    ├── ByteTrackTracker  (adapter over ultralytics built-in ByteTrack;
    │                      use --tracker bytetrack to select it)
    └── AotTracker        (FUTURE: AOT/DeAOT advanced VOS -- only if needed)

Only detections of the SAME class are ever matched to each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TrackPoint:
    t: float
    frame_index: int
    cx: float  # bbox centre x (pixels, original frame scale)
    cy: float  # bbox centre y


@dataclass
class Track:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    first_frame: int
    last_frame: int
    hits: int = 1
    last_confidence: float = 0.0
    last_bbox: list[float] = field(default_factory=list)
    trajectory: list[TrackPoint] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.last_seen - self.first_seen, 3)

    def displacement(self) -> float:
        """Pixel distance between first and last centres (approx. movement)."""
        if len(self.trajectory) < 2:
            return 0.0
        a, b = self.trajectory[0], self.trajectory[-1]
        return round(((b.cx - a.cx) ** 2 + (b.cy - a.cy) ** 2) ** 0.5, 1)


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centre(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


class Tracker(Protocol):
    """Future trackers (AOT/DeAOT, ...) implement this interface."""

    def update(self, detections, frame_index: int, timestamp: float): ...
    def tracks(self) -> list[Track]: ...
    def reset(self) -> None: ...


class IoUTracker:
    """Greedy per-class IoU tracker.

    Args:
        min_iou: minimum IoU to associate a detection with an open track.
        max_age: drop an open track after this many sampled frames with no hit.
    """

    name = "iou"

    def __init__(self, min_iou: float = 0.3, max_age: int = 5) -> None:
        self.min_iou = min_iou
        self.max_age = max_age
        self._open: dict[int, Track] = {}
        self._missed: dict[int, int] = {}
        self._finished: list[Track] = []
        self._next_id = 1

    def update(self, detections, frame_index: int, timestamp: float):
        # Group open tracks by class so matching stays per-class.
        for det in detections:
            best_id, best_iou = None, self.min_iou
            for tid, tr in self._open.items():
                if tr.class_name != det.class_name:
                    continue
                score = iou(tr.last_bbox, det.bbox)
                if score > best_iou:
                    best_id, best_iou = tid, score
            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                cx, cy = _centre(det.bbox)
                tr = Track(
                    track_id=tid, class_name=det.class_name,
                    first_seen=timestamp, last_seen=timestamp,
                    first_frame=frame_index, last_frame=frame_index,
                    hits=1, last_confidence=det.confidence,
                    last_bbox=list(det.bbox),
                    trajectory=[TrackPoint(timestamp, frame_index, cx, cy)],
                )
                self._open[tid] = tr
                self._missed[tid] = 0
                det.track_id = tid
            else:
                tr = self._open[best_id]
                cx, cy = _centre(det.bbox)
                tr.last_seen = timestamp
                tr.last_frame = frame_index
                tr.hits += 1
                tr.last_confidence = det.confidence
                tr.last_bbox = list(det.bbox)
                tr.trajectory.append(TrackPoint(timestamp, frame_index, cx, cy))
                self._missed[best_id] = 0
                det.track_id = best_id

        matched = {d.track_id for d in detections if d.track_id is not None}
        for tid in list(self._open):
            if tid not in matched:
                self._missed[tid] = self._missed.get(tid, 0) + 1
                if self._missed[tid] > self.max_age:
                    self._finished.append(self._open.pop(tid))
                    self._missed.pop(tid, None)
        return detections

    def tracks(self) -> list[Track]:
        return sorted(
            list(self._finished) + list(self._open.values()),
            key=lambda t: t.track_id,
        )

    def reset(self) -> None:
        self._open.clear()
        self._missed.clear()
        self._finished.clear()
        self._next_id = 1


class ByteTrackTracker:
    """Adapter over ultralytics' built-in ByteTrack.

    Uses `model.track(frame, persist=True)` so ID association is done by the
    detection stack itself. Requires the same `ultralytics` install as
    `YoloDetector`. Only used when explicitly selected (`--tracker bytetrack`);
    `IoUTracker` remains the default.
    """

    name = "bytetrack"

    def __init__(self, model_name: str = "yolov8n.pt",
                 conf: float = 0.35, imgsz: int = 640) -> None:
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz
        self._model = None
        self._tracks: dict[int, Track] = {}

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Install it with:\n"
                "    pip install -r backend/requirements.txt"
            ) from e
        self._model = YOLO(self._resolve_weights(self.model_name))
        try:
            self._model.to("cpu")
        except Exception:
            pass

    @staticmethod
    def _resolve_weights(model_name: str) -> str:
        try:
            from app.inference.detector import resolve_model_path
        except ImportError:  # repo-root invocation
            from backend.app.inference.detector import resolve_model_path
        return resolve_model_path(model_name)

    def update(self, frame_bgr, frame_index: int, timestamp: float):
        """NOTE: unlike IoUTracker, this takes the raw frame (not detections)
        because tracking happens inside the YOLO model call."""
        if self._model is None:
            raise RuntimeError("ByteTrackTracker.load() must be called first.")
        try:
            from app.inference.detector import ROAD_CLASSES  # backend/ workdir
        except ImportError:  # repo-root invocation
            try:
                from backend.app.inference.detector import ROAD_CLASSES
            except ImportError:  # last-resort local fallback
                ROAD_CLASSES = frozenset({
                    "person", "bicycle", "car", "motorcycle",
                    "bus", "truck", "traffic light", "stop sign",
                })
        names = self._model.names
        res = self._model.track(
            frame_bgr, persist=True, conf=self.conf,
            imgsz=self.imgsz, verbose=False,
        )[0]
        out = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            # NOTE: on the first frames with detections ByteTrack has no
            # confirmed IDs yet (boxes.id is None). Emit those as
            # detection-only (track_id=None) instead of dropping them,
            # otherwise early evidence is silently lost.
            ids = boxes.id
            tids = [int(i) for i in ids] if ids is not None else [None] * len(boxes)
            try:
                from app.inference.detector import Detection
            except ImportError:
                from backend.app.inference.detector import Detection
            for box, tid in zip(boxes, tids):
                label = str(names[int(box.cls[0])])
                if label not in ROAD_CLASSES:
                    continue
                xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                det = Detection(
                    class_name=label, confidence=round(float(box.conf[0]), 4),
                    bbox=xyxy, frame_index=frame_index,
                    timestamp=round(float(timestamp), 3),
                    track_id=tid,
                )
                out.append(det)
                if tid is None:
                    continue  # ID not confirmed yet; track starts on next frames
                cx, cy = _centre(xyxy)
                tr = self._tracks.get(int(tid))
                if tr is None or tr.class_name != label:
                    self._tracks[int(tid)] = Track(
                        track_id=int(tid), class_name=label,
                        first_seen=timestamp, last_seen=timestamp,
                        first_frame=frame_index, last_frame=frame_index,
                        hits=1, last_confidence=det.confidence,
                        last_bbox=xyxy,
                        trajectory=[TrackPoint(timestamp, frame_index, cx, cy)],
                    )
                else:
                    tr.last_seen = timestamp
                    tr.last_frame = frame_index
                    tr.hits += 1
                    tr.last_confidence = det.confidence
                    tr.last_bbox = xyxy
                    tr.trajectory.append(TrackPoint(timestamp, frame_index, cx, cy))
        return out

    def tracks(self) -> list[Track]:
        return sorted(self._tracks.values(), key=lambda t: t.track_id)

    def reset(self) -> None:
        self._tracks.clear()


def make_tracker(name: str, **kwargs) -> Tracker:
    if name == "iou":
        return IoUTracker(**kwargs)
    if name == "bytetrack":
        return ByteTrackTracker(**kwargs)  # type: ignore[return-value]
    raise ValueError(f"Unknown tracker '{name}'. Expected 'iou' or 'bytetrack'.")

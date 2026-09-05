"""Object + temporal intelligence orchestration (Task 2, Phases 4-6).

Pipeline (all incremental, CPU-friendly):
    probe video -> sample N fps -> YOLO detect per sampled frame
      -> track across frames -> object-activity scoring
      -> incident window (start/peak/end) -> save pre/peak/post frames
      -> unified analysis dict (+ analysis.json)

IMPORTANT HONESTY RULE (Phase 4): YOLO does NOT understand "collision",
"fire", etc. The `timeline` here is an *object-activity window* -- where
road-scene objects are most present/active -- and only *enriches* the CLIP
incident classifier, which remains the primary incident prediction. The
`incident` block is passed through verbatim (from CLIP once training
finishes) and is NEVER invented by this layer: without a supplied incident
it reports class "unknown", confidence 0.0, source "none".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class IncidentInput:
    class_name: str = "unknown"
    confidence: float = 0.0
    source: str = "none"  # "clip" | "passthrough" | "none"


def probe(path: Path) -> dict[str, float]:
    """Lightweight cv2 probe (mirrors backend/app/services/video.py)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Unreadable video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
        if count <= 0:
            raise ValueError("Zero-frame video: no decodable frames found.")
        if fps <= 0:
            fps = 30.0
        return {"fps": fps, "frame_count": count, "duration": count / fps}
    finally:
        cap.release()


def frame_activity_score(detections) -> float:
    """Object-activity score for one sampled frame (evidence, NOT classification).

    score = sum(conf) + 0.5 per person (vulnerable road user bonus)
            + 0.5 per detection beyond the first two (crowding bonus).
    """
    if not detections:
        return 0.0
    score = sum(d.confidence for d in detections)
    n_person = sum(1 for d in detections if d.class_name == "person")
    score += 0.5 * n_person
    score += 0.5 * max(0, len(detections) - 2)
    return round(float(score), 4)


def smooth(scores: list[float], width: int = 3) -> list[float]:
    if not scores:
        return []
    half = width // 2
    out = []
    for i in range(len(scores)):
        seg = scores[max(0, i - half): i + half + 1]
        out.append(sum(seg) / len(seg))
    return out


def extract_incident_window(
    times: list[float],
    scores: list[float],
    duration: float,
    min_width: float = 6.0,
    ratio: float = 0.4,
) -> dict[str, float]:
    """Expand around the peak while the smoothed score stays >= ratio*peak.

    Falls back to the middle third when nothing was ever detected (all-zero
    scores) so callers always get a valid, honest window.
    """
    if not times:
        raise ValueError("No sampled frames to derive a window from.")
    if max(scores) <= 0:
        start = round(duration / 3.0, 2)
        end = round(2.0 * duration / 3.0, 2)
        return {"start": start, "peak": round((start + end) / 2.0, 2), "end": end}
    peak_i = max(range(len(scores)), key=lambda i: scores[i])
    peak, thresh = times[peak_i], scores[peak_i] * ratio
    a, b = peak_i, peak_i
    while a > 0 and scores[a - 1] >= thresh:
        a -= 1
    while b < len(scores) - 1 and scores[b + 1] >= thresh:
        b += 1
    start, end = times[a], times[b]
    # Enforce a minimum width for a useful pre/peak/post strip.
    if end - start < min_width:
        mid = (start + end) / 2.0
        start = max(0.0, mid - min_width / 2.0)
        end = min(duration, mid + min_width / 2.0)
        if end - start < min_width:  # very short clip: cover it all
            start, end = 0.0, duration
    peak = min(max(peak, start), end)
    return {
        "start": round(start, 2),
        "peak": round(peak, 2),
        "end": round(end, 2),
    }


def save_evidence_frames(
    video_path: Path, window: dict[str, float], out_dir: Path
) -> list[dict[str, Any]]:
    """Seek-read exactly 3 frames (pre/peak/post) and write them as JPEGs."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Unreadable video during evidence export: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        items = [
            ("pre", window["start"]),
            ("peak", window["peak"]),
            ("post", window["end"]),
        ]
        evidence = []
        for kind, t in items:
            idx = min(max(0, int(round(t * fps))), max(0, count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:  # fallback: grab whatever is current
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f"Could not read evidence frame at t={t}s")
            dest = out_dir / f"{kind}.jpg"
            cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            evidence.append({
                "kind": kind,
                "timestamp": round(float(t), 2),
                "frame_index": idx,
                "path": str(dest),
            })
        return evidence
    finally:
        cap.release()


def analyze_video_objects(
    video_path: Path,
    detector,
    tracker,
    sample_fps: float = 2.0,
    max_frames: int = 600,
    evidence_dir: Path | None = None,
    incident: IncidentInput | None = None,
    progress: bool = False,
    evidence_name: str | None = None,
) -> dict[str, Any]:
    """Run the full object-intelligence pipeline on one video.

    Frames are streamed one at a time (never the whole video in RAM).
    Returns the unified analysis dict (Phase 6 schema).

    `evidence_name` overrides the per-video artifact subdir (defaults to the
    video stem). The backend passes the job_id so concurrent uploads of the
    same filename never collide.
    """
    import cv2

    video_path = Path(video_path)
    meta = probe(video_path)
    native_fps, duration = meta["fps"], meta["duration"]
    step = max(1, int(round(native_fps / max(sample_fps, 0.25))))

    incident = incident or IncidentInput()
    video_id = video_path.stem
    # evidence_dir is the artifact ROOT; per-video subdir is always appended
    # so CLI and service callers get data/evidence/<video_id>/{pre,peak,post}.jpg.
    ev_dir = Path(evidence_dir or Path("data/evidence")) / (evidence_name or video_id)

    # --- incremental detect + track loop ---
    # ByteTrack consumes raw frames (tracking inside the model call);
    # the default IoU tracker consumes per-frame detections.
    use_bytetrack = getattr(tracker, "name", "iou") == "bytetrack"
    cap = cv2.VideoCapture(str(video_path))
    per_frame: list[dict[str, Any]] = []
    n_analyzed = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Unreadable video: {video_path}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = idx / native_fps
                if use_bytetrack:
                    dets = tracker.update(frame, idx, t)
                else:
                    dets = detector.detect(frame, idx, t)
                    tracker.update(dets, idx, t)
                per_frame.append({
                    "timestamp": round(t, 3),
                    "frame_index": idx,
                    "activity": frame_activity_score(dets),
                    "objects": [
                        {"class": d.class_name, "confidence": d.confidence,
                         "bbox": d.bbox, "track_id": d.track_id}
                        for d in dets
                    ],
                })
                n_analyzed += 1
                if progress and n_analyzed % 25 == 0:
                    print(f"  ... {n_analyzed} frames analyzed", flush=True)
                if n_analyzed >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    if not per_frame:
        raise ValueError("Zero-frame video: sampling produced no frames.")

    # --- temporal evidence ---
    times = [f["timestamp"] for f in per_frame]
    raw = [f["activity"] for f in per_frame]
    scores = smooth(raw)
    for f, s in zip(per_frame, scores):
        f["activity_smooth"] = round(s, 4)
    window = extract_incident_window(times, scores, duration)
    evidence = save_evidence_frames(video_path, window, ev_dir)

    # --- persistent-object summary ---
    tracks = tracker.tracks()
    by_class: dict[str, set[int]] = {}
    for tr in tracks:
        by_class.setdefault(tr.class_name, set()).add(tr.track_id)
    objects = [
        {"class": cls, "count": len(ids),
         "track_ids": sorted(ids)}
        for cls, ids in sorted(by_class.items())
    ]
    track_list = [
        {"track_id": tr.track_id, "class": tr.class_name,
         "first_seen": round(tr.first_seen, 2),
         "last_seen": round(tr.last_seen, 2),
         "duration": tr.duration, "hits": tr.hits,
         "displacement_px": tr.displacement(),
         "trajectory": [
             {"t": round(p.t, 2), "frame": p.frame_index,
              "cx": round(p.cx, 1), "cy": round(p.cy, 1)}
             for p in tr.trajectory
         ]}
        for tr in tracks
    ]

    return {
        "video_id": video_id,
        "source": str(video_path),
        "duration_sec": round(duration, 2),
        "fps": round(native_fps, 2),
        "frames_analyzed": n_analyzed,
        "sample_fps": sample_fps,
        "detector": {"name": getattr(detector, "name", "yolo"),
                     "model": getattr(getattr(detector, "config", None),
                                      "model", ""),
                     "conf": getattr(getattr(detector, "config", None),
                                     "conf", None),
                     "imgsz": getattr(getattr(detector, "config", None),
                                      "imgsz", None)},
        "tracker": getattr(tracker, "name", "iou"),
        "incident": {"class": incident.class_name,
                     "confidence": incident.confidence,
                     "source": incident.source},
        "timeline": {"start": window["start"], "peak": window["peak"],
                     "end": window["end"]},
        "objects": objects,
        "tracks": track_list,
        "evidence": evidence,
        "frames": per_frame,
    }


def write_analysis_json(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "analysis.json"
    dest.write_text(json.dumps(result, indent=2))
    return dest

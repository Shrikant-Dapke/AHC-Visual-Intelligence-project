"""Temporal event detection over the FROZEN CLIP classifier (Task 12).

Turns the video-level CLIP incident classifier into a temporal event
detector WITHOUT retraining anything:

    coarse scan (N-sec windows, K frames each, mean-pooled CLIP + predict_proba)
      -> per-class score series over time
      -> threshold + merge + min-duration -> candidate spans
      -> fine scan (1-sec windows) around candidates to refine boundaries
      -> optional YOLO/track support boost (modular weights, default 0)
      -> events: {class_name, start, end, confidence, explanation}

All timestamps come from the scan itself -- never fabricated, never copied
from ground truth. GT lives ONLY in the eval harness (scripts/eval_temporal.py).

CPU-friendly: one shared encoder + classifier (caller-owned singletons via
get_temporal_stack), frames embedded in batches, one YOLO pass per video max.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = REPO_ROOT / "scripts"


def _ensure_scripts_on_path() -> None:
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))


@dataclass
class TemporalConfig:
    """All tunables in one place. Tune THESE on T025-T034, nothing else."""

    coarse_sec: float = 3.0
    coarse_frames: int = 2
    fine_sec: float = 1.0
    fine_frames: int = 3
    fine_pad_sec: float = 6.0  # refine ±pad around each candidate span
    smooth_width: int = 3  # moving-average over coarse windows
    min_duration: float = 4.0  # drop shorter spans (noise suppression)
    merge_gap: float = 6.0  # join same-class spans separated by <= gap
    default_threshold: float = 0.45
    thresholds: dict = field(default_factory=dict)  # class -> float
    # YOLO fusion weights (additive boost on fused score; 0 = CLIP only).
    w_block_still: float = 0.0  # vehicle_blocking_traffic: still-vehicle support
    w_cong_density: float = 0.0  # traffic_congestion: vehicle-density support
    # Minimum peak confidence for an event to survive (after fusion).
    min_peak: float = 0.35
    # Score readout: "proba" (0..1) or "logit" (raw decision values, wider
    # dynamic range when proba saturates). Same frozen classifier either way.
    score_mode: str = "proba"
    # Adaptive threshold: max(fixed threshold, quantile_q of the video's own
    # score series). Handles saturated videos (whole clip scores ~1.0) by
    # detecting local surges instead. 0.0 = off.
    adaptive_quantile: float = 0.0
    # Top-1 gate: a window contributes to class C only when C is the
    # window's top-1 class. Cuts cross-talk between similar classes.
    top1_gate: bool = False

    def threshold_for(self, cls: str) -> float:
        return float(self.thresholds.get(cls, self.default_threshold))


@dataclass
class WindowScore:
    start: float
    end: float
    probs: dict  # class -> probability
    logits: dict = None  # class -> raw decision value (None = proba-only)
    top: str = ""
    top_p: float = 0.0

    def __post_init__(self):
        if self.logits is None:
            self.logits = {}


@dataclass
class TemporalEvent:
    class_name: str
    start: float
    end: float
    confidence: float  # peak fused score inside the span
    peak_time: float = 0.0
    explanation: str = ""
    support: dict = field(default_factory=dict)  # measured signals, no GT
    fallback: bool = False  # True if whole-video fallback span


class TemporalStack:
    """Shared frozen encoder + classifier. Load once, reuse everywhere."""

    def __init__(self, model_dir: Path | str):
        _ensure_scripts_on_path()
        import joblib

        from benchmark_embedding_models import ClipEncoder

        self.model_dir = Path(model_dir)
        self.encoder = ClipEncoder()
        self.encoder.load()
        self.classifier = joblib.load(self.model_dir / "classifier.joblib")
        self.classes = [str(c) for c in self.classifier.classes_]

    def proba(self, frames: list) -> dict:
        """Mean-pooled CLIP embedding -> {class: prob} for one frame set."""
        import numpy as np

        x = self.encoder.embed(frames).mean(axis=0, keepdims=True).astype("float32")
        p = self.classifier.predict_proba(x)[0]
        return {c: round(float(v), 4) for c, v in zip(self.classes, p)}

    def scores(self, frames: list) -> tuple[dict, dict]:
        """Same frozen classifier, two readouts: probabilities AND raw
        decision logits. Logits have far more dynamic range when predict_proba
        saturates near 0/1 (common on window slices). No retraining."""
        import numpy as np

        x = self.encoder.embed(frames).mean(axis=0, keepdims=True).astype("float32")
        p = self.classifier.predict_proba(x)[0]
        try:
            z = self.classifier.decision_function(x)[0]
        except Exception:
            z = np.zeros_like(p)
        probs = {c: round(float(v), 4) for c, v in zip(self.classes, p)}
        logits = {c: round(float(v), 4) for c, v in zip(self.classes, z)}
        return probs, logits


def get_temporal_stack(model_dir: Path | str) -> TemporalStack:
    return TemporalStack(model_dir)


def sample_window_frames(video_path: Path, start: float, end: float, n: int) -> list | None:
    """Evenly spaced RGB letterboxed frames inside [start, end]."""
    _ensure_scripts_on_path()
    import cv2
    import numpy as np

    from benchmark_embedding_models import CANON_H, CANON_W, letterbox

    if end <= start or n <= 0:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        out = []
        for i in range(n):
            frac = (i + 0.5) / n
            t = start + frac * (end - start)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            chs = [letterbox(rgb[:, :, c]) for c in range(3)]
            canon = np.stack(chs, axis=-1)
            if canon.shape != (CANON_H, CANON_W, 3):
                return None
            out.append(canon)
        return out
    finally:
        cap.release()


def scan_windows(
    video_path: Path,
    duration: float,
    stack: TemporalStack,
    window_sec: float,
    frames_per_window: int,
    batch: int = 16,
    progress: bool = False,
    span: tuple[float, float] | None = None,
) -> list[WindowScore]:
    """Non-overlapping windows over span (default [0, duration))."""
    import numpy as np

    lo, hi = (0.0, duration) if span is None else span
    lo, hi = max(0.0, lo), min(duration, hi)
    if hi - lo < 0.5:
        return []
    n = max(1, int((hi - lo) // window_sec))
    bounds = []
    for i in range(n):
        s = round(lo + i * window_sec, 2)
        e = round(min(lo + (i + 1) * window_sec, hi), 2)
        bounds.append((s, e))
    if hi - (lo + n * window_sec) >= 1.0:
        bounds.append((round(lo + n * window_sec, 2), round(hi, 2)))

    # Collect frame sets first, embed in batches (one encoder call per batch).
    jobs: list[tuple[float, float, list]] = []
    for s, e in bounds:
        frames = sample_window_frames(video_path, s, e, frames_per_window)
        if frames:
            jobs.append((s, e, frames))
    out: list[WindowScore] = []
    for i in range(0, len(jobs), batch):
        chunk = jobs[i: i + batch]
        all_frames = [f for _, _, fr in chunk for f in fr]
        E = stack.encoder.embed(all_frames)
        k = frames_per_window
        for j, (s, e, _) in enumerate(chunk):
            x = E[j * k: (j + 1) * k].mean(axis=0, keepdims=True).astype("float32")
            p = stack.classifier.predict_proba(x)[0]
            try:
                z = stack.classifier.decision_function(x)[0]
            except Exception:
                z = np.zeros_like(p)
            probs = {c: round(float(v), 4) for c, v in zip(stack.classes, p)}
            logits = {c: round(float(v), 4) for c, v in zip(stack.classes, z)}
            top_i = int(np.argsort(p)[::-1][0])
            out.append(WindowScore(start=s, end=e, probs=probs, logits=logits,
                                   top=stack.classes[top_i],
                                   top_p=probs[stack.classes[top_i]]))
        if progress:
            print(f"  ... {min(i + batch, len(jobs))}/{len(jobs)} windows",
                  flush=True)
    out.sort(key=lambda w: w.start)
    return out


def readout(w: WindowScore, cls: str, mode: str) -> float:
    """Score readout honoring cfg.score_mode (proba or logit)."""
    if mode == "logit":
        return float(w.logits.get(cls, 0.0)) if w.logits else 0.0
    return float(w.probs.get(cls, 0.0))


def smooth_series(windows: list[WindowScore], cls: str, width: int,
                  mode: str = "proba", top1_gate: bool = False) -> list[float]:
    vals = []
    for w in windows:
        v = readout(w, cls, mode)
        if top1_gate and w.top != cls:
            v = 0.0
        vals.append(v)
    if width <= 1 or not vals:
        return vals
    half = width // 2
    return [sum(vals[max(0, i - half): i + half + 1])
            / len(vals[max(0, i - half): i + half + 1])
            for i in range(len(vals))]


def adaptive_threshold(scores: list[float], fixed: float, q: float) -> float:
    """max(fixed, q-quantile of series). No-op when q <= 0."""
    if q <= 0 or not scores:
        return fixed
    s = sorted(scores)
    return max(fixed, s[min(len(s) - 1, int(q * len(s)))])


def spans_from_scores(
    windows: list[WindowScore],
    scores: list[float],
    threshold: float,
    merge_gap: float,
    min_duration: float,
) -> list[tuple[float, float, float, float]]:
    """Contiguous above-threshold runs -> merged (start, end, peak, peak_t)."""
    runs: list[tuple[float, float, float, float]] = []
    cur = None  # [start, end, peak, peak_t]
    for w, s in zip(windows, scores):
        if s >= threshold:
            if cur is None:
                cur = [w.start, w.end, s, (w.start + w.end) / 2]
            else:
                cur[1] = w.end
                if s > cur[2]:
                    cur[2] = s
                    cur[3] = (w.start + w.end) / 2
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    # Merge runs separated by small gaps; drop short noise.
    merged: list[list] = []
    for s, e, p, pt in runs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
            if p > merged[-1][2]:
                merged[-1][2] = p
                merged[-1][3] = pt
        else:
            merged.append([s, e, p, pt])
    return [(round(s, 2), round(e, 2), round(p, 4), round(pt, 2))
            for s, e, p, pt in merged if e - s >= min_duration]


def fuse_scores(
    windows: list[WindowScore],
    cls: str,
    support: dict | None,
    cfg: TemporalConfig,
) -> list[float]:
    """CLIP series + optional modular YOLO support boost (weights tunable)."""
    base = smooth_series(windows, cls, cfg.smooth_width, cfg.score_mode,
                         cfg.top1_gate)
    if not support:
        return base
    w = 0.0
    key = ""
    if cls == "vehicle_blocking_traffic" and cfg.w_block_still > 0:
        w, key = cfg.w_block_still, "stillness"
    elif cls == "traffic_congestion" and cfg.w_cong_density > 0:
        w, key = cfg.w_cong_density, "density"
    if w <= 0 or not key:
        return base
    fused = [b + w * support.get(key, {}).get(_wkey(windows[i]), 0.0)
             for i, b in enumerate(base)]
    if cfg.score_mode == "proba":
        fused = [min(1.0, v) for v in fused]
    return fused


def _wkey(w: WindowScore) -> tuple:
    return (w.start, w.end)


def yolo_window_support(
    tracks: list,
    windows: list[WindowScore],
    frame_w: int = 1280,
    frame_h: int = 720,
) -> dict:
    """Per-window support signals from ALREADY-COMPUTED tracks (no re-run).

    Returns {"stillness": {(s,e): 0..1}, "density": {...}, "person": {...}}.
    - stillness: best overlap-weighted (1 - motion) of a persistent vehicle.
    - density: unique vehicles active in window / 8 (capped at 1).
    - person: 1.0 if any person track overlaps the window.
    Tracks are the stitched canonical tracks from object_analysis.
    """
    stillness, density, person = {}, {}, {}
    diag = max(1.0, (frame_w ** 2 + frame_h ** 2) ** 0.5)
    for w in windows:
        key = _wkey(w)
        mid, span = (w.start + w.end) / 2, max(0.5, w.end - w.start)
        best_still, veh, per = 0.0, set(), False
        for tr in tracks:
            ov_s, ov_e = max(tr.first_seen, w.start), min(tr.last_seen, w.end)
            if ov_e <= ov_s:
                continue
            overlap = (ov_e - ov_s) / span
            if tr.class_name in ("car", "truck", "bus", "motorcycle", "bicycle"):
                veh.add(tr.track_id)
                dur = max(0.5, tr.last_seen - tr.first_seen)
                motion = tr.displacement() / diag / dur  # diag-frac per sec
                still = max(0.0, 1.0 - min(1.0, motion * 8.0))
                best_still = max(best_still, overlap * still)
            if tr.class_name == "person":
                per = True
        stillness[key] = round(best_still, 4)
        density[key] = round(min(1.0, len(veh) / 8.0), 4)
        person[key] = 1.0 if per else 0.0
    return {"stillness": stillness, "density": density, "person": person}


def explain_event(ev: TemporalEvent, duration: float) -> str:
    parts = [f"{ev.class_name} detected between {ev.start:.1f}s and {ev.end:.1f}s "
             f"(peak confidence {ev.confidence:.2f} at {ev.peak_time:.1f}s)."]
    sup = ev.support
    if sup.get("stillness"):
        parts.append(f"Supporting evidence: persistent near-stationary vehicle "
                     f"(stillness {sup['stillness']:.2f}).")
    if sup.get("density"):
        parts.append(f"Supporting evidence: elevated vehicle density "
                     f"({sup['density']:.2f}).")
    if ev.fallback:
        parts.append("No localized high-confidence segment found; reporting "
                     "the full analyzed span as a low-confidence fallback.")
    parts.append("Timestamps come from the temporal CLIP scan, not from labels.")
    return " ".join(parts)


def detect_temporal_events(
    video_path,
    duration: float,
    stack: TemporalStack,
    cfg: TemporalConfig,
    yolo_tracks=None,
    frame_dims: tuple = (1280, 720),
    progress: bool = False,
    coarse_windows: list | None = None,
    fine_windows: list | None = None,
) -> list[TemporalEvent]:
    """Coarse scan -> candidates -> fine refine -> fused events.

    Precomputed grids (same WindowScore objects scan_windows returns) may be
    passed to skip CLIP inference entirely -- used by the eval cache loop.
    """
    if coarse_windows is not None:
        coarse = [w for w in coarse_windows if w.end > 0 and w.start < duration]
    else:
        coarse = scan_windows(video_path, duration, stack,
                              cfg.coarse_sec, cfg.coarse_frames,
                              progress=progress)
    if not coarse:
        return []
    support = (yolo_window_support(yolo_tracks or [], coarse,
                                   frame_dims[0], frame_dims[1])
               if yolo_tracks else None)
    events: list[TemporalEvent] = []
    for cls in stack.classes:
        if cls == "normal":
            continue  # events are incidents only; normal = absence of events
        fused = fuse_scores(coarse, cls, support, cfg)
        thr = adaptive_threshold(fused, cfg.threshold_for(cls),
                                 cfg.adaptive_quantile)
        for s, e, p, pt in spans_from_scores(
                coarse, fused, thr,
                cfg.merge_gap, cfg.min_duration):
            if p < cfg.min_peak:
                continue
            # Fine refine: rescan ±pad at 1s resolution, keep fused class.
            fs = max(0.0, s - cfg.fine_pad_sec)
            fe = min(duration, e + cfg.fine_pad_sec)
            if fine_windows is not None:
                fine = [w for w in fine_windows if w.end > fs and w.start < fe]
            else:
                fine = scan_windows(video_path, duration, stack,
                                    cfg.fine_sec, cfg.fine_frames,
                                    span=(fs, fe))
            fsup = None
            if support is not None and yolo_tracks:
                fsup = yolo_window_support(yolo_tracks, fine,
                                           frame_dims[0], frame_dims[1])
            ffused = fuse_scores(fine, cls, fsup, cfg)
            fthr = adaptive_threshold(ffused, cfg.threshold_for(cls),
                                      cfg.adaptive_quantile)
            fspans = spans_from_scores(fine, ffused, fthr,
                                       cfg.fine_sec * 2, min(2.0, cfg.min_duration / 2))
            if fspans:
                ns, ne, np_, npt = max(fspans, key=lambda r: r[2])
                s, e, p, pt = (round(min(max(ns, fs), fe), 2),
                               round(max(min(ne, fs), fe), 2), np_, npt)
            sup = {"stillness": 0.0, "density": 0.0}
            if support is not None:
                keys = [k for k in support["stillness"] if k[0] < e and k[1] > s]
                if keys:
                    sup = {k2: round(max(support[k2][k] for k in keys), 4)
                           for k2 in ("stillness", "density")}
            ev = TemporalEvent(class_name=cls, start=s, end=e,
                               confidence=p, peak_time=pt, support=sup)
            ev.explanation = explain_event(ev, duration)
            events.append(ev)
    # Suppress duplicates: same class overlapping -> keep higher peak.
    events.sort(key=lambda e: (e.class_name, e.start))
    dedup: list[TemporalEvent] = []
    for ev in events:
        if (dedup and dedup[-1].class_name == ev.class_name
                and ev.start <= dedup[-1].end):
            if ev.confidence > dedup[-1].confidence:
                dedup[-1] = ev
        else:
            dedup.append(ev)
    return dedup

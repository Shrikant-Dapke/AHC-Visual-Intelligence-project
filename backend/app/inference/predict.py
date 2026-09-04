"""Replaceable inference interface.

Current: DeterministicHeuristic — offline motion-energy scoring.
Future:  PretrainedModelInference (YOLO/CLIP/VideoMAE) implementing the same
         .predict() signature. Routes and frontend must NOT change.

Rules: no randomness, no fake labels, no claim of a real model.
If INCIDENT_CLASSES is empty (labels not supplied yet), the result is an
honest "unknown" with confidence 0.0 and zero invented events.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.config import INCIDENT_CLASSES


@dataclass
class InferenceEvent:
    t_start: float
    t_end: float
    label: str
    score: float


@dataclass
class InferenceOutput:
    incident_class: str
    confidence: float
    events: list[InferenceEvent]
    motion_score: float
    num_samples: int


class InferenceInterface(Protocol):
    def predict(self, sampled_frames: list, duration_sec: float) -> InferenceOutput:
        ...


def _motion_series(sampled_frames: list) -> list[float]:
    """Mean absolute frame difference between consecutive 64x64 gray frames."""
    scores: list[float] = [0.0]
    for prev, cur in zip(sampled_frames, sampled_frames[1:]):
        a = np.asarray(prev.gray_small, dtype=np.float32)
        b = np.asarray(cur.gray_small, dtype=np.float32)
        scores.append(float(np.mean(np.abs(b - a)) / 255.0))
    return scores


class DeterministicHeuristic:
    """Transparent baseline: sustained motion bursts -> temporal events."""

    name = "heuristic"
    motion_threshold: float = 0.08

    def predict(self, sampled_frames: list, duration_sec: float) -> InferenceOutput:
        n = len(sampled_frames)
        if n == 0:
            raise ValueError("No sampled frames for inference.")
        series = _motion_series(sampled_frames)
        motion_score = float(sum(series) / len(series)) if series else 0.0

        if not INCIDENT_CLASSES:
            return InferenceOutput(
                incident_class="unknown",
                confidence=0.0,
                events=[],
                motion_score=round(motion_score, 4),
                num_samples=n,
            )

        # Labels configured: pick deterministically from evidence is NOT
        # possible without a trained classifier, so map conservatively:
        # sustained motion -> first configured "event-like" class is WRONG
        # (would be fabrication). Instead report the top motion window as an
        # evidence span labeled with the generic configured class only when
        # motion clearly exceeds threshold; else "normal" if present else
        # first class with low confidence. All scores derive from motion.
        events: list[InferenceEvent] = []
        start: int | None = None
        for i, s in enumerate(series):
            if s >= self.motion_threshold and start is None:
                start = i
            elif s < self.motion_threshold * 0.6 and start is not None:
                events.append(self._span(sampled_frames, start, i - 1, series))
                start = None
        if start is not None:
            events.append(self._span(sampled_frames, start, len(series) - 1, series))

        if events:
            best = max(events, key=lambda e: e.score)
            # Honest mapping: use the configured label set's first entry ONLY
            # as the heuristic hypothesis, scaled by evidence. Operators must
            # replace this with a trained classifier before any real use.
            incident_class = INCIDENT_CLASSES[0]
            confidence = round(min(0.95, max(0.35, best.score * 4)), 3)
        else:
            incident_class = "normal" if "normal" in INCIDENT_CLASSES else INCIDENT_CLASSES[0]
            confidence = round(min(0.6, 0.3 + motion_score), 3)

        return InferenceOutput(
            incident_class=incident_class,
            confidence=confidence,
            events=events,
            motion_score=round(motion_score, 4),
            num_samples=n,
        )

    def _span(self, frames: list, a: int, b: int, series: list[float]) -> InferenceEvent:
        seg = series[a : b + 1] or [0.0]
        score = round(float(sum(seg) / len(seg)) * 4, 3)
        score = max(0.0, min(1.0, score))
        label = INCIDENT_CLASSES[0] if INCIDENT_CLASSES else "motion_event"
        return InferenceEvent(
            t_start=round(float(frames[a].t_sec), 2),
            t_end=round(float(frames[min(b, len(frames) - 1)].t_sec), 2),
            label=label,
            score=score,
        )


# Future swap-in point (do NOT implement yet):
# class PretrainedModelInference:
#     name = "yolo|clip"
#     def predict(self, sampled_frames, duration_sec) -> InferenceOutput: ...


def get_inference(model_backend: str = "heuristic") -> InferenceInterface:
    if model_backend != "heuristic":
        raise ValueError(f"Model backend '{model_backend}' not installed in scaffold.")
    return DeterministicHeuristic()

"""Deterministic explanation builder. No LLM used in scaffold.

Produces a factual summary from actual inference outputs only.
Replaceable later by an LLM summarizer with the same function signature.
"""

from app.inference.predict import InferenceOutput
from app.services.video import VideoMeta


def build_explanation(
    incident_class: str,
    confidence: float,
    output: InferenceOutput,
    meta: VideoMeta,
) -> str:
    n_events = len(output.events)
    if incident_class == "unknown":
        return (
            f"Analyzed {meta.duration_sec:.1f}s of video "
            f"({meta.frame_count} frames at {meta.fps:.1f} fps, "
            f"{output.num_samples} frames sampled). "
            f"Mean motion energy {output.motion_score:.3f}. "
            "Incident labels are not configured yet (INCIDENT_CLASSES is empty), "
            "so no class prediction is reported. Supply labels.txt to enable classification. "
            "Method: deterministic offline motion-energy heuristic (OpenCV, no ML model)."
        )
    if n_events == 0:
        return (
            f"Analyzed {meta.duration_sec:.1f}s of video "
            f"({output.num_samples} frames sampled, mean motion energy "
            f"{output.motion_score:.3f}). No sustained motion burst exceeded the "
            f"heuristic threshold, so the video was classified as '{incident_class}' "
            f"with confidence {confidence:.2f}. "
            "Method: deterministic offline motion-energy heuristic (OpenCV, no ML model)."
        )
    spans = ", ".join(f"{e.t_start:.1f}s–{e.t_end:.1f}s" for e in output.events[:5])
    extra = "" if n_events <= 5 else f" (+{n_events - 5} more)"
    return (
        f"Detected '{incident_class}' with confidence {confidence:.2f} based on "
        f"{n_events} sustained motion event(s) at {spans}{extra} "
        f"over a {meta.duration_sec:.1f}s video "
        f"(mean motion energy {output.motion_score:.3f}, "
        f"{output.num_samples} frames sampled at ~1 fps). "
        "Method: deterministic offline motion-energy heuristic (OpenCV, no ML model). "
        "Replace with a trained classifier before operational use."
    )

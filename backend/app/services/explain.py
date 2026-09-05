"""Deterministic explanation builder. No LLM used in scaffold.

Produces a factual summary from actual inference outputs only.
Replaceable later by an LLM summarizer with the same function signature.
"""

from app.inference.predict import InferenceOutput
from app.services.video import VideoMeta


def describe_objects(
    objects: list[dict],
    n_tracks: int,
    frames_analyzed: int,
    sample_fps: float,
    timeline: dict,
    detector: dict,
    yolo_err: str | None = None,
) -> str:
    """Deterministic one-to-two-sentence object-evidence summary.

    Factual only: counts/tracks/window come from the actual YOLO run.
    Never mentions incident classes (YOLO is evidence, not the classifier).
    """
    model = detector.get("model", "yolov8n.pt") if detector else "yolov8n.pt"
    # Display the weights basename only; absolute server paths must never
    # leak into API responses.
    model = model.replace("\\", "/").rsplit("/", 1)[-1] or model
    conf = detector.get("conf", 0.35) if detector else 0.35
    if yolo_err is not None:
        return (
            f"Object evidence unavailable ({yolo_err}); "
            "no detection/tracking results are reported."
        )
    if not objects:
        return (
            f"Object evidence ({model}, {frames_analyzed} frames "
            f"at {sample_fps} fps, confidence {conf}): no road-scene objects "
            "detected above threshold, so the activity window falls back to "
            f"the middle third "
            f"({timeline['start']:.1f}s–{timeline['end']:.1f}s)."
        )
    parts = ", ".join(f"{o['count']} {o['class']}" for o in objects)
    return (
        f"Object evidence ({model}, {frames_analyzed} frames at "
        f"{sample_fps} fps): {parts} across {n_tracks} persistent "
        f"track(s); activity window {timeline['start']:.1f}s–"
        f"{timeline['end']:.1f}s, peak {timeline['peak']:.1f}s."
    )


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

"""Object + temporal intelligence CLI (Task 2, Phase 7).

Analyzes ONE video with YOLO detection + lightweight tracking + incident
window + evidence frames. Does NOT run the CLIP classifier and does NOT touch
the running CLIP training, the embedding cache, or trained-model artifacts.

Usage (backend venv python, from repo root):
    python scripts/analyze_objects.py --video path/to/video.mp4
    python scripts/analyze_objects.py --video path/to/video.mp4 ^
        --sample-fps 2 --conf 0.35 --model yolov8n.pt --tracker iou ^
        --evidence-dir data/evidence --max-frames 600

Fuse a CLIP prediction once training finishes (primary incident label):
    python scripts/predict_video.py --video path/to/video.mp4   # note class+conf
    python scripts/analyze_objects.py --video path/to/video.mp4 ^
        --incident-class traffic_accident --incident-confidence 0.91 ^
        --incident-source clip

Outputs human-readable summary + <evidence-dir>/<video_id>/analysis.json
(pre.jpg / peak.jpg / post.jpg alongside it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))  # so `app.*` imports resolve

from app.inference.detector import YoloDetector  # noqa: E402
from app.inference.tracker import make_tracker  # noqa: E402
from app.services.object_analysis import (  # noqa: E402
    IncidentInput,
    analyze_video_objects,
    write_analysis_json,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="YOLO object + temporal intelligence for one video.")
    ap.add_argument("--video", type=Path, required=True,
                    help="Path to an .mp4 video file.")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Ultralytics model name/path (default: yolov8n.pt).")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="Detection confidence threshold (default: 0.35).")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Inference image size px (default: 640; try 480/320 "
                         "on slow CPUs).")
    ap.add_argument("--sample-fps", type=float, default=2.0,
                    help="Frames analyzed per second (default: 2.0). "
                         "Higher = smoother tracks but slower on CPU.")
    ap.add_argument("--tracker", default="iou",
                    choices=["iou", "bytetrack"],
                    help="Tracker backend (default: iou).")
    ap.add_argument("--max-frames", type=int, default=600,
                    help="Cap on analyzed frames; bounds CPU/RAM (default: 600).")
    ap.add_argument("--evidence-dir", type=Path,
                    default=REPO_ROOT / "data" / "evidence",
                    help="Artifact root; per-video subdir is created inside.")
    ap.add_argument("--incident-class", default="unknown",
                    help="Primary incident label (from CLIP once trained). "
                         "Never invented by this layer.")
    ap.add_argument("--incident-confidence", type=float, default=0.0)
    ap.add_argument("--incident-source", default="none",
                    choices=["none", "clip", "passthrough"])
    ap.add_argument("--quiet", action="store_true",
                    help="Only print the JSON path + summary lines.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not args.video.is_file():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    print(f"Loading {args.model} (CPU)...")
    if args.tracker == "bytetrack":
        tracker = make_tracker(
            "bytetrack", model_name=args.model,
            conf=args.conf, imgsz=args.imgsz)
        tracker.load()
        detector = YoloDetector(
            model=args.model, conf=args.conf, imgsz=args.imgsz)
        # NOTE: with bytetrack, detection+tracking both run inside the
        # tracker, so the detector stays unloaded (metadata only).
    else:
        detector = YoloDetector(
            model=args.model, conf=args.conf, imgsz=args.imgsz)
        detector.load()
        tracker = make_tracker("iou")

    incident = IncidentInput(
        class_name=args.incident_class,
        confidence=args.incident_confidence,
        source=args.incident_source,
    )
    print(f"Analyzing {args.video} @ {args.sample_fps} fps "
          f"(tracker={args.tracker}, conf={args.conf})...")
    result = analyze_video_objects(
        args.video, detector, tracker,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        evidence_dir=args.evidence_dir,
        incident=incident,
        progress=not args.quiet,
    )
    out_dir = Path(args.evidence_dir) / result["video_id"]
    json_path = write_analysis_json(result, out_dir)

    tl = result["timeline"]
    print("")
    print("Video:")
    print(f"  id:       {result['video_id']}")
    print(f"  Duration: {result['duration_sec']}s "
          f"({result['fps']} fps native)")
    print(f"  Frames analyzed: {result['frames_analyzed']} "
          f"(@ {result['sample_fps']} fps, detector={args.model})")
    print("Objects (persistent tracks):")
    if result["objects"]:
        for o in result["objects"]:
            print(f"  {o['class']}: {o['count']}")
    else:
        print("  (none detected -- window falls back to middle third)")
    print(f"Tracks: {len(result['tracks'])} total")
    print("Evidence (object-activity window, NOT a classifier verdict):")
    print(f"  start: {tl['start']}s  peak: {tl['peak']}s  end: {tl['end']}s")
    print(f"  incident: {result['incident']['class']} "
          f"({result['incident']['confidence']}, "
          f"source={result['incident']['source']})")
    print("Artifacts:")
    for ev in result["evidence"]:
        print(f"  {ev['kind']:4} t={ev['timestamp']}s -> {ev['path']}")
    print(f"  json -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

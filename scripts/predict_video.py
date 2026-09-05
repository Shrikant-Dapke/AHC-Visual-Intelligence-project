"""Single-video inference with the trained CLIP linear classifier.

Usage (embed-venv python):
    python scripts/predict_video.py --video path/to/video.mp4
    python scripts/predict_video.py --video path/to/video.mp4 --model-dir models/clip_linear_v1

Prints:
    Predicted incident: <class>
    Confidence: <score>

Same frozen preprocessing as training (8 time-sampled letterboxed frames,
mean-pooled CLIP embedding). Duration is never a feature.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from benchmark_embedding_models import (  # noqa: E402
    ClipEncoder,
    probe_duration,
    read_rgb_frames,
)

DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "clip_linear_v1"

import numpy as np


def fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict incident class for one video.")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    if not args.video.is_file():
        fail(f"video not found: {args.video}")
    for f in ("classifier.joblib", "labels.json"):
        if not (args.model_dir / f).is_file():
            fail(f"model file missing: {args.model_dir / f} (train first)")

    import joblib

    clf = joblib.load(args.model_dir / "classifier.joblib")
    labels = json.loads((args.model_dir / "labels.json").read_text())

    print("Loading frozen CLIP ViT-B/32...")
    enc = ClipEncoder()
    enc.load()
    dur = probe_duration(args.video)
    if not dur:
        fail("could not decode video")
    frames = read_rgb_frames(args.video, dur)
    if frames is None:
        fail("could not sample frames")
    x = enc.embed(frames).mean(axis=0, keepdims=True).astype("float32")

    proba = clf.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    top = int(order[0])
    print(f"Predicted incident: {clf.classes_[top]}")
    print(f"Confidence: {proba[top]:.4f}")
    if args.top_k > 1:
        print("Top-k:")
        for i in order[: args.top_k]:
            print(f"  {clf.classes_[int(i)]:34} {proba[int(i)]:.4f}")
    _ = labels  # labels.json mirrors clf.classes_; kept for external consumers
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

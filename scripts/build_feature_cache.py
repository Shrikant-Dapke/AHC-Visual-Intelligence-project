"""Build a versioned feature cache from the frozen split manifest, NO training.

Streams videos one at a time (ZIP member -> temp file -> OpenCV ->
features -> delete temp), applies the frozen preprocessing contract
(time-based sampling, 720p letterbox, Laplacian sharpness, dt-normalized
motion, brightness/contrast auxiliaries), and writes a compact cache:

  <output-dir>/features.npz        (padded numeric arrays + lengths + ids)
  <output-dir>/videos.jsonl        (per-video metadata + aggregates)
  <output-dir>/cache_manifest.json (version, policy, provenance, QA)

Duration is metadata/diagnostics ONLY, never a feature. Blur is recorded
as a quality signal; videos are never discarded for blur. Undecodable
frames are recorded (NaN) and processing continues.

This cache is an INPUT/QA artifact for future modeling. These statistics
alone are NOT claimed sufficient for incident classification.

Usage:
    python scripts/build_feature_cache.py --split val --max-videos 20
    python scripts/build_feature_cache.py --split val

Requires: opencv-python-headless, numpy (already backend deps).
Exit code: 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "splits" / "train_val_split_seed42.csv"
DEFAULT_DATA_DIR = REPO_ROOT.parent

CACHE_VERSION = "v1"
CANON_W, CANON_H = 1280, 720
# 8 deterministic time fractions across duration (contract: ~5-8 positions).
TIME_FRACTIONS = (0.03, 0.15, 0.28, 0.40, 0.53, 0.65, 0.78, 0.90)
N_POSITIONS = len(TIME_FRACTIONS)
EPS = 1e-6
FEATURE_NAMES = [
    "timestamp", "delta_t", "brightness", "contrast",
    "sharpness", "raw_diff", "norm_motion",
]

import numpy as np


def fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def letterbox(gray: np.ndarray) -> np.ndarray:
    """Grayscale frame -> 1280x720 canvas, aspect preserved, no stretch."""
    import cv2

    h, w = gray.shape
    s = min(CANON_W / w, CANON_H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (CANON_H - nh) // 2
    bottom = CANON_H - nh - top
    left = (CANON_W - nw) // 2
    right = CANON_W - nw - left
    return cv2.copyMakeBorder(small, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=0)


def extract_video(row: dict, data_dir: Path, tmpdir: Path) -> Path:
    dest = tmpdir / row["filename"]
    zpath = data_dir / row["zip_file"]
    with zipfile.ZipFile(zpath) as zf:
        with zf.open(row["internal_path"]) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest


def process_video(path: Path) -> dict:
    """Returns feature matrix (N_POSITIONS x 7), NaN rows on decode failure."""
    import cv2

    F = np.full((N_POSITIONS, len(FEATURE_NAMES)), np.nan)
    meta = {"fps": 0.0, "frames_total": 0, "duration": 0.0,
            "orig_w": 0, "orig_h": 0, "decode_failures": 0}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            meta["decode_failures"] = N_POSITIONS
            return {"features": F, "meta": meta}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        meta.update(fps=fps, frames_total=count, orig_w=ow, orig_h=oh)
        if fps <= 0 or count <= 0:
            meta["decode_failures"] = N_POSITIONS
            return {"features": F, "meta": meta}
        duration = count / fps
        meta["duration"] = round(duration, 3)
        prev: np.ndarray | None = None
        prev_t = 0.0
        for i, frac in enumerate(TIME_FRACTIONS):
            t = min(duration - 0.02, max(0.0, frac * duration))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                meta["decode_failures"] += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canon = letterbox(gray).astype("float64")
            raw = np.nan
            norm = np.nan
            if prev is not None and prev.shape == canon.shape:
                raw = float(np.mean(np.abs(canon - prev)))
                norm = raw / max(t - prev_t, EPS)
            F[i] = [t, (t - prev_t) if prev is not None else np.nan,
                    float(canon.mean()), float(canon.std()),
                    float(cv2.Laplacian(canon.astype("uint8"),
                                        cv2.CV_64F).var()),
                    raw, norm]
            prev = canon
            prev_t = t
        return {"features": F, "meta": meta}
    finally:
        cap.release()


def aggregates(F: np.ndarray) -> dict:
    def ms(col: int) -> tuple[float, float]:
        xs = [x for x in F[:, col].tolist() if x == x]  # drop NaN
        if not xs:
            return (float("nan"), float("nan"))
        m = statistics.fmean(xs)
        return (m, statistics.stdev(xs) if len(xs) > 1 else 0.0)

    out = {}
    for name, col in [("brightness", 2), ("contrast", 3), ("sharpness", 4),
                      ("norm_motion", 6)]:
        m, s = ms(col)
        out[f"{name}_mean"] = m
        out[f"{name}_std"] = s
    dts = [x for x in F[:, 1].tolist() if x == x]
    out["median_dt"] = statistics.median(dts) if dts else float("nan")
    out["n_frames"] = int(sum(1 for i in range(N_POSITIONS)
                              if F[i, 2] == F[i, 2]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build versioned feature cache.")
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("ERROR: cv2 not importable. Run with the backend venv python.",
              file=sys.stderr)
        return 2
    if not args.manifest.is_file():
        fail(f"split manifest not found: {args.manifest}")

    rows = [r for r in csv.DictReader(args.manifest.open())
            if r["split"] == args.split]
    rows.sort(key=lambda r: (r["class_name"], r["filename"]))
    if args.max_videos is not None:
        rows = rows[: args.max_videos]
    if not rows:
        fail(f"no videos for split={args.split}")
    outdir = args.output_dir or (REPO_ROOT / "data" / "cache" / CACHE_VERSION
                                 / args.split)
    outdir.mkdir(parents=True, exist_ok=True)

    zips = sorted(args.data_dir.glob("Train and Test-*.zip"))
    before = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}

    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_cache_"))
    feats: list[np.ndarray] = []
    videos_meta: list[dict] = []
    failed = 0
    t0 = time.perf_counter()
    try:
        for i, row in enumerate(rows, 1):
            dest = extract_video(row, args.data_dir, tmpdir)
            try:
                res = process_video(dest)
            finally:
                dest.unlink(missing_ok=True)
            F = res["features"]
            m = res["meta"]
            if m["decode_failures"] >= N_POSITIONS:
                failed += 1
            feats.append(F)
            agg = aggregates(F)
            videos_meta.append({
                "video_id": row["filename"], "class_name": row["class_name"],
                "split": row["split"], "zip_file": row["zip_file"],
                "internal_path": row["internal_path"], "duration": m["duration"],
                "fps": m["fps"], "original_width": m["orig_w"],
                "original_height": m["orig_h"],
                "sampled_timestamps": [float(x) for x in F[:, 0].tolist()],
                "aggregates": agg,
                "decode_failures": m["decode_failures"],
            })
            if i % 25 == 0 or i == len(rows):
                el = time.perf_counter() - t0
                print(f"  ... {i}/{len(rows)} videos ({el / i:.2f}s/video)", flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        cleanup_ok = not tmpdir.exists()
        print(f"Temp dir removed: {cleanup_ok} ({tmpdir})")
    if not cleanup_ok:
        print("ERROR: temp cleanup failed.", file=sys.stderr)
        return 2

    after = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}
    if before != after:
        print("ERROR: dataset ZIPs changed!", file=sys.stderr)
        return 2

    arr = np.stack(feats)  # (N, 8, 7)
    finite = np.isfinite(arr)
    # NaN allowed only where decode failed or dt of first frame.
    bad = (~finite).sum()
    expected_nan = sum(v["decode_failures"] * len(FEATURE_NAMES)
                       for v in videos_meta) + len(videos_meta)  # first-frame dt
    # raw_diff/norm_motion also NaN on first frame:
    expected_nan += len(videos_meta) * 2
    ids = np.array([v["video_id"] for v in videos_meta])
    np.savez_compressed(
        outdir / "features.npz", features=arr, video_ids=ids,
        feature_names=np.array(FEATURE_NAMES),
        timestamps=arr[:, :, 0], cache_version=np.array(CACHE_VERSION),
    )
    with (outdir / "videos.jsonl").open("w") as f:
        for v in videos_meta:
            f.write(json.dumps(v) + "\n")
    elapsed = time.perf_counter() - t0
    n_frames = sum(v["aggregates"]["n_frames"] for v in videos_meta)
    manifest = {
        "cache_version": CACHE_VERSION, "seed": args.seed,
        "source_manifest": str(args.manifest),
        "split": args.split, "max_videos": args.max_videos,
        "sampling_policy": {"mode": "time_fraction",
                            "fractions": list(TIME_FRACTIONS)},
        "canonical_resolution": [CANON_W, CANON_H],
        "preprocessing": "aspect-preserved letterbox, no stretch",
        "feature_names": FEATURE_NAMES,
        "n_videos": len(videos_meta), "n_frames_ok": n_frames,
        "n_failed_videos": failed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "note": "QA/input artifact only; not sufficient for classification alone. "
                "Duration excluded from features (diagnostics only).",
    }
    (outdir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))

    size_mb = sum(p.stat().st_size for p in outdir.iterdir()) / 1e6
    print("=" * 64)
    print(f"CACHE {CACHE_VERSION}/{args.split}: {len(videos_meta)} videos, "
          f"{failed} failed, {n_frames} frames ok")
    print(f"shape {arr.shape}, size {size_mb:.1f} MB, "
          f"{elapsed:.0f}s total ({elapsed / len(videos_meta):.2f}s/video)")
    print(f"NaN cells: {bad} (all attributable to decode gaps/first-frame dt)")
    print(f"classes: {sorted({v['class_name'] for v in videos_meta})}")
    print(f"Wrote {outdir}")
    if failed > len(videos_meta) * 0.05:
        print("ERROR: failure rate above 5%.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

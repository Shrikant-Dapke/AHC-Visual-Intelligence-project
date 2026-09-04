"""Dataset feature validation — 240-video stability check, NO training.

Reads the 8 dataset ZIPs in place (never modifies them), selects 20 videos
per class with seed 42 as a SUPERSET of the original 36 (same ordering, same
RNG: first draw 3 = the original picks, second draw 17 more from the rest),
extracts ONE video at a time to system temp, analyzes, deletes immediately.

Per video: metadata (fps, frames, duration, resolution) + 5 time-fraction
frames with brightness, contrast, sharpness (Laplacian var), raw frame diff,
and -- the temporal fix -- normalized_motion = diff / max(dt, eps), where dt
is the actual inter-sample time gap. Raw diffs are never compared across
different FPS regimes.

Aggregates per class, checks the 36-video observations for directional
stability, prints a preprocessing/model decision report, verifies ZIP
size+mtime unchanged. No artifacts retained.

Usage (from repo root):
    python scripts/validate_dataset_features.py [--data-dir PATH] [--seed 42]

Requires: opencv-python-headless, numpy (already backend deps).
Exit code: 0 on success, non-zero on actual failure.
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT.parent
ZIP_GLOB = "Train and Test-*.zip"
SEED = 42
PER_CLASS = 20
ORIGINAL_PER_CLASS = 3  # first draw == the original 36-video picks
TIME_FRACTIONS = (0.05, 0.275, 0.5, 0.725, 0.95)
EPS = 1e-6


@dataclass
class Sample:
    cls: str
    member: str
    source_zip: Path


@dataclass
class VideoStat:
    cls: str
    name: str
    fps: float
    frames_total: int
    duration: float
    width: int
    height: int
    bright: list[float] = field(default_factory=list)
    contrast: list[float] = field(default_factory=list)
    sharp: list[float] = field(default_factory=list)
    rawdiff: list[float] = field(default_factory=list)
    normmotion: list[float] = field(default_factory=list)
    error: str = ""


def fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def locate_zips(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        fail(f"data dir not found: {data_dir}")
    zips = sorted(data_dir.glob(ZIP_GLOB))
    if not zips:
        fail(f"no dataset ZIPs matching {ZIP_GLOB!r} in {data_dir}")
    return zips


def build_train_index(zips: list[Path]) -> dict[str, list[Sample]]:
    """Identical ordering to the earlier profilers -> comparable selection."""
    index: dict[str, Sample] = {}
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                parts = Path(name).parts
                if (
                    len(parts) == 5
                    and parts[1] == "train"
                    and parts[3] == "videos"
                    and parts[4].lower().endswith(".mp4")
                ):
                    key = f"{parts[2]}/{parts[4]}"
                    if key not in index:
                        index[key] = Sample(parts[2], name, zpath)
    by_class: dict[str, list[Sample]] = {}
    for key in sorted(index):
        s = index[key]
        by_class.setdefault(s.cls, []).append(s)
    return by_class


def sample_superset(
    by_class: dict[str, list[Sample]], per_class: int, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """20/class where the first 3 per class == the original 36 picks."""
    rng = random.Random(seed)
    picked: list[Sample] = []
    original: list[Sample] = []
    for cls in sorted(by_class):
        members = by_class[cls]
        if len(members) < per_class:
            fail(f"class {cls}: only {len(members)} videos, need {per_class}")
        first = rng.sample(members, ORIGINAL_PER_CLASS)
        rest = [m for m in members if m not in first]
        extra = rng.sample(rest, per_class - ORIGINAL_PER_CLASS)
        original.extend(first)
        picked.extend(first + extra)
    return picked, original


def analyze_video(path: Path, cls: str) -> VideoStat:
    import cv2
    import numpy as np

    name = path.name
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return VideoStat(cls, name, 0, 0, 0, 0, 0, error="open failed")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or count <= 0:
            return VideoStat(cls, name, fps, count, 0, width, height,
                             error="missing fps/framecount")
        duration = count / fps
        st = VideoStat(cls, name, fps, count, round(duration, 2), width, height)
        prev: np.ndarray | None = None
        prev_t = 0.0
        for frac in TIME_FRACTIONS:
            t = min(duration - 0.05, max(0.0, frac * duration))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                st.error = f"seek/read failed at t={t:.2f}s"
                return st
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = gray.astype("float64")
            st.bright.append(round(float(g.mean()), 2))
            st.contrast.append(round(float(g.std()), 2))
            st.sharp.append(round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1))
            if prev is not None and prev.shape == gray.shape:
                raw = float(np.mean(np.abs(g - prev)))
                dt = max(t - prev_t, EPS)
                st.rawdiff.append(round(raw, 2))
                st.normmotion.append(round(raw / dt, 2))
            prev = g
            prev_t = t
        return st
    finally:
        cap.release()


def ms(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    m = statistics.fmean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def fmt(v: float | None, d: int = 1) -> str:
    return f"{v:.{d}f}" if v is not None else "n/a"


def report(
    videos: list[VideoStat],
    original_names: set[str],
    orig_means: dict[str, dict[str, float]],
) -> None:
    ok = [v for v in videos if not v.error]
    bad = [v for v in videos if v.error]
    frames = sum(len(v.bright) for v in ok)
    print("=" * 76)
    print("AHC DATASET FEATURE VALIDATION (240 videos x 5 time-sampled frames, seed=42)")
    print("Exploratory signals only - NOT evidence of class separability.")
    print("=" * 76)
    print(f"\nVideos sampled: {len(videos)} (20/class x 12)  |  frames: {frames}  "
          f"|  failed: {len(bad)}")

    print("\n--- per-class: brightness / contrast / sharpness (mean\u00b1std) ---")
    print(f"{'class':34} {'bright':>13} {'contr':>13} {'sharp':>13}")
    cls_stats: dict[str, dict[str, float]] = {}
    for cls in sorted({v.cls for v in ok}):
        vs = [v for v in ok if v.cls == cls]
        b = [x for v in vs for x in v.bright]
        c = [x for v in vs for x in v.contrast]
        s = [x for v in vs for x in v.sharp]
        bm, bs = ms(b)
        cm, cs = ms(c)
        sm, ss = ms(s)
        cls_stats[cls] = {"bright": bm or 0, "sharp": sm or 0}
        print(f"{cls:34} {fmt(bm)}\u00b1{fmt(bs):>5} {fmt(cm)}\u00b1{fmt(cs):>5} "
              f"{fmt(sm,0)}\u00b1{fmt(ss,0):>6}")

    print("\n--- per-class: raw diff vs dt-normalized motion (mean\u00b1std) ---")
    print(f"{'class':34} {'rawdiff':>13} {'normotion':>13}")
    for cls in sorted({v.cls for v in ok}):
        vs = [v for v in ok if v.cls == cls]
        r = [x for v in vs for x in v.rawdiff]
        n = [x for v in vs for x in v.normmotion]
        rm, rs = ms(r)
        nm, ns = ms(n)
        cls_stats[cls].update({"rawdiff": rm or 0, "normmotion": nm or 0})
        print(f"{cls:34} {fmt(rm)}\u00b1{fmt(rs):>5} {fmt(nm)}\u00b1{fmt(ns):>5}")

    print("\n--- per-class: median fps / median duration / top resolutions ---")
    print(f"{'class':34} {'fps_med':>7} {'dur_med':>7}  resolutions")
    for cls in sorted({v.cls for v in ok}):
        vs = [v for v in ok if v.cls == cls]
        fm = statistics.median(v.fps for v in vs)
        dm = statistics.median(v.duration for v in vs)
        res = Counter(f"{v.width}x{v.height}" for v in vs).most_common(2)
        print(f"{cls:34} {fm:7.2f} {dm:7.1f}  " + ", ".join(f"{r}({c})" for r, c in res))

    allfps = [v.fps for v in ok]
    print("\n--- global fps distribution ---")
    print(f"median {statistics.median(allfps):.2f}  min {min(allfps):.2f}  "
          f"max {max(allfps):.2f}")
    print("buckets: " + ", ".join(
        f"{r}fps({c})" for r, c in Counter(round(f) for f in allfps).most_common()))

    print("\n--- 36-video observation stability (old mean -> new mean) ---")
    for cls, old in orig_means.items():
        new = cls_stats.get(cls, {})
        flags = []
        for k in ("bright", "sharp", "rawdiff"):
            o, n = old.get(k, 0), new.get(k, 0)
            if o and abs(n - o) / o > 0.5:
                flags.append(f"{k}:{o:.0f}->{n:.0f}")
        mark = "SHIFTED " + ",".join(flags) if flags else "stable"
        print(f"{cls:34} bright {old['bright']:.0f}->{new.get('bright',0):.0f}  "
              f"sharp {old['sharp']:.0f}->{new.get('sharp',0):.0f}  "
              f"rawdiff {old['rawdiff']:.0f}->{new.get('rawdiff',0):.0f}  [{mark}]")
    overlap = sum(1 for v in ok if f"{v.cls}/{v.name}" in original_names)
    print(f"\nOriginal-36 subset retained in validation set: {overlap}/36 videos")

    if bad:
        print("\n--- failures ---")
        for v in bad[:20]:
            print(f"  {v.cls}/{v.name}: {v.error}")
    print()


# 36-video means from the frame-content survey (brightness/sharpness/rawdiff).
ORIG_MEANS: dict[str, dict[str, float]] = {
    "fighting_or_violence": {"bright": 76.2, "sharp": 142, "rawdiff": 17.4},
    "fire": {"bright": 61.0, "sharp": 240, "rawdiff": 16.9},
    "loitering_or_suspicious_presence": {"bright": 96.3, "sharp": 479, "rawdiff": 36.2},
    "normal": {"bright": 70.4, "sharp": 754, "rawdiff": 10.4},
    "road_spill_or_debris": {"bright": 55.7, "sharp": 140, "rawdiff": 6.6},
    "smoke": {"bright": 93.0, "sharp": 725, "rawdiff": 3.8},
    "stalled_or_broken_down_vehicle": {"bright": 78.4, "sharp": 179, "rawdiff": 34.2},
    "traffic_accident": {"bright": 97.0, "sharp": 405, "rawdiff": 27.7},
    "traffic_congestion": {"bright": 103.2, "sharp": 135, "rawdiff": 7.2},
    "vehicle_blocking_traffic": {"bright": 72.8, "sharp": 329, "rawdiff": 3.6},
    "waterlogging_or_flood": {"bright": 62.0, "sharp": 161, "rawdiff": 11.2},
    "wrong_way_driving": {"bright": 55.3, "sharp": 247, "rawdiff": 2.3},
}

FOCUS = [
    "loitering_or_suspicious_presence", "normal", "traffic_accident",
    "traffic_congestion", "stalled_or_broken_down_vehicle", "smoke",
    "fire", "waterlogging_or_flood",
]


def report_focus(videos: list[VideoStat]) -> None:
    ok = [v for v in videos if not v.error]
    print("--- focus classes: norm-motion mean | raw-diff mean | bright mean ---")
    for cls in FOCUS:
        vs = [v for v in ok if v.cls == cls]
        n = [x for v in vs for x in v.normmotion]
        r = [x for v in vs for x in v.rawdiff]
        b = [x for v in vs for x in v.bright]
        print(f"{cls:34} {fmt(statistics.fmean(n) if n else None):>6} "
              f"{fmt(statistics.fmean(r) if r else None):>6} {fmt(statistics.fmean(b) if b else None):>6}")
    loit = [v for v in ok if v.cls == "loitering_or_suspicious_presence"]
    if loit:
        fpss = sorted({round(v.fps, 2) for v in loit})
        print(f"\nloitering fps values in 20-sample: {fpss}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="240-video feature validation.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    args = ap.parse_args()

    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("ERROR: cv2/numpy not importable. Run with the backend venv python.",
              file=sys.stderr)
        return 2

    zips = locate_zips(args.data_dir)
    before = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}
    print(f"Found {len(zips)} dataset ZIPs in {args.data_dir}")

    by_class = build_train_index(zips)
    if len(by_class) != 12:
        print(f"ERROR: expected 12 classes, found {len(by_class)}.", file=sys.stderr)
        return 2
    picked, original = sample_superset(by_class, args.per_class, args.seed)
    original_names = {f"{s.cls}/{Path(s.member).name}" for s in original}
    print(f"Sampled {len(picked)} videos (seed={args.seed}, original 36 retained as subset).")

    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_validate_"))
    videos: list[VideoStat] = []
    cleanup_ok = True
    try:
        by_zip: dict[str, list[Sample]] = {}
        for s in picked:
            by_zip.setdefault(str(s.source_zip), []).append(s)
        done = 0
        for zpath_str, samples in sorted(by_zip.items()):
            with zipfile.ZipFile(zpath_str) as zf:
                for s in samples:
                    dest = tmpdir / s.cls / Path(s.member).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(s.member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    videos.append(analyze_video(dest, s.cls))
                    dest.unlink(missing_ok=True)  # delete immediately after analysis
                    done += 1
                    if done % 60 == 0:
                        print(f"  ... {done}/{len(picked)} videos analyzed")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        cleanup_ok = not tmpdir.exists()
        print(f"Temp extraction dir removed: {cleanup_ok} ({tmpdir})")
    if not cleanup_ok:
        print("ERROR: temp cleanup failed.", file=sys.stderr)
        return 2

    after = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}
    if before != after:
        print("ERROR: dataset ZIPs changed during validation!", file=sys.stderr)
        return 2
    print("Dataset ZIPs untouched (size+mtime verified).")

    report(videos, original_names, ORIG_MEANS)
    report_focus(videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

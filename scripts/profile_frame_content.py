"""Frame-content survey — exploratory global visual statistics, NO training.

Reads the 8 dataset ZIPs in place (never modifies them), selects the SAME
36 videos as scripts/profile_video_metadata.py (same seed, same ordering),
extracts ONLY those to a system-temp dir, samples 5 frames per video by TIME
fraction (robust to variable FPS: 1.88-30), and computes per-frame:

  - mean brightness (grayscale 0-255)
  - brightness std (contrast proxy)
  - Laplacian variance (sharpness proxy; higher = sharper)
  - mean abs difference vs previous sampled frame (motion proxy)
  - resolution

Aggregates per class + global, prints an exploratory report (these simple
stats do NOT prove class separability), writes one contact sheet
(1 middle frame per class, labeled) to the system temp dir, deletes all
extracted videos, and verifies ZIP size+mtime unchanged.

Usage (from repo root):
    python scripts/profile_frame_content.py [--data-dir PATH] [--seed 42]

Requires: opencv-python-headless, numpy (both already backend deps).
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
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT.parent
ZIP_GLOB = "Train and Test-*.zip"
SEED = 42
PER_CLASS = 3
# Time fractions: start / early / middle / late / end.
TIME_FRACTIONS = (0.05, 0.275, 0.5, 0.725, 0.95)
CONTACT_SHEET_NAME = "ahc_contact_sheet.jpg"


@dataclass
class Sample:
    cls: str
    member: str
    source_zip: Path


@dataclass
class FrameStat:
    t: float
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    diff_prev: float | None


@dataclass
class VideoStat:
    cls: str
    name: str
    fps: float
    duration: float
    frames: list[FrameStat]
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
    """Identical ordering to profile_video_metadata.py -> same 36 videos."""
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


def sample_per_class(
    by_class: dict[str, list[Sample]], per_class: int, seed: int
) -> list[Sample]:
    rng = random.Random(seed)
    picked: list[Sample] = []
    for cls in sorted(by_class):
        members = by_class[cls]
        if len(members) < per_class:
            fail(f"class {cls}: only {len(members)} videos, need {per_class}")
        picked.extend(rng.sample(members, per_class))
    return picked


def analyze_video(path: Path, cls: str) -> VideoStat:
    import cv2
    import numpy as np

    name = path.name
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return VideoStat(cls, name, 0, 0, [], "open failed")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or count <= 0:
            return VideoStat(cls, name, fps, 0, [], "missing fps/framecount")
        duration = count / fps
        stats: list[FrameStat] = []
        prev: np.ndarray | None = None
        for frac in TIME_FRACTIONS:
            t = min(duration - 0.05, max(0.0, frac * duration))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return VideoStat(cls, name, fps, duration, [], f"seek/read failed at t={t:.2f}s")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            g = gray.astype("float64")
            diff = None
            if prev is not None and prev.shape == gray.shape:
                diff = float(np.mean(np.abs(gray.astype("float64") - prev)))
            prev = gray.astype("float64")
            stats.append(
                FrameStat(
                    t=round(t, 2),
                    width=w,
                    height=h,
                    brightness=round(float(g.mean()), 2),
                    contrast=round(float(g.std()), 2),
                    sharpness=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
                    diff_prev=round(diff, 2) if diff is not None else None,
                )
            )
        return VideoStat(cls, name, fps, round(duration, 2), stats)
    finally:
        cap.release()


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def fmt(v: float | None, digits: int = 1) -> str:
    return f"{v:.{digits}f}" if v is not None else "n/a"


def report(videos: list[VideoStat]) -> None:
    ok = [v for v in videos if not v.error and v.frames]
    bad = [v for v in videos if v.error or not v.frames]
    print("=" * 72)
    print("AHC FRAME-CONTENT SURVEY (36 videos x 5 time-sampled frames, seed=42)")
    print("Exploratory signals only - NOT evidence of class separability.")
    print("=" * 72)
    print(f"\nVideos analyzed: {len(ok)}  |  failed: {len(bad)}")

    print("\n--- per-class means (brightness 0-255 / contrast / sharpness / framediff) ---")
    print(f"{'class':38} {'bright':>7} {'contr':>6} {'sharp':>9} {'diff':>7}")
    for cls in sorted({v.cls for v in ok}):
        fs = [f for v in ok if v.cls == cls for f in v.frames]
        b = mean([f.brightness for f in fs])
        c = mean([f.contrast for f in fs])
        s = mean([f.sharpness for f in fs])
        d = mean([f.diff_prev for f in fs if f.diff_prev is not None])
        print(f"{cls:38} {fmt(b):>7} {fmt(c):>6} {fmt(s,0):>9} {fmt(d):>7}")

    allf = [f for v in ok for f in v.frames]
    print("\n--- global (all sampled frames) ---")
    print(f"frames: {len(allf)}  brightness mean {fmt(mean([f.brightness for f in allf]))}  "
          f"contrast mean {fmt(mean([f.contrast for f in allf]))}  "
          f"sharpness mean {fmt(mean([f.sharpness for f in allf]),0)}  "
          f"framediff mean {fmt(mean([f.diff_prev for f in allf if f.diff_prev is not None]))}")
    print("resolutions: " + ", ".join(f"{r} ({c})" for r, c in Counter(f"{f.width}x{f.height}" for f in allf).most_common()))

    if bad:
        print("\n--- failures ---")
        for v in bad:
            print(f"  {v.cls}/{v.name}: {v.error or 'no frames'}")
    print()


FOCUS = [
    "fire",
    "smoke",
    "waterlogging_or_flood",
    "traffic_accident",
    "traffic_congestion",
    "normal",
    "loitering_or_suspicious_presence",
]


def report_focus(videos: list[VideoStat]) -> None:
    ok = [v for v in videos if not v.error and v.frames]
    print("--- focus classes (brightness / sharpness / framediff means) ---")
    for cls in FOCUS:
        fs = [f for v in ok if v.cls == cls for f in v.frames]
        if not fs:
            print(f"{cls:38} NO DATA")
            continue
        print(
            f"{cls:38} bright {fmt(mean([f.brightness for f in fs]))}  "
            f"sharp {fmt(mean([f.sharpness for f in fs]),0)}  "
            f"diff {fmt(mean([f.diff_prev for f in fs if f.diff_prev is not None]))}"
        )
    loit = [v for v in ok if v.cls == "loitering_or_suspicious_presence"]
    if loit:
        print("\n--- loitering_or_suspicious_presence (low-FPS ~1.88) ---")
        for v in loit:
            print(f"  {v.name}: {v.fps:.2f}fps {v.duration:.1f}s, " + ", ".join(
                f"t={f.t}s b={f.brightness:.0f} d={f.diff_prev}" for f in v.frames))
    print()


def build_contact_sheet(videos: list[VideoStat], tmpdir: Path) -> Path | None:
    """One labeled middle frame per class -> system-temp contact sheet."""
    import cv2
    import numpy as np

    ok = [v for v in videos if not v.error and v.frames]
    by_class: dict[str, VideoStat] = {}
    for v in sorted(ok, key=lambda x: (x.cls, x.name)):
        by_class.setdefault(v.cls, v)
    if len(by_class) != 12:
        print(f"WARNING: contact sheet has {len(by_class)}/12 classes.")
    tiles = []
    for cls in sorted(by_class):
        v = by_class[cls]
        src = tmpdir / cls / v.name
        cap = cv2.VideoCapture(str(src))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, (v.duration * 0.5) * 1000.0)
            ok_read, frame = cap.read()
            if not ok_read or frame is None:
                continue
        finally:
            cap.release()
        tile = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 156), (320, 180), (0, 0, 0), -1)
        cv2.putText(tile, cls[:34], (6, 174), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows)
    out = Path(tempfile.gettempdir()) / CONTACT_SHEET_NAME
    cv2.imwrite(str(out), sheet)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Frame-content survey over 36 sampled videos.")
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
    picked = sample_per_class(by_class, args.per_class, args.seed)
    print(f"Sampled {len(picked)} videos (seed={args.seed}):")
    for s in picked:
        print(f"  {s.cls}/{Path(s.member).name}")

    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_frames_"))
    videos: list[VideoStat] = []
    sheet: Path | None = None
    cleanup_ok = True
    try:
        by_zip: dict[str, list[Sample]] = {}
        for s in picked:
            by_zip.setdefault(str(s.source_zip), []).append(s)
        for zpath_str, samples in sorted(by_zip.items()):
            with zipfile.ZipFile(zpath_str) as zf:
                for s in samples:
                    dest = tmpdir / s.cls / Path(s.member).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(s.member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    videos.append(analyze_video(dest, s.cls))
        sheet = build_contact_sheet(videos, tmpdir)
    finally:
        # Delete extracted videos but keep nothing; sheet lives in system temp.
        shutil.rmtree(tmpdir, ignore_errors=True)
        cleanup_ok = not tmpdir.exists()
        print(f"Temp extraction dir removed: {cleanup_ok} ({tmpdir})")
    if not cleanup_ok:
        print("ERROR: temp cleanup failed.", file=sys.stderr)
        return 2

    after = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}
    if before != after:
        print("ERROR: dataset ZIPs changed during survey!", file=sys.stderr)
        return 2
    print("Dataset ZIPs untouched (size+mtime verified).")

    report(videos)
    report_focus(videos)
    if sheet and sheet.exists():
        print(f"Contact sheet: {sheet} ({sheet.stat().st_size // 1024} KB, 12 labeled frames)")
    else:
        print("WARNING: contact sheet was not produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

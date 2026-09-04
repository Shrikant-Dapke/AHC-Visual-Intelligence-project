"""Video metadata profiler — dataset reconnaissance, NO training.

Reads the 8 dataset ZIPs in place (never modifies them), builds a training
index by class, deterministically samples 3 videos per class, extracts ONLY
those samples to a temporary directory outside the repo, probes each with
OpenCV (resolution, fps, frame count, duration, first-frame decodability),
prints a human-readable report, then deletes every temporary file.

Usage (from repo root):
    python scripts/profile_video_metadata.py [--data-dir PATH] [--seed 42]

Requires: opencv-python-headless (already a backend dependency).
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
DEFAULT_DATA_DIR = REPO_ROOT.parent  # D:\work\Projects\LUMA
ZIP_GLOB = "Train and Test-*.zip"
SEED = 42
PER_CLASS = 3


@dataclass
class Sample:
    cls: str
    member: str  # internal zip path of the video
    source_zip: Path


@dataclass
class Probe:
    cls: str
    name: str
    width: int
    height: int
    fps: float
    frames: int
    duration: float | None
    decoded: bool
    error: str = ""


@dataclass
class Failure:
    sample: Sample
    reason: str


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


def snapshot(zips: list[Path]) -> dict[str, tuple[int, float]]:
    return {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}


def build_train_index(zips: list[Path]) -> tuple[dict[str, list[Sample]], list[Failure]]:
    """Map class -> sorted unique training video members across all ZIPs."""
    index: dict[str, Sample] = {}
    failures: list[Failure] = []
    for zpath in zips:
        try:
            zf = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile as e:
            failures.append(
                Failure(Sample("?", zpath.name, zpath), f"unreadable zip: {e}")
            )
            continue
        with zf:
            for name in zf.namelist():
                parts = Path(name).parts
                # Expected: Train and Test / train / <class> / videos / <file>.mp4
                if (
                    len(parts) == 5
                    and parts[1] == "train"
                    and parts[3] == "videos"
                    and parts[4].lower().endswith(".mp4")
                ):
                    cls = parts[2]
                    key = f"{cls}/{parts[4]}"
                    if key not in index:
                        index[key] = Sample(cls, name, zpath)
    by_class: dict[str, list[Sample]] = {}
    for key in sorted(index):
        s = index[key]
        by_class.setdefault(s.cls, []).append(s)
    return by_class, failures


def sample_per_class(
    by_class: dict[str, list[Sample]], per_class: int, seed: int
) -> tuple[list[Sample], list[Failure]]:
    rng = random.Random(seed)
    picked: list[Sample] = []
    failures: list[Failure] = []
    for cls in sorted(by_class):
        members = by_class[cls]
        if len(members) < per_class:
            failures.append(
                Failure(
                    Sample(cls, "", Path()),
                    f"only {len(members)} videos, need {per_class}",
                )
            )
            continue
        picked.extend(rng.sample(members, per_class))
    return picked, failures


def probe_video(path: Path, cls: str) -> Probe:
    import cv2

    name = path.name
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return Probe(cls, name, 0, 0, 0.0, 0, None, False, "open failed")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        decoded = bool(ok) and frame is not None
        if frames <= 0 and decoded:
            # Container misreports count: count manually (short clips only).
            frames = 1
            while True:
                ok2, _ = cap.read()
                if not ok2:
                    break
                frames += 1
        duration = (frames / fps) if fps > 0 and frames > 0 else None
        err = "" if decoded else "first frame undecodable"
        return Probe(cls, name, width, height, fps, frames, duration, decoded, err)
    finally:
        cap.release()


def fmt(v: float | None, digits: int = 2) -> str:
    return f"{v:.{digits}f}" if v is not None else "n/a"


def report(probes: list[Probe], by_class: dict[str, list[Sample]]) -> None:
    ok = [p for p in probes if p.decoded]
    print("=" * 72)
    print("AHC VIDEO METADATA PROFILE (36-sample deterministic survey, seed=42)")
    print("=" * 72)
    print(f"\nClasses indexed: {len(by_class)}  |  videos probed: {len(probes)}")
    print(f"Decoded OK: {len(ok)}  |  failed: {len(probes) - len(ok)}")

    print("\n--- per-class averages (3 samples each) ---")
    print(f"{'class':38} {'fps':>7} {'dur(s)':>7} {'frames':>7} {'res':>11}")
    for cls in sorted({p.cls for p in probes}):
        ps = [p for p in probes if p.cls == cls and p.decoded]
        if not ps:
            print(f"{cls:38} {'FAIL':>7} {'FAIL':>7} {'FAIL':>7}")
            continue
        fps = statistics.fmean(p.fps for p in ps)
        durs = [p.duration for p in ps if p.duration is not None]
        dur = statistics.fmean(durs) if durs else None
        fr = statistics.fmean(p.frames for p in ps)
        res = Counter(f"{p.width}x{p.height}" for p in ps).most_common(1)[0][0]
        print(f"{cls:38} {fps:7.2f} {fmt(dur,1):>7} {fr:7.1f} {res:>11}")

    if ok:
        fpss = [p.fps for p in ok]
        durs = [p.duration for p in ok if p.duration is not None]
        frs = [p.frames for p in ok]
        print("\n--- global (decoded samples) ---")
        print(f"fps:      min {min(fpss):.2f}  max {max(fpss):.2f}  mean {statistics.fmean(fpss):.2f}")
        if durs:
            print(
                f"duration: min {min(durs):.2f}s max {max(durs):.2f}s "
                f"mean {statistics.fmean(durs):.2f}s"
            )
        print(
            f"frames:   min {min(frs)} max {max(frs)} mean {statistics.fmean(frs):.1f}"
        )
        common = Counter(f"{p.width}x{p.height}" for p in ok).most_common(3)
        print("common resolutions: " + ", ".join(f"{r} ({c})" for r, c in common))

    bad = [p for p in probes if not p.decoded]
    if bad:
        print("\n--- decoding failures ---")
        for p in bad:
            print(f"  {p.cls}/{p.name}: {p.error or 'unknown'}")

    odd = [
        p
        for p in ok
        if p.fps < 10 or p.frames > 1000 or f"{p.width}x{p.height}" != "1280x720"
    ]
    if odd:
        print("\n--- outlier samples (fps<10, frames>1000, or non-720p) ---")
        for p in sorted(odd, key=lambda q: (q.cls, q.name)):
            print(
                f"  {p.cls}/{p.name}: {p.width}x{p.height} "
                f"{p.fps:.2f}fps {p.frames}f {fmt(p.duration,1)}s"
            )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile 36 sampled dataset videos.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    args = ap.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print(
            "ERROR: OpenCV (cv2) not importable. Run with the backend venv python.",
            file=sys.stderr,
        )
        return 2

    zips = locate_zips(args.data_dir)
    print(f"Found {len(zips)} dataset ZIPs in {args.data_dir}")
    before = snapshot(zips)

    by_class, failures = build_train_index(zips)
    total_indexed = sum(len(v) for v in by_class.values())
    print(f"Indexed {total_indexed} unique training videos in {len(by_class)} classes.")
    if len(by_class) != 12:
        print(
            f"ERROR: expected 12 training classes, found {len(by_class)}: "
            + ", ".join(sorted(by_class)),
            file=sys.stderr,
        )
        return 2

    picked, failures2 = sample_per_class(by_class, args.per_class, args.seed)
    failures += failures2
    expected = 12 * args.per_class
    if len(picked) != expected:
        print(
            f"ERROR: sampled {len(picked)} videos, expected {expected}.",
            file=sys.stderr,
        )
        return 2
    print(f"Sampled {len(picked)} videos (seed={args.seed}). Extracting to temp dir...")

    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_profile_"))
    probes: list[Probe] = []
    cleanup_ok = True
    try:
        # Group by source zip to open each archive once.
        by_zip: dict[str, list[Sample]] = {}
        for s in picked:
            by_zip.setdefault(str(s.source_zip), []).append(s)
        for zpath_str, samples in sorted(by_zip.items()):
            with zipfile.ZipFile(zpath_str) as zf:
                for s in samples:
                    dest = tmpdir / s.cls / Path(s.member).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with zf.open(s.member) as src, dest.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                    except KeyError:
                        failures.append(
                            Failure(s, "member missing from zip at extract time")
                        )
                        continue
                    probes.append(probe_video(dest, s.cls))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        cleanup_ok = not tmpdir.exists()
        print(f"Temp dir removed: {cleanup_ok} ({tmpdir})")
    if not cleanup_ok:
        print("ERROR: temp cleanup failed.", file=sys.stderr)
        return 2

    after = snapshot(zips)
    if before != after:
        print("ERROR: dataset ZIPs changed during profiling!", file=sys.stderr)
        return 2
    print("Dataset ZIPs untouched (size+mtime verified).")

    for f in failures:
        print(f"WARNING: {f.sample.cls}/{f.sample.member}: {f.reason}")

    report(probes, by_class)

    if any(not p.decoded for p in probes):
        print("NOTE: some samples failed to decode (see above); exit 0, flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

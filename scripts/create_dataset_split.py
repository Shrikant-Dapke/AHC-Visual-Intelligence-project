"""Freeze a deterministic stratified 80/20 train/validation split, NO training.

Reads the 8 dataset ZIPs in place (never modifies or extracts them beyond
reading entry names), discovers the actual internal structure, indexes all
labeled training videos, excludes the 34 unlabeled test videos, and writes:

  data/splits/train_val_split_seed42.csv   (split,class_name,zip_file,internal_path,filename)
  data/splits/train_val_summary.txt        (human-readable verification summary)

Stratification is by class only (seed 42). Duration is NOT a split feature:
per-video duration requires decoding each video (i.e. full extraction), so it
is reported here only as a leakage diagnostic from the earlier 240-video
validation survey medians, not measured per video.

Exit code: 0 on success (all checks pass), non-zero otherwise.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT.parent
ZIP_GLOB = "Train and Test-*.zip"
SEED = 42
VAL_FRACTION = 0.20
SPLIT_DIR = REPO_ROOT / "data" / "splits"
MANIFEST_NAME = "train_val_split_seed42.csv"
SUMMARY_NAME = "train_val_summary.txt"

# Median durations (s) from the 240-video validation survey (20/class).
# Reference only: duration must NOT be used as a classifier feature.
DURATION_REFERENCE: dict[str, float] = {
    "fighting_or_violence": 30.0,
    "fire": 5.8,
    "loitering_or_suspicious_presence": 29.9,
    "normal": 15.7,
    "road_spill_or_debris": 7.3,
    "smoke": 5.8,
    "stalled_or_broken_down_vehicle": 11.5,
    "traffic_accident": 5.0,
    "traffic_congestion": 5.4,
    "vehicle_blocking_traffic": 11.0,
    "waterlogging_or_flood": 5.7,
    "wrong_way_driving": 10.5,
}


@dataclass(frozen=True)
class Entry:
    class_name: str
    zip_file: str
    internal_path: str
    filename: str


def fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def discover(zips: list[Path]) -> tuple[list[Entry], list[str]]:
    """Discover structure from actual entry paths; return (train, test_names)."""
    train: dict[str, Entry] = {}
    test: set[str] = set()
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".mp4"):
                    continue
                parts = Path(name).parts
                if len(parts) < 2 or parts[0] != "Train and Test":
                    continue
                if len(parts) >= 2 and parts[1] == "test":
                    test.add(parts[-1])
                elif (
                    len(parts) == 5
                    and parts[1] == "train"
                    and parts[3] == "videos"
                ):
                    key = f"{parts[2]}/{parts[4]}"
                    if key not in train:
                        train[key] = Entry(parts[2], zpath.name, name, parts[4])
    return [train[k] for k in sorted(train)], sorted(test)


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze stratified 80/20 split.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    args = ap.parse_args()

    zips = sorted(args.data_dir.glob(ZIP_GLOB))
    if not zips:
        print(f"ERROR: no ZIPs matching {ZIP_GLOB!r} in {args.data_dir}",
              file=sys.stderr)
        return 2
    print(f"Found {len(zips)} dataset ZIPs in {args.data_dir}")

    train_entries, test_names = discover(zips)
    classes = sorted({e.class_name for e in train_entries})
    print(f"Discovered {len(train_entries)} training videos, "
          f"{len(classes)} classes, {len(test_names)} test videos.")

    errors: list[str] = []
    if len(train_entries) != 3173:
        errors.append(f"expected 3173 training videos, found {len(train_entries)}")
    if len(classes) != 12:
        errors.append(f"expected 12 classes, found {len(classes)}: {classes}")
    if len(test_names) != 34:
        errors.append(f"expected 34 test videos, found {len(test_names)}")

    rng = random.Random(args.seed)
    rows: list[tuple[str, Entry]] = []
    for cls in classes:
        members = sorted(
            (e for e in train_entries if e.class_name == cls),
            key=lambda e: (e.filename, e.internal_path),
        )
        n_val = int(round(len(members) * args.val_fraction))
        n_val = max(1, min(n_val, len(members) - 1))
        val = set(rng.sample([e.filename for e in members], n_val))
        for e in members:
            rows.append(("val" if e.filename in val else "train", e))

    # ---- verification ----
    keys = [(r[1].class_name, r[1].filename) for r in rows]
    if len(set(keys)) != len(rows):
        errors.append("duplicate video keys in split")
    train_keys = {k for s, k in zip([r[0] for r in rows], keys) if s == "train"}
    val_keys = {k for s, k in zip([r[0] for r in rows], keys) if s == "val"}
    if train_keys & val_keys:
        errors.append("train/val overlap detected")
    if len(rows) != 3173:
        errors.append(f"split covers {len(rows)} videos, expected 3173")
    test_filenames = set(test_names)
    leaked = [e.filename for _, e in rows if e.filename in test_filenames]
    if leaked:
        errors.append(f"test videos leaked into split: {leaked[:5]}")
    for cls in classes:
        n_tr = sum(1 for s, e in rows if s == "train" and e.class_name == cls)
        n_va = sum(1 for s, e in rows if s == "val" and e.class_name == cls)
        if n_tr == 0 or n_va == 0:
            errors.append(f"class {cls} missing from a split")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SPLIT_DIR / MANIFEST_NAME
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class_name", "zip_file", "internal_path", "filename"])
        for split, e in sorted(rows, key=lambda r: (r[0], r[1].class_name, r[1].filename)):
            w.writerow([split, e.class_name, e.zip_file, e.internal_path, e.filename])

    n_train = sum(1 for s, _ in rows if s == "train")
    n_val = len(rows) - n_train
    lines = [
        "AHC train/val split summary (seed=42, stratified 80/20 by class)",
        f"train: {n_train}  val: {n_val}  total: {len(rows)}  classes: {len(classes)}",
        "",
        f"{'class':38} {'train':>6} {'val':>5} {'val%':>6}",
    ]
    for cls in classes:
        n_tr = sum(1 for s, e in rows if s == "train" and e.class_name == cls)
        n_va = sum(1 for s, e in rows if s == "val" and e.class_name == cls)
        pct = 100 * n_va / (n_tr + n_va)
        ref = DURATION_REFERENCE.get(cls)
        ref_s = f"  median-dur(ref) {ref:.1f}s" if ref else ""
        lines.append(f"{cls:38} {n_tr:6d} {n_va:5d} {pct:5.1f}%{ref_s}")
    lines += [
        "",
        f"duplicates: none ({len(set(keys))} unique keys)",
        f"train/val overlap: none",
        f"test exclusion: {len(test_names)} test videos, 0 in split",
        "duration: reference medians only (per-video decode not performed); "
        "DO NOT use duration as a feature",
    ]
    (SPLIT_DIR / SUMMARY_NAME).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    stray = [p.name for p in SPLIT_DIR.iterdir()
             if p.name not in (MANIFEST_NAME, SUMMARY_NAME)]
    if stray:
        errors.append(f"unexpected files in data/splits: {stray}")

    if errors:
        print("\nFAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"\nWrote {manifest} and {SPLIT_DIR / SUMMARY_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

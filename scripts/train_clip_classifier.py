"""Train a lightweight 12-class incident classifier on frozen CLIP features.

Pipeline: frozen split manifest -> stream ZIP member -> temp file -> CLIP
ViT-B/32 frame embeddings (8 time-sampled, letterboxed, reused from
benchmark_embedding_models) -> mean-pool to one 512-d vector per video ->
multinomial logistic regression (class_weight=balanced, seed 42).

Video-level split (no frame leakage). CLIP is NEVER fine-tuned. Duration is
never a feature. Videos stream one at a time; only embeddings persist.

Outputs (all gitignored):
  <emb-dir>/<split>/<filename>.npz  per-video cache: x (512,), y, video_id
  <model-dir>/classifier.joblib  trained sklearn model
  <model-dir>/labels.json        index -> class name
  <model-dir>/metadata.json      provenance + validation metrics

The per-video cache makes extraction resumable: existing valid entries are
skipped, so an interrupted run restarts in seconds.

Usage (embed-venv python):
    python scripts/train_clip_classifier.py --per-class 2 --emb-dir ... --model-dir ...  # smoke
    python scripts/train_clip_classifier.py                            # full

Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from benchmark_embedding_models import (  # noqa: E402 (reuse pilot code)
    ClipEncoder,
    probe_duration,
    read_rgb_frames,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "splits" / "train_val_split_seed42.csv"
DEFAULT_DATA_DIR = REPO_ROOT.parent
DEFAULT_EMB_DIR = REPO_ROOT / "data" / "cache" / "v3" / "clip_embeddings"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "clip_linear_v1"

import numpy as np


def fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_rows(manifest: Path, split: str, limit: int | None,
                per_class: int | None) -> list[dict]:
    rows = [r for r in csv.DictReader(manifest.open()) if r["split"] == split]
    rows.sort(key=lambda r: (r["class_name"], r["filename"]))
    if per_class is not None:
        out = []
        by_cls: dict[str, list[dict]] = {}
        for r in rows:
            by_cls.setdefault(r["class_name"], []).append(r)
        for cls in sorted(by_cls):
            out.extend(by_cls[cls][:per_class])
        return out
    return rows[:limit] if limit else rows


def cache_path(emb_dir: Path, split: str, filename: str) -> Path:
    return emb_dir / split / f"{filename}.npz"


def load_cached(path: Path) -> tuple[np.ndarray, str] | None:
    """Return (x, y) or None if missing/invalid."""
    try:
        d = np.load(str(path))
        x = np.asarray(d["x"], dtype="float32")
        if x.shape != (512,) or not np.isfinite(x).all():
            return None
        return x, str(d["y"])
    except Exception:
        return None


def embed_split(enc: ClipEncoder, rows: list[dict], data_dir: Path,
                tmpdir: Path, emb_dir: Path, split: str
                ) -> tuple[list[str], int, int]:
    """Embed rows with skip-if-cached. Returns (video_ids_in_order, n_skip, n_fail)."""
    split_dir = emb_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    n_skip = n_fail = 0
    t0 = time.perf_counter()
    total = len(rows)
    for i, row in enumerate(rows, 1):
        cpath = cache_path(emb_dir, split, row["filename"])
        hit = load_cached(cpath)
        if hit is not None and hit[1] == row["class_name"]:
            ids.append(row["filename"])
            n_skip += 1
        else:
            dest = tmpdir / row["filename"]
            with zipfile.ZipFile(data_dir / row["zip_file"]) as zf:
                with zf.open(row["internal_path"]) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            try:
                dur = probe_duration(dest)
                frames = read_rgb_frames(dest, dur) if dur else None
                if frames is None:
                    n_fail += 1
                    print(f"  WARN {split} {row['filename']}: unreadable, skipped")
                else:
                    E = enc.embed(frames)
                    assert E.shape == (8, 512), E.shape
                    x = E.mean(axis=0).astype("float32")
                    np.savez_compressed(str(cpath), x=x, y=row["class_name"],
                                        video_id=row["filename"])
                    ids.append(row["filename"])
            finally:
                dest.unlink(missing_ok=True)
        if i % 25 == 0 or i == total:
            el = time.perf_counter() - t0
            done = i
            eta = (el / done * (total - done)) if done else 0
            print(f"  ... {split} {done}/{total} skipped={n_skip} failed={n_fail} "
                  f"elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
    return ids, n_skip, n_fail


def assemble(ids: list[str], emb_dir: Path, split: str,
             rows: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """Load cached embeddings for ids in order. Only the matrix is in RAM."""
    by_name = {r["filename"]: r["class_name"] for r in rows}
    Xs, ys = [], []
    for vid in ids:
        hit = load_cached(cache_path(emb_dir, split, vid))
        if hit is None:
            fail(f"cache entry missing after embedding: {split}/{vid}")
        Xs.append(hit[0])
        ys.append(by_name[vid])
    return np.stack(Xs), ys, ids


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CLIP linear incident classifier.")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--max-videos-per-split", type=int, default=None)
    ap.add_argument("--max-videos", type=int, default=None,
                    help="alias for --max-videos-per-split")
    ap.add_argument("--per-class", type=int, default=None,
                    help="stratified cap per class (for smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    if not args.manifest.is_file():
        fail(f"manifest not found: {args.manifest}")
    args.emb_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(args.data_dir.glob("Train and Test-*.zip"))
    before = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}

    print("Loading frozen CLIP ViT-B/32 (no fine-tuning)...")
    enc = ClipEncoder()
    size_mb, load_sec = enc.load()
    print(f"CLIP ready in {load_sec:.1f}s (~{size_mb} MB)")

    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_train_"))
    try:
        limit = args.max_videos_per_split or args.max_videos
        train_rows = load_rows(args.manifest, "train", limit, args.per_class)
        val_rows = load_rows(args.manifest, "val", limit, args.per_class)
        print(f"Train videos: {len(train_rows)}, val videos: {len(val_rows)}")
        tr_ids, tr_skip, ftr = embed_split(enc, train_rows, args.data_dir,
                                           tmpdir, args.emb_dir, "train")
        va_ids, va_skip, fva = embed_split(enc, val_rows, args.data_dir,
                                           tmpdir, args.emb_dir, "val")
        print(f"Cache hits skipped: train={tr_skip}, val={va_skip}")
        Xtr, ytr, idstr = assemble(tr_ids, args.emb_dir, "train", train_rows)
        Xva, yva, idsva = assemble(va_ids, args.emb_dir, "val", val_rows)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if tmpdir.exists():
            fail("temp cleanup failed")
        print(f"Temp dir removed: True ({tmpdir})")
    after = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}
    if before != after:
        fail("dataset ZIPs changed!")

    classes = sorted(set(ytr) | set(yva))
    print(f"Assembled matrices: train {Xtr.shape}, val {Xva.shape} "
          f"(embeddings streamed from per-video cache)")

    print(f"Training multinomial logistic regression (C={args.C}, balanced, seed={args.seed})...")
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000,
                             random_state=args.seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    proba = clf.predict_proba(Xva)

    acc = accuracy_score(yva, pred)
    macro_f1 = f1_score(yva, pred, average="macro")
    prec, rec, f1, sup = precision_recall_fscore_support(
        yva, pred, labels=clf.classes_, zero_division=0)
    cm = confusion_matrix(yva, pred, labels=clf.classes_)

    print("\n" + "=" * 64)
    print(f"VAL accuracy={acc:.4f} macro-F1={macro_f1:.4f} "
          f"(train fails={ftr}, val fails={fva})")
    print(f"{'class':34} {'prec':>6} {'rec':>6} {'f1':>6} {'sup':>5}")
    for c, p, r, f, s in zip(clf.classes_, prec, rec, f1, sup):
        print(f"{c:34} {p:6.3f} {r:6.3f} {f:6.3f} {s:5d}")
    print("\nConfusion matrix rows=true, cols=pred:")
    print(" ".join(f"{c[:6]:>6}" for c in clf.classes_))
    for c, row in zip(clf.classes_, cm):
        print(f"{c[:12]:12} " + " ".join(f"{v:6d}" for v in row))

    import joblib
    joblib.dump(clf, args.model_dir / "classifier.joblib")
    labels = {int(i): c for i, c in enumerate(clf.classes_)}
    (args.model_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    (args.model_dir / "metadata.json").write_text(json.dumps({
        "model": "clip-vit-b32 frozen + sklearn LogisticRegression",
        "seed": args.seed, "C": args.C, "aggregation": "mean-pool-8-frames",
        "classes": list(clf.classes_),
        "val_accuracy": round(float(acc), 4),
        "val_macro_f1": round(float(macro_f1), 4),
        "n_train": len(ytr), "n_val": len(yva),
        "train_failures": ftr, "val_failures": fva,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"\nSaved model -> {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

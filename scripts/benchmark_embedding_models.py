"""Embedding feasibility benchmark — model selection ONLY, NO training.

Compares pretrained image encoders on a small validation pilot for the AHC
hackathon: practicality (download size, CPU inference cost, RAM) first,
exploratory class-separation signal second. No fine-tuning, no classifier.

Reuses the frozen split manifest and the frozen preprocessing policy
(time-fraction sampling, aspect-preserved 1280x720 letterbox) from
scripts/build_feature_cache.py via import -- no second implementation.

Video set: --max-videos 20 with --sample-mode diverse (default): per-class
lists shuffled with seed 42, then round-robin, so all 12 classes are
represented (required for within/between-class diagnostics). The literal
first-20 manifest rows are a single class (fighting_or_violence) and cannot
support separation analysis; use --sample-mode first to reproduce those.

Outputs (gitignored) under --output-dir (default data/cache/v2/embedding_pilot):
  <model>.npz, benchmark_report.json

Usage (from repo root, with the embedding venv python):
    python scripts/benchmark_embedding_models.py --split val --max-videos 20

Exit 0 if >=1 encoder benchmarked, 2 if none could run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_feature_cache import (  # noqa: E402  (reuse frozen policy)
    CANON_H,
    CANON_W,
    TIME_FRACTIONS,
    letterbox,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "splits" / "train_val_split_seed42.csv"
DEFAULT_DATA_DIR = REPO_ROOT.parent
DEFAULT_OUT = REPO_ROOT / "data" / "cache" / "v2" / "embedding_pilot"

import numpy as np


def fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def read_rgb_frames(video_path: Path, duration: float):
    """Seek by time fraction, return list of canonical 1280x720 RGB frames."""
    import cv2

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frac in TIME_FRACTIONS:
            t = min(duration - 0.02, max(0.0, frac * duration))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            chs = [letterbox(rgb[:, :, c]) for c in range(3)]
            canon = np.stack(chs, axis=-1)
            assert canon.shape == (CANON_H, CANON_W, 3), canon.shape
            frames.append(canon)
        return frames
    finally:
        cap.release()


def probe_duration(video_path: Path) -> float | None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or count <= 0:
            return None
        return count / fps
    finally:
        cap.release()


@dataclass
class EncoderResult:
    name: str
    dim: int
    download_mb: float | None
    load_sec: float
    frame_times: list
    video_ids: list
    frame_embs: np.ndarray  # (V, 8, D)
    classes: list
    determinism_maxdiff: float | None


class Encoder:
    name = "base"
    dim = 0

    def load(self) -> float | None:
        """Load weights; return approx download MB if known."""
        raise NotImplementedError

    def embed(self, frames_rgb: list) -> np.ndarray:
        """(8, H, W, 3) uint8 canonical frames -> (8, D) float32."""
        raise NotImplementedError


class ClipEncoder(Encoder):
    name = "clip-vit-b32"
    dim = 512

    def load(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.manual_seed(0)
        t0 = time.perf_counter()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        self._t = __import__("torch")
        dt = time.perf_counter() - t0
        return self._hf_size_mb("openai/clip-vit-base-patch32"), dt

    def _hf_size_mb(self, repo: str) -> float | None:
        try:
            from huggingface_hub import scan_cache_dir

            total = 0
            for repo_info in scan_cache_dir().repos:
                if repo_info.repo_id == repo:
                    total += repo_info.size_on_disk
            if total:
                return round(total / 1e6, 1)
        except Exception:
            pass
        try:
            from huggingface_hub import HfApi

            files = HfApi().model_info(repo).siblings
            total = sum(f.size or 0 for f in files
                        if f.rfilename.endswith((".bin", ".safetensors")))
            return round(total / 1e6, 1) if total else None
        except Exception:
            return None

    def embed(self, frames_rgb):
        import torch

        inputs = self.processor(images=frames_rgb, return_tensors="pt")
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out[0].mean(dim=1)
            pooled = self.model.visual_projection(pooled)
        return pooled.cpu().numpy().astype("float32")


class ResNet18Encoder(Encoder):
    name = "resnet18-imagenet1k"
    dim = 512

    def load(self):
        import torch
        from torchvision.models import resnet18, ResNet18_Weights

        torch.manual_seed(0)
        t0 = time.perf_counter()
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=self.weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.tf = self.weights.transforms()
        dt = time.perf_counter() - t0
        try:
            import torch.hub as hub

            path = hub.get_dir() + "/checkpoints/resnet18-f37072fd.pth"
            size = Path(path).stat().st_size / 1e6 if Path(path).exists() else 44.7
        except Exception:
            size = 44.7
        return round(size, 1), dt

    def embed(self, frames_rgb):
        import torch
        from PIL import Image

        batch = torch.stack([self.tf(Image.fromarray(f)) for f in frames_rgb])
        with torch.no_grad():
            feats = self.model(batch)
        return feats.cpu().numpy().astype("float32")


class DinoV2Encoder(Encoder):
    name = "dinov2-vits14"
    dim = 384

    def load(self):
        import torch

        torch.manual_seed(0)
        t0 = time.perf_counter()
        self.torch = torch
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()
        dt = time.perf_counter() - t0
        try:
            import torch.hub as hub

            ckpts = list(Path(hub.get_dir(), "checkpoints").glob("*vits14*.pt"))
            size = (ckpts[0].stat().st_size / 1e6) if ckpts else 85.7
        except Exception:
            size = 85.7
        return round(size, 1), dt

    def embed(self, frames_rgb):
        import torch
        from torchvision import transforms as T

        tf = T.Compose([
            T.ToPILImage(),
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        batch = torch.stack([tf(torch.from_numpy(f).permute(2, 0, 1)) for f in frames_rgb])
        with torch.no_grad():
            feats = self.model(batch)
        return feats.cpu().numpy().astype("float32")


ENCODERS = [ClipEncoder, ResNet18Encoder, DinoV2Encoder]


def select_videos(rows: list, max_videos: int, mode: str, seed: int) -> list:
    rows = sorted(rows, key=lambda r: (r["class_name"], r["filename"]))
    if mode == "first":
        return rows[:max_videos]
    rng = random.Random(seed)
    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r["class_name"], []).append(r)
    for v in by_class.values():
        rng.shuffle(v)
    picked, idx = [], {c: 0 for c in by_class}
    order = sorted(by_class)
    while len(picked) < max_videos:
        progressed = False
        for c in order:
            if idx[c] < len(by_class[c]) and len(picked) < max_videos:
                picked.append(by_class[c][idx[c]])
                idx[c] += 1
                progressed = True
        if not progressed:
            break
    return picked


def aggregate(E: np.ndarray) -> dict:
    return {
        "mean": E.mean(axis=0),
        "max": E.max(axis=0),
        "mean_std": np.concatenate([E.mean(axis=0), E.std(axis=0)]),
    }


def separation(video_embs: np.ndarray, classes: list) -> dict:
    """Cosine-distance within vs between class diagnostic (exploratory)."""
    n = len(classes)
    norm = video_embs / (np.linalg.norm(video_embs, axis=1, keepdims=True) + 1e-12)
    within, between = [], []
    per_class: dict[str, list] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(1 - norm[i] @ norm[j])
            bucket = within if classes[i] == classes[j] else between
            bucket.append(d)
            if classes[i] == classes[j]:
                per_class.setdefault(classes[i], []).append(d)
    out = {
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
        "within_mean": float(np.mean(within)) if within else None,
        "between_mean": float(np.mean(between)) if between else None,
    }
    if within and between and np.mean(within) > 0:
        out["separation_ratio"] = float(np.mean(between) / np.mean(within))
    else:
        out["separation_ratio"] = None
    out["per_class_within"] = {c: round(float(np.mean(v)), 4) for c, v in per_class.items()}
    return out


def benchmark_encoder(enc: Encoder, videos: list, data_dir: Path, tmpdir: Path) -> EncoderResult | None:
    import torch

    print(f"\n### {enc.name} (expect dim {enc.dim})")
    rss0 = rss_mb()
    try:
        size, load_sec = enc.load()
    except Exception as e:
        print(f"SKIPPED {enc.name}: load failed: {type(e).__name__}: {e}")
        return None
    print(f"loaded in {load_sec:.1f}s, ~{size} MB, RSS {rss0:.0f}->{rss_mb():.0f} MB")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    all_embs, ids, classes, frame_times = [], [], [], []
    peak = rss_mb()
    det_diff = None
    for i, row in enumerate(videos, 1):
        dest = tmpdir / row["filename"]
        with zipfile.ZipFile(data_dir / row["zip_file"]) as zf:
            with zf.open(row["internal_path"]) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        try:
            dur = probe_duration(dest)
            if not dur:
                print(f"  WARN {row['filename']}: unprobable, skipped")
                continue
            frames = read_rgb_frames(dest, dur)
            if frames is None:
                print(f"  WARN {row['filename']}: frame read failed, skipped")
                continue
            t0 = time.perf_counter()
            E = enc.embed(frames)
            dt = time.perf_counter() - t0
            assert E.shape == (8, enc.dim), f"unexpected shape {E.shape}"
            if not np.isfinite(E).all():
                print(f"  WARN {row['filename']}: NaN/Inf in embeddings")
            if i == 1:  # determinism probe: same frames twice
                E2 = enc.embed(frames)
                det_diff = float(np.max(np.abs(E - E2)))
            all_embs.append(E)
            ids.append(row["filename"])
            classes.append(row["class_name"])
            frame_times.append(dt / 8)
            peak = max(peak, rss_mb())
        finally:
            dest.unlink(missing_ok=True)
        if i % 5 == 0 or i == len(videos):
            print(f"  ... {i}/{len(videos)} videos", flush=True)

    if not all_embs:
        print(f"SKIPPED {enc.name}: no videos embedded")
        return None
    arr = np.stack(all_embs)
    print(f"determinism maxdiff (same frames x2): {det_diff}")
    print(f"peak RSS: {peak:.0f} MB")
    return EncoderResult(enc.name, enc.dim, size, load_sec, frame_times,
                         ids, arr, classes, det_diff)


def main() -> int:
    ap = argparse.ArgumentParser(description="Embedding feasibility benchmark.")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--max-videos", type=int, default=20)
    ap.add_argument("--sample-mode", default="diverse", choices=["diverse", "first"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.manifest.is_file():
        fail(f"split manifest not found: {args.manifest}")
    rows = [r for r in csv.DictReader(args.manifest.open()) if r["split"] == args.split]
    videos = select_videos(rows, args.max_videos, args.sample_mode, args.seed)
    print(f"Pilot set: {len(videos)} videos ({args.sample_mode}), "
          f"{len({v['class_name'] for v in videos})} classes")
    for v in videos:
        print(f"  {v['class_name']}/{v['filename']}")

    zips = sorted(args.data_dir.glob("Train and Test-*.zip"))
    before = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_embed_"))

    results: list[EncoderResult] = []
    skipped: list[str] = []
    t_all = time.perf_counter()
    cleanup_ok = True
    try:
        for cls in ENCODERS:
            r = benchmark_encoder(cls(), videos, args.data_dir, tmpdir)
            if r is None:
                skipped.append(cls.name if isinstance(cls.name, str) else str(cls))
            else:
                results.append(r)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        cleanup_ok = not tmpdir.exists()
        print(f"Temp dir removed: {cleanup_ok} ({tmpdir})")
    if not cleanup_ok:
        return 2
    after = {str(z): (z.stat().st_size, z.stat().st_mtime) for z in zips}
    if before != after:
        print("ERROR: dataset ZIPs changed!", file=sys.stderr)
        return 2
    print("Dataset ZIPs untouched (size+mtime verified).")

    if not results:
        print("ERROR: no encoder could run.", file=sys.stderr)
        return 2

    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "videos": len(videos), "device": "cpu", "models": {}}
    print("\n" + "=" * 72)
    print("EMBEDDING FEASIBILITY REPORT (exploratory, NOT a performance claim)")
    print("=" * 72)
    for r in results:
        agg_mean = np.stack([aggregate(E)["mean"] for E in r.frame_embs])
        agg_max = np.stack([aggregate(E)["max"] for E in r.frame_embs])
        agg_ms = np.stack([aggregate(E)["mean_std"] for E in r.frame_embs])
        sep_mean = separation(agg_mean, r.classes)
        sep_max = separation(agg_max, r.classes)
        total = sum(r.frame_times) * 8
        entry = {
            "dim": r.dim, "download_mb": r.download_mb, "device": "cpu",
            "load_sec": round(r.load_sec, 1),
            "per_frame_sec": round(float(np.mean(r.frame_times)), 3),
            "per_video_sec": round(float(np.mean(r.frame_times)) * 8, 2),
            "total_sec": round(total, 1),
            "videos_embedded": len(r.video_ids),
            "determinism_maxdiff": r.determinism_maxdiff,
            "finite": bool(np.isfinite(r.frame_embs).all()),
            "sep_mean_pool": sep_mean, "sep_max_pool": sep_max,
            "meanstd_dim": int(agg_ms.shape[1]),
        }
        summary["models"][r.name] = entry
        np.savez_compressed(
            args.output_dir / f"{r.name}.npz", frame_embs=r.frame_embs,
            video_ids=np.array(r.video_ids), classes=np.array(r.classes),
            agg_mean=agg_mean, agg_max=agg_max, agg_mean_std=agg_ms)
        print(f"\n--- {r.name} (dim {r.dim}, ~{r.download_mb} MB, cpu) ---")
        print(f"load {r.load_sec:.1f}s | frame {np.mean(r.frame_times):.3f}s | "
              f"video {np.mean(r.frame_times)*8:.2f}s | total {total:.0f}s")
        print(f"finite={entry['finite']} determinism_maxdiff={r.determinism_maxdiff}")
        for tag, s in (("mean-pool", sep_mean), ("max-pool", sep_max)):
            print(f"{tag}: within={s['within_mean']:.4f} (n={s['n_within_pairs']}) "
                  f"between={s['between_mean']:.4f} (n={s['n_between_pairs']}) "
                  f"ratio={s['separation_ratio'] and round(s['separation_ratio'],3)}")
    if skipped:
        print(f"\nSkipped: {skipped}")
        summary["skipped"] = skipped
    (args.output_dir / "benchmark_report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.output_dir} ({time.perf_counter()-t_all:.0f}s end-to-end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

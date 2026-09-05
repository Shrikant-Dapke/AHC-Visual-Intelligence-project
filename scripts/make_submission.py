"""Official submission generator — Evaluation E001-E028 (Task 12).

Reads the Evaluation ZIP (videos + per-level videos.csv; GT intentionally
absent), runs the CURRENT frozen pipeline, writes submission_run_01.json:

- L1 (E001-E020): video-level CLIP class only; start/end MUST be null.
  normal -> no events; anomaly -> one event with null timestamps.
- L2/L3 (E021-E028): temporal detector events with measured timestamps.
  If temporal finds nothing but video-level CLIP says anomaly, emit one
  whole-video fallback event (marked in explanation). Never fabricate.

Level comes from the ZIP directory layout (L1/L2/L3 folders), never from GT.
No retraining, no weight changes. Videos stream via temp, one at a time.

Usage (backend venv python, from repo root):
    python scripts/make_submission.py --eval-zip <path> --out submission_run_01.json

Exit 0 on success with schema validation; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "clip_linear_v1"


class TimedEncoder:
    """Wraps encoder.embed to record real per-batch timings (no behavior change)."""

    def __init__(self, enc):
        self._e = enc
        self.calls: list[tuple[int, float]] = []  # (n_frames, seconds)

    def embed(self, frames):
        import time as _t

        t0 = _t.perf_counter()
        out = self._e.embed(frames)
        self.calls.append((len(frames), _t.perf_counter() - t0))
        return out

    def __getattr__(self, name):
        return getattr(self._e, name)


class TimedClassifier:
    """Wraps predict_proba/decision_function with real timings."""

    def __init__(self, clf):
        self._c = clf
        self.calls: list[float] = []

    def _timed(self, fn, *a, **k):
        import time as _t

        t0 = _t.perf_counter()
        out = fn(*a, **k)
        self.calls.append(_t.perf_counter() - t0)
        return out

    def predict_proba(self, x):
        return self._timed(self._c.predict_proba, x)

    def decision_function(self, x):
        return self._timed(self._c.decision_function, x)

    def predict(self, x):
        return self._c.predict(x)

    def __getattr__(self, name):
        return getattr(self._c, name)


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def model_runtime_entry(name: str, times: list[float]) -> dict:
    ms = [t * 1000.0 for t in times]
    return {"model_name": name, "call_count": len(ms),
            "total_time_ms": round(sum(ms), 2),
            "average_time_ms": round(sum(ms) / max(len(ms), 1), 2),
            "p50_time_ms": round(pct(ms, 0.5), 2),
            "p95_time_ms": round(pct(ms, 0.95), 2),
            "max_time_ms": round(max(ms) if ms else 0.0, 2)}


def hardware_string() -> str:
    import torch

    return (f"{platform.processor() or platform.machine()} "
            f"({__import__('os').cpu_count()} cores), "
            f"CPU-only PyTorch {torch.__version__}, "
            f"CUDA {'available' if torch.cuda.is_available() else 'unavailable'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate official submission.")
    ap.add_argument("--eval-zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "submission_run_01.json")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--levels", default="L1,L2,L3")
    ap.add_argument("--score-mode", default="proba", choices=["proba", "logit"])
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--per-class-threshold", default="")
    ap.add_argument("--min-duration", type=float, default=4.0)
    ap.add_argument("--merge-gap", type=float, default=6.0)
    ap.add_argument("--min-peak", type=float, default=0.35)
    ap.add_argument("--coarse-sec", type=float, default=3.0)
    ap.add_argument("--adaptive-q", type=float, default=0.0)
    ap.add_argument("--top1-gate", action="store_true")
    ap.add_argument("--submission-id", default="ahc-visual-intelligence-run-01")
    args = ap.parse_args()

    from app.services.temporal_events import (
        TemporalConfig, TemporalStack,
    )
    from benchmark_embedding_models import (
        TIME_FRACTIONS, probe_duration, read_rgb_frames,
    )

    if not args.eval_zip.is_file():
        print(f"ERROR: eval zip not found: {args.eval_zip}", file=sys.stderr)
        return 2

    per_thr = {}
    for kv in (args.per_class_threshold or "").split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            per_thr[k.strip()] = float(v)
    cfg = TemporalConfig(
        score_mode=args.score_mode, default_threshold=args.threshold,
        thresholds=per_thr, min_duration=args.min_duration,
        merge_gap=args.merge_gap, min_peak=args.min_peak,
        coarse_sec=args.coarse_sec, adaptive_quantile=args.adaptive_q,
        top1_gate=args.top1_gate)

    t_all = time.perf_counter()
    stack = TemporalStack(args.model_dir)
    stack.encoder = TimedEncoder(stack.encoder)
    stack.classifier = TimedClassifier(stack.classifier)
    print(f"stack ready (model: clip-vit-b32 frozen + logreg, "
          f"classes={len(stack.classes)})", flush=True)

    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    predictions = []
    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_sub_"))
    try:
        with zipfile.ZipFile(args.eval_zip) as zf:
            for lvl in levels:
                rows = list(csv.DictReader(io.StringIO(
                    zf.read(f"Evaluation/{lvl}/videos.csv").decode())))
                for row in rows:
                    vid = row["video_id"].strip()
                    member = f"Evaluation/{lvl}/{row['filename'].strip()}"
                    dest = tmpdir / f"{vid}.mp4"
                    with zf.open(member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    try:
                        predictions.append(process_video(
                            vid, lvl, dest, stack, cfg))
                    finally:
                        dest.unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"temp cleaned: {not tmpdir.exists()}", flush=True)

    submission = {
        "schema_version": "1.0",
        "submission_id": args.submission_id,
        "model_name": "clip-vit-b32-frozen_sklearn-logreg-corrected-v2",
        "temporal_config": {
            "score_mode": cfg.score_mode,
            "default_threshold": cfg.default_threshold,
            "per_class_thresholds": per_thr,
            "min_duration": cfg.min_duration,
            "merge_gap": cfg.merge_gap,
            "min_peak": cfg.min_peak,
            "coarse_sec": cfg.coarse_sec,
            "adaptive_quantile": cfg.adaptive_quantile,
            "top1_gate": cfg.top1_gate,
            "yolo_fusion": "off",
        },
        "run_metadata": {
            "total_wall_time_ms": round((time.perf_counter() - t_all) * 1000.0, 2),
            "max_parallel_videos": 1,
            "hardware": hardware_string(),
        },
        "predictions": predictions,
    }
    validate_submission(submission)
    args.out.write_text(json.dumps(submission, indent=1))
    print(f"wrote {args.out} ({len(predictions)} videos)", flush=True)
    return 0


def process_video(vid: str, lvl: str, dest: Path, stack, cfg) -> dict:
    from benchmark_embedding_models import probe_duration, read_rgb_frames

    from app.services.temporal_events import detect_temporal_events

    t0 = time.perf_counter()
    n_embed_calls0 = len(stack.encoder.calls)
    n_clf_calls0 = len(stack.classifier.calls)
    dur = probe_duration(dest)
    if not dur:
        raise RuntimeError(f"could not decode {vid}")
    # Video-level class (frozen policy: 8 time-sampled frames, mean-pool).
    frames = read_rgb_frames(dest, dur)
    if frames is None:
        raise RuntimeError(f"could not sample frames for {vid}")
    probs = stack.proba(frames)
    top = max(probs, key=probs.get)

    events = []
    if lvl == "L1":
        # Organizer rule: L1 is classification-only; timestamps MUST be null.
        if top != "normal":
            events.append({
                "class_name": top,
                "start_time_sec": None,
                "end_time_sec": None,
                "explanation": (
                    f"Video-level CLIP classification (clip-vit-b32 frozen + "
                    f"logistic regression, corrected-v2 labels): {top} "
                    f"(confidence {probs[top]:.2f}) spans the analyzed video "
                    f"[0.0, {dur:.1f}s]. Level 1 is classification-only; "
                    f"no within-video temporal localization is reported."),
            })
    else:
        evs = detect_temporal_events(dest, dur, stack, cfg)
        for e in evs:
            events.append({
                "class_name": e.class_name,
                "start_time_sec": e.start,
                "end_time_sec": e.end,
                "explanation": e.explanation,
                "confidence": e.confidence,
            })
        if not events and top != "normal":
            # Honest fallback: anomaly present per video-level verdict but no
            # localized span survived thresholds; report full span as fallback.
            ev0 = {
                "class_name": top,
                "start_time_sec": 0.0,
                "end_time_sec": round(dur, 2),
                "explanation": (
                    f"Video-level CLIP verdict is {top} "
                    f"(confidence {probs[top]:.2f}) but no temporal span "
                    f"survived detection thresholds; reporting the full "
                    f"analyzed span as a low-confidence fallback."),
                "confidence": round(probs[top], 4),
                "fallback": True,
            }
            events.append(ev0)

    e_calls = stack.encoder.calls[n_embed_calls0:]
    c_calls = stack.classifier.calls[n_clf_calls0:]
    frames_n = sum(n for n, _ in e_calls)
    pred = {"video_id": vid, "events": events,
            "runtime_metadata": {
                "frames_processed": frames_n,
                "chunks_processed": len(e_calls),
                "end_to_end_internal_time_ms": round(
                    (time.perf_counter() - t0) * 1000.0, 2),
                "model_runtimes": [
                    model_runtime_entry("clip-vit-b32-embed",
                                        [t for _, t in e_calls]),
                    model_runtime_entry("logistic-regression-head", c_calls),
                ]}}
    print(f"{vid} ({lvl}, {dur:.0f}s): " +
          (", ".join(f"{e['class_name']} {e['start_time_sec']}-{e['end_time_sec']}"
                     for e in events) or "no events"), flush=True)
    return pred


def validate_submission(sub: dict) -> None:
    assert sub["schema_version"] == "1.0"
    assert isinstance(sub["predictions"], list) and sub["predictions"]
    for p in sub["predictions"]:
        assert set(p) >= {"video_id", "events", "runtime_metadata"}, p.keys()
        lvl = ("L1" if p["video_id"] <= "E020" else
               "L2" if p["video_id"] <= "E024" else "L3")
        for e in p["events"]:
            assert set(e) >= {"class_name", "start_time_sec",
                              "end_time_sec", "explanation"}, e.keys()
            assert isinstance(e["class_name"], str) and e["class_name"]
            assert isinstance(e["explanation"], str) and e["explanation"]
            if lvl == "L1":
                assert e["start_time_sec"] is None and e["end_time_sec"] is None, \
                    f"L1 timestamps must be null: {p['video_id']}"
            else:
                assert isinstance(e["start_time_sec"], (int, float)), e
                assert isinstance(e["end_time_sec"], (int, float)), e
                assert 0 <= e["start_time_sec"] < e["end_time_sec"], e
    print(f"schema validation OK ({len(sub['predictions'])} predictions)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

"""TEMP-only evaluation harness for temporal event detection (Task 12).

GT (test/ground_truth.csv) is read from the dataset ZIP in memory and used
ONLY here to score predictions. It NEVER enters inference code paths.
Predictions are generated independently by backend/app/services/temporal_events.py.

Usage (backend venv python, from repo root):
    python scripts/eval_temporal.py --videos T025,T026,T027
    python scripts/eval_temporal.py --all-l23
    python scripts/eval_temporal.py --all-l23 --score-mode logit --threshold 0.6

Matching: greedy by IoU; a predicted event matches a GT event iff same class
and IoU >= --iou (default 0.3; also reported at 0.5).

Outputs: stdout report. Optional --out-dir (defaults to system temp) for JSON.
Videos are extracted to temp one at a time and deleted immediately.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DATA_DIR = REPO_ROOT.parent
MODEL_DIR = REPO_ROOT / "models" / "clip_linear_v1"


@dataclass
class GTEvent:
    video: str
    level: int
    cls: str
    start: float | None
    end: float | None


def load_gt() -> tuple[list[GTEvent], dict[str, str]]:
    """Read GT + video map from ZIPs in memory. EVAL ONLY."""
    gt: list[GTEvent] = []
    vmap: dict[str, str] = {}
    for zp in sorted(DATA_DIR.glob("Train and Test-*.zip")):
        with zipfile.ZipFile(zp) as zf:
            names = set(zf.namelist())
            if "Train and Test/test/ground_truth.csv" in names:
                text = zf.read("Train and Test/test/ground_truth.csv").decode()
                for row in csv.DictReader(io.StringIO(text)):
                    s = row["start_time_sec"].strip()
                    e = row["end_time_sec"].strip()
                    gt.append(GTEvent(
                        video=row["video_id"].strip(),
                        level=int(row["level"]),
                        cls=row["class_name"].strip(),
                        start=float(s) if s else None,
                        end=float(e) if e else None))
            if "Train and Test/test/videos.csv" in names:
                text = zf.read("Train and Test/test/videos.csv").decode()
                for row in csv.DictReader(io.StringIO(text)):
                    vmap[row["video_id"].strip()] = (
                        f"Train and Test/test/{row['filename'].strip()}")
    return gt, vmap


def extract_video(zip_path: Path, member: str, tmpdir: Path) -> Path:
    dest = tmpdir / Path(member).name
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest


def find_zip_for(member: str):
    for zp in sorted(DATA_DIR.glob("Train and Test-*.zip")):
        with zipfile.ZipFile(zp) as zf:
            if member in zf.namelist():
                return zp
    return None


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match_events(preds: list, gts: list, thr: float):
    """Greedy IoU matching, same class only. Returns (matches, unpred, ungt)."""
    cands = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p["class"] != g["class"]:
                continue
            v = iou(p["start"], p["end"], g["start"], g["end"])
            if v >= thr:
                cands.append((v, i, j))
    cands.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for v, i, j in cands:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matches.append({"pred": preds[i], "gt": gts[j], "iou": round(v, 3)})
    return (matches,
            [p for i, p in enumerate(preds) if i not in used_p],
            [g for j, g in enumerate(gts) if j not in used_g])


def scan_cache_key(vid: str, duration: float, coarse_sec: float,
                     coarse_frames: int) -> str:
    import hashlib
    raw = f"{vid}|{duration:.2f}|{coarse_sec}|{coarse_frames}|clip-vit-b32".encode()
    return hashlib.md5(raw).hexdigest()[:16]


def load_cached_scan(cache_dir: Path, key: str):
    from app.services.temporal_events import WindowScore
    p = cache_dir / f"{key}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return [WindowScore(start=w["start"], end=w["end"], probs=w["probs"],
                            logits=w.get("logits") or {},
                            top=w["top"], top_p=w["top_p"])
                for w in data["windows"]]
    except Exception:
        return None


def save_cached_scan(cache_dir: Path, key: str, windows) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {"windows": [{"start": w.start, "end": w.end, "probs": w.probs,
                         "logits": w.logits, "top": w.top, "top_p": w.top_p}
                        for w in windows]}
    (cache_dir / f"{key}.json").write_text(json.dumps(data))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate temporal events (temp-only).")
    ap.add_argument("--videos", default="",
                    help="comma list like T025,T026 (default: all L2/3 + L1 check)")
    ap.add_argument("--all-l23", action="store_true")
    ap.add_argument("--l1-check", action="store_true",
                    help="also run video-level classification check on T001-T024")
    ap.add_argument("--score-mode", default="proba", choices=["proba", "logit"])
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--per-class-threshold", default="",
                    help="e.g. traffic_congestion:0.6,fire:0.5")
    ap.add_argument("--min-duration", type=float, default=4.0)
    ap.add_argument("--merge-gap", type=float, default=6.0)
    ap.add_argument("--min-peak", type=float, default=0.35)
    ap.add_argument("--coarse-sec", type=float, default=3.0)
    ap.add_argument("--w-block", type=float, default=0.0)
    ap.add_argument("--w-cong", type=float, default=0.0)
    ap.add_argument("--adaptive-q", type=float, default=0.0)
    ap.add_argument("--top1-gate", action="store_true")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--scan-cache", type=Path,
                    default=REPO_ROOT / "data" / "cache" / "v3" / "temporal_scans",
                    help="persist coarse+fine CLIP grids across tuning runs")
    ap.add_argument("--no-scan-cache", action="store_true")
    args = ap.parse_args()

    from app.services.temporal_events import (
        TemporalConfig, TemporalStack, detect_temporal_events,
    )
    from benchmark_embedding_models import probe_duration

    gt_all, vmap = load_gt()
    if args.videos:
        vids = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        vids = [f"T{i:03d}" for i in range(25, 35)]

    per_thr = {}
    for kv in (args.per_class_threshold or "").split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            per_thr[k.strip()] = float(v)
    cfg = TemporalConfig(
        score_mode=args.score_mode, default_threshold=args.threshold,
        thresholds=per_thr, min_duration=args.min_duration,
        merge_gap=args.merge_gap, min_peak=args.min_peak,
        coarse_sec=args.coarse_sec, w_block_still=args.w_block,
        w_cong_density=args.w_cong, adaptive_quantile=args.adaptive_q,
        top1_gate=args.top1_gate)
    print(f"config: mode={cfg.score_mode} thr={args.threshold} "
          f"per-class={per_thr or '{}'} min_dur={cfg.min_duration} "
          f"merge={cfg.merge_gap} w_block={args.w_block} w_cong={args.w_cong} "
          f"adapt_q={args.adaptive_q} top1={args.top1_gate} "
          f"yolo={'off' if args.no_yolo else 'on'}")

    t0 = time.perf_counter()
    stack = TemporalStack(MODEL_DIR)
    print(f"stack ready in {time.perf_counter()-t0:.1f}s")

    yolo = None
    if not args.no_yolo:
        from app.inference.detector import YoloDetector
        from app.inference.tracker import make_tracker
        from app.services.object_analysis import analyze_video_objects
        yolo = {"det": None, "tracker_fn": make_tracker,
                "analyze": analyze_video_objects, "YoloDetector": YoloDetector}

    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_eval_"))
    results = []
    try:
        for vid in vids:
            member = vmap.get(vid)
            if not member:
                print(f"{vid}: NOT IN ZIP, skipped")
                continue
            zp = find_zip_for(member)
            dest = extract_video(zp, member, tmpdir)
            try:
                dur = probe_duration(dest)
                tv0 = time.perf_counter()
                tracks, dims = None, (1280, 720)
                if yolo is not None:
                    yolo_cache = (args.scan_cache / f"yolo_{vid}.json"
                                  if not args.no_scan_cache else None)
                    if yolo_cache is not None and yolo_cache.is_file():
                        try:
                            yd = json.loads(yolo_cache.read_text())
                            tracks = _tracks_from_list(yd["tracks"])
                            dims = tuple(yd["dims"])
                        except Exception:
                            tracks = None
                    if tracks is None:
                        if yolo["det"] is None:
                            yolo["det"] = yolo["YoloDetector"]()
                            yolo["det"].load()
                        ev_tmp = tmpdir / f"ev_{vid}"
                        sfps = min(2.0, 600.0 / max(dur, 1.0))
                        obj = yolo["analyze"](
                            dest, yolo["det"], yolo["tracker_fn"]("iou"),
                            sample_fps=sfps, max_frames=600,
                            evidence_dir=ev_tmp, evidence_name=vid, progress=False)
                        # tracks already stitched inside analyze_video_objects
                        tracks = _tracks_from_obj(obj)
                        dims = (obj.get("frame_width") or 1280,
                                obj.get("frame_height") or 720)
                        shutil.rmtree(ev_tmp, ignore_errors=True)
                        if yolo_cache is not None:
                            yolo_cache.parent.mkdir(parents=True, exist_ok=True)
                            yolo_cache.write_text(json.dumps({
                                "dims": list(dims),
                                "tracks": [
                                    {"track_id": t.track_id,
                                     "class_name": t.class_name,
                                     "first_seen": t.first_seen,
                                     "last_seen": t.last_seen,
                                     "displacement_px": t.displacement()}
                                    for t in tracks]}))
                from app.services.temporal_events import scan_windows
                coarse, fine = None, None
                if not args.no_scan_cache:
                    kc = scan_cache_key(vid, dur, cfg.coarse_sec, cfg.coarse_frames)
                    kf = scan_cache_key(vid, dur, cfg.fine_sec, cfg.fine_frames)
                    coarse = load_cached_scan(args.scan_cache, kc)
                    fine = load_cached_scan(args.scan_cache, kf)
                    if coarse is None:
                        coarse = scan_windows(
                            dest, dur, stack, cfg.coarse_sec, cfg.coarse_frames,
                            progress=False)
                    save_cached_scan(args.scan_cache, kc, coarse)
                    if fine is None:
                        fine = scan_windows(
                            dest, dur, stack, cfg.fine_sec, cfg.fine_frames,
                            progress=False)
                    save_cached_scan(args.scan_cache, kf, fine)
                evs = detect_temporal_events(
                    dest, dur, stack, cfg,
                    yolo_tracks=tracks, frame_dims=dims,
                    coarse_windows=coarse, fine_windows=fine)
                dt = time.perf_counter() - tv0
                preds = [{"video": vid, "class": e.class_name,
                          "start": e.start, "end": e.end,
                          "conf": e.confidence, "fallback": e.fallback}
                         for e in evs]
                results.append({"video": vid, "duration": round(dur, 1),
                                "seconds": round(dt, 1), "preds": preds})
                print(f"{vid} ({dur:.0f}s, {dt:.0f}s): " +
                      (", ".join(f"{p['class']} {p['start']}-{p['end']}@{p['conf']:.2f}"
                                 for p in preds) or "NO EVENTS"))
            finally:
                dest.unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"temp cleaned: {not tmpdir.exists()}")

    # ---- scoring (GT used ONLY here) ----
    gt_by_vid: dict[str, list] = {}
    for g in gt_all:
        if g.start is not None and g.end is not None and g.cls != "normal":
            gt_by_vid.setdefault(g.video, []).append(
                {"class": g.cls, "start": g.start, "end": g.end})
    print(f"\n=== event match @IoU>={args.iou} (D2=T025-T030, D3=T031-T034) ===")
    print(f"{'video':6} {'pred_class':28} {'pred_span':>15} "
          f"{'gt_class':28} {'gt_span':>15} {'iou':>6}")
    tot_p = tot_tp = tot_g = 0
    d2 = [0, 0, 0]
    d3 = [0, 0, 0]
    per_cls: dict[str, list] = {}
    for r in results:
        gts = gt_by_vid.get(r["video"], [])
        m, up, ug = match_events(r["preds"], gts, args.iou)
        lvl = "D2" if r["video"] <= "T030" else "D3"
        for x in m:
            p, g = x["pred"], x["gt"]
            print(f"{r['video']:6} {p['class']:28} "
                  f"{p['start']:>6.1f}-{p['end']:<6.1f} "
                  f"{g['class']:28} {g['start']:>6.1f}-{g['end']:<6.1f} {x['iou']:>6.2f}")
            per_cls.setdefault(p["class"], [0, 0, 0])
            per_cls[p["class"]][1] += 1
        for p in up:
            print(f"{r['video']:6} {p['class']:28} "
                  f"{p['start']:>6.1f}-{p['end']:<6.1f} {'--no-match--':>44}")
        for g in ug:
            print(f"{r['video']:6} {'--missed--':28} {'':>15} "
                  f"{g['class']:28} {g['start']:>6.1f}-{g['end']:<6.1f}")
            per_cls.setdefault(g["class"], [0, 0, 0])
            per_cls[g["class"]][2] += 1
        for p in r["preds"]:
            per_cls.setdefault(p["class"], [0, 0, 0])
            per_cls[p["class"]][0] += 1
        tot_p += len(r["preds"])
        tot_tp += len(m)
        tot_g += len(gts)
        acc = d2 if lvl == "D2" else d3
        acc[0] += len(r["preds"])
        acc[1] += len(m)
        acc[2] += len(gts)
        if m:
            mae_s = sum(abs(x["pred"]["start"] - x["gt"]["start"]) for x in m) / len(m)
            mae_e = sum(abs(x["pred"]["end"] - x["gt"]["end"]) for x in m) / len(m)
            print(f"  -> {r['video']}: {len(m)}/{len(r['preds'])} matched, "
                  f"start-MAE {mae_s:.1f}s end-MAE {mae_e:.1f}s")

    def pr(tp_, p_, g_):
        return (tp_ / p_ if p_ else 0.0, tp_ / g_ if g_ else 0.0)

    for tag, (p_, tp_, g_) in (("ALL", (tot_p, tot_tp, tot_g)),
                               ("D2", tuple(d2)), ("D3", tuple(d3))):
        pr_, re_ = pr(tp_, p_, g_)
        print(f"{tag}: pred={p_} matched={tp_} gt={g_} "
              f"P={pr_:.3f} R={re_:.3f}")
    print("per-class [pred, matched, gt]:")
    for c, (p_, tp_, g_) in sorted(per_cls.items()):
        pr_, re_ = pr(tp_, p_, g_)
        print(f"  {c:34} P={pr_:.3f} R={re_:.3f} ({tp_}/{p_}/{g_})")

    if args.l1_check:
        print("\n=== L1 classification-only (video-level CLIP vs GT class) ===")
        gt_cls = {g.video: g.cls for g in gt_all if g.video <= "T024"}
        _l1_check(stack, gt_cls)

    out = {"config": {"score_mode": cfg.score_mode,
                      "threshold": args.threshold,
                      "per_class": per_thr, "min_duration": cfg.min_duration,
                      "merge_gap": cfg.merge_gap, "w_block": args.w_block,
                      "w_cong": args.w_cong},
           "results": results}
    odir = args.out_dir or Path(tempfile.gettempdir())
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "eval_temporal.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {(odir / 'eval_temporal.json')}")
    return 0


def _tracks_from_obj(obj: dict):
    """Rebuild light track records from analyze_video_objects output."""
    from types import SimpleNamespace

    out = []
    for t in obj.get("tracks", []):
        d = t.get("displacement_px", 0.0)
        out.append(SimpleNamespace(
            track_id=t["track_id"], class_name=t["class"],
            first_seen=t["first_seen"], last_seen=t["last_seen"],
            displacement=(lambda d=d: d)))
    return out


def _tracks_from_list(items: list):
    from types import SimpleNamespace

    out = []
    for t in items:
        d = t.get("displacement_px", 0.0)
        out.append(SimpleNamespace(
            track_id=t["track_id"], class_name=t["class_name"],
            first_seen=t["first_seen"], last_seen=t["last_seen"],
            displacement=(lambda d=d: d)))
    return out


def _l1_check(stack, gt_cls: dict) -> None:
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from benchmark_embedding_models import probe_duration, read_rgb_frames

    tmpdir = Path(tempfile.mkdtemp(prefix="ahc_l1_"))
    try:
        ok = tot = 0
        for i in range(1, 25):
            vid = f"T{i:03d}"
            member = f"Train and Test/test/videos/{vid}.mp4"
            zp = find_zip_for(member)
            if not zp:
                continue
            dest = extract_video(zp, member, tmpdir)
            try:
                dur = probe_duration(dest)
                frames = read_rgb_frames(dest, dur)
                probs = stack.proba(frames)
                pred = max(probs, key=probs.get)
                tot += 1
            finally:
                dest.unlink(missing_ok=True)
            good = "OK " if pred == gt_cls.get(vid) else "MISS"
            ok += pred == gt_cls.get(vid)
            print(f"  {vid}: pred={pred:34} gt={gt_cls.get(vid):34} {good}")
        print(f"  L1 accuracy: {ok}/{tot} = {ok/max(tot,1):.3f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

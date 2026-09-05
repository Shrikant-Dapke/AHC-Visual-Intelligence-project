# AHC Visual Intelligence

AI-powered road/incident video intelligence — hackathon demo.

Upload an MP4 road/traffic video → get the CLIP incident prediction with confidence and source attribution, YOLO object evidence with tracking, an incident activity window (start/peak/end), pre/peak/post evidence frames, and a deterministic evidence-based explanation — all in a dark dashboard.

## Hackathon objective

Demo-ready and reliable on a CPU-only laptop. Pipeline: upload → CLIP incident classification (primary) → YOLOv8n object detection + tracking (supporting evidence) → temporal activity window → evidence JPEGs → dashboard. The motion-heuristic baseline remains as an honest fallback when CLIP is unavailable.

## Architecture

```
browser (Next.js 14, TS)
  │  POST /api/analyze (multipart .mp4)
  ▼
FastAPI backend
  ├─ inference/clip.py      (frozen CLIP ViT-B/32 + sklearn head; lazy singleton)
  ├─ inference/detector.py  (YOLOv8n, CPU, incremental — evidence only)
  ├─ inference/tracker.py   (IoU default, ByteTrack optional)
  ├─ services/unified.py    (CLIP verdict + YOLO evidence → AnalyzeResult)
  ├─ services/video.py      (cv2.VideoCapture probe + sampling, no ffmpeg)
  ├─ services/explain.py    (deterministic summary, no LLM)
  └─ api/routes.py          (health / analyze / results/{job_id} / evidence/{job_id}/{kind})
```

Frontend talks to the backend only via `NEXT_PUBLIC_API_BASE_URL` (no hardcoded prod URLs).

## Repository structure

```
frontend/  # Next.js App Router dashboard
backend/   # FastAPI app, CLIP + YOLO + explain services
shared/    # types.ts mirroring backend schemas
scripts/   # training, CLIP single-video inference, object-analysis CLI
data/splits/  # versioned train/val manifest (the only committed data)
```

## Local setup

Prereqs: Python 3.12+, Node 20+, git. No ffmpeg needed.

## Backend setup

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/Activate.ps1
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --port 8000
# workdir: backend/
```

Health: `http://localhost:8000/api/health` (also reports `clip_available` / `yolo_available`).

First run downloads YOLOv8n weights once (~6 MB, cached at repo root as `yolov8n.pt`, never committed). CLIP transformer weights (~1.2 GB) resolve from the shared Hugging Face cache when present, otherwise download once on first CLIP inference.

## Frontend setup

```powershell
npm install --prefix frontend
npm run dev --prefix frontend
# open http://localhost:3000
```

Set `frontend/.env.local` from `.env.example` if the backend is not on `localhost:8000`.

## Model configuration

All backend paths are overridable via environment (or `backend/.env`, see `backend/.env.example`); relative paths resolve against the repo root:

- `CLIP_MODEL_DIR` (default `models/clip_linear_v1`) — must contain `classifier.joblib` + `labels.json` as written by `scripts/train_clip_classifier.py`. While training is still running the backend reports CLIP as unavailable and falls back honestly — it never fakes predictions.
- `EVIDENCE_DIR` (default `data/evidence`)
- `YOLO_ENABLED`, `YOLO_MODEL` (default `yolov8n.pt`), `YOLO_CONF`, `YOLO_IMGSZ`, `YOLO_SAMPLE_FPS` (default `2.0`), `YOLO_TRACKER` (`iou`|`bytetrack`), `YOLO_MAX_FRAMES`

## Demo workflow

1. Open `http://localhost:3000`, drop an MP4 (max 200 MB).
2. Press **Analyze video** — staged status is shown (no fake percentages; CPU analysis takes ~1–3 min for a short clip once models are warm).
3. Read the hero: incident class, confidence, and source badge (**CLIP classifier** vs **Motion heuristic** — never confused).
4. Scrub the incident window (Start—Peak—End chips seek the player), open pre/peak/post evidence, review object counts and track insights.

Small real test clip used for verification: `traffic_accident` video → `vehicle_blocking_traffic` 79.2% (CLIP) with 6 car tracks. Test videos are unlabeled files, so treat demo labels as indicative, not ground truth.

## Official submission

```powershell
python scripts/make_submission.py --eval-zip <path-to-Evaluation.zip> --out submission_run_01.json
# backend venv python, from repo root
```

Reads the Evaluation ZIP in place (per-level `videos.csv`; videos stream via temp one at a time): L1 emits video-level classes with null timestamps (classification-only), L2/L3 emit measured temporal spans with a whole-video fallback when nothing localizes. Validates the schema before writing. Tuning/analysis harness: `scripts/eval_temporal.py` (ground truth, when available, is used for scoring only — never inside inference).

## API endpoints

- `GET /api/health` → `{ status, model_backend, labels_configured, clip_available, clip_reason, yolo_available, yolo_reason }`
- `POST /api/analyze` (multipart, field `file`, `.mp4` only) → unified `AnalyzeResult`
- `GET /api/results/{job_id}` → stored `AnalyzeResult` (in-memory, no DB; survives while the process lives, but the UI does not refetch on page refresh)
- `GET /api/evidence/{job_id}/{pre|peak|post}` → evidence JPEG (kind-whitelisted, traversal-proof; raw filesystem paths are never exposed)

Unified result adds (all backward-compatible): `incident_source` (`clip`|`heuristic`|`none` — YOLO never sets this), `timeline {start,peak,end}`, `objects`, `tracks`, `evidence [{kind,timestamp,url}]`, `warnings`, and `thumbnail_urls` pointing at the evidence routes.

## ML / inference architecture

- **Incident classifier (primary):** frozen CLIP ViT-B/32 image embeddings (8 time-sampled letterboxed frames, mean-pooled — duration is never a feature) + multinomial logistic regression (`class_weight=balanced`, seed 42). Trained on 2,537 videos, validated on 636: **accuracy 0.8129, macro-F1 0.7264**, 12 classes, 0 failed videos. These are post-correction metrics: the organizer confirmed 108 mislabeled `wrong_way_driving` training/val videos, which were corrected to `normal` in the manifest (88 train + 20 val) and only the linear head was retrained — CLIP was never fine-tuned. See `models/clip_linear_v1/metadata.json` and `scripts/train_clip_classifier.py`. Single-video inference: `python scripts/predict_video.py --video path/to/video.mp4`.
- **Object evidence (supporting):** pretrained YOLOv8n at ~2 fps with IoU tracking (constant-velocity prediction, conservative fragment stitching, ByteTrack optional). Detects road classes (car/truck/bus/motorcycle/bicycle/person/traffic light/stop sign) and derives an object-activity window plus pre/peak/post frames; per-frame boxes render as a live overlay in the dashboard. YOLO is evidence, not the classifier — it cannot overwrite the CLIP verdict. CLI: `python scripts/analyze_objects.py --video path/to/video.mp4`.
- **Temporal events (L2/L3 localization):** the frozen CLIP classifier is reused as a coarse-to-fine window scanner (`services/temporal_events.py`) to emit measured incident spans — no retraining, no ground-truth copying. L1 stays classification-only (null timestamps).
- **Fallback:** the deterministic motion-energy heuristic (`inference/predict.py`) covers CLIP-unavailable operation and always labels its own verdicts as heuristic.

## Dataset information

- 3,173 unique training videos, 34 test videos, 12 incident classes (per brief).
- Videos/weights/embeddings are NEVER committed (see `.gitignore`: `data/cache/`, `models/`, `uploads/`, `data/evidence/`, `*.pt`, `*.npz`).
- **Labels:** the 12 trained class names live in `models/clip_linear_v1/labels.json` (mirrored from the training manifest). `INCIDENT_CLASSES` in `backend/app/config.py` remains the scaffold placeholder and is not on the inference path.

## Current limitations

- CPU-only inference: roughly ~20 s one-time CLIP encoder load per process, then ~1–2 s CLIP + ~1–2 s YOLO per short clip warm (far slower cold or on long videos); single-worker requests serialize.
- Aggregate validation metrics only (accuracy 0.8129 / macro-F1 0.7264 in `metadata.json`, post-correction); fine-grained failure modes (e.g. smoke vs fire) were not separately quantified — demo labels are indicative.
- Test videos are unlabeled spot-checks, not a scored set.
- No auth, no DB (in-memory jobs), uploads deleted after analysis; the dashboard does not restore results on page refresh (refetch via `GET /api/results/{job_id}` works while the backend runs).
- `docker-compose.yml` predates the ML integration (backend image lacks `scripts/`, model weights, and HF cache; compose also expects a `backend/.env` file). Local virtualenvs are the supported demo path until the Docker context is reworked.

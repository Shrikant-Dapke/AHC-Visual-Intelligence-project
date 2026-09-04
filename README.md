# AHC Visual Intelligence

AI-powered road/incident video intelligence — hackathon demo.

Upload an MP4 road/traffic video → get incident class, confidence, event timestamps, timeline visualization, and a deterministic evidence-based explanation.

## Hackathon objective

Demo-ready, reliable, and deployable by tomorrow. This scaffold prioritizes a short working demo path over research-grade training: upload → OpenCV sampling → deterministic heuristic inference → dashboard.

## Architecture

```
browser (Next.js 14, TS)
  │  POST /api/analyze (multipart .mp4)
  ▼
FastAPI backend
  ├─ services/video.py    (cv2.VideoCapture probe + ~1 fps sampling, no ffmpeg)
  ├─ inference/predict.py (DeterministicHeuristic, replaceable interface)
  ├─ services/explain.py  (deterministic summary, no LLM yet)
  └─ api/routes.py        (health / analyze / results/{job_id})
```

Frontend talks to the backend only via `NEXT_PUBLIC_API_BASE_URL` (no hardcoded prod URLs).

## Repository structure

```
frontend/  # Next.js App Router dashboard
backend/   # FastAPI app, video + inference + explain services
shared/    # types.ts mirroring backend schemas
docker-compose.yml
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

Health: `http://localhost:8000/api/health`

## Frontend setup

```powershell
npm install --prefix frontend
npm run dev --prefix frontend
# open http://localhost:3000
```

Set `frontend/.env.local` from `.env.example` if the backend is not on `localhost:8000`.

## Docker setup

```powershell
docker compose up --build
# frontend http://localhost:3000, backend http://localhost:8000
```

## API endpoints

- `GET /api/health` → `{ status, model_backend, labels_configured }`
- `POST /api/analyze` (multipart, field `file`, `.mp4` only) → `AnalyzeResult`
- `GET /api/results/{job_id}` → stored `AnalyzeResult` (in-memory, no DB)

Result contract:

```json
{
  "job_id": "string",
  "incident_class": "string",
  "confidence": 0.0,
  "events": [{ "t_start": 1.2, "t_end": 3.4, "label": "string", "score": 0.8 }],
  "explanation": "string",
  "thumbnail_urls": []
}
```

## ML / inference architecture

Current: `DeterministicHeuristic` in `backend/app/inference/predict.py` — mean absolute frame-difference on 64×64 grayscale samples at ~1 fps, thresholded into temporal spans. Deterministic, offline, no weights, no randomness.

Future: `PretrainedModelInference` (YOLO/CLIP/VideoMAE) implementing the same `predict(frames, duration)` → `InferenceOutput` signature. Swap inside `get_inference()`; routes and frontend do not change.

## Dataset information

- 3,173 unique training videos, 34 test videos, 12 incident classes (per brief).
- Videos/weights are NEVER committed (see `.gitignore`).
- **Labels:** `INCIDENT_CLASSES` in `backend/app/config.py` (and `shared/types.ts`) is intentionally empty until the real `labels.txt` / `classes.txt` / dataset README is supplied. Until then inference honestly returns `incident_class: "unknown"`, confidence `0.0`. Fake class names are not generated.

## Current limitations

- Heuristic motion energy only — not a trained incident classifier.
- No thumbnails yet (`thumbnail_urls: []`), no auth, no DB (in-memory jobs), uploads deleted after analysis.
- ~1 fps sampling, 64×64 grayscale, 1200-sample cap per video.

## Future pretrained-model integration

1. Supply the 12 labels → fill `INCIDENT_CLASSES` (both config files).
2. Add model lib to `backend/requirements.txt`, implement `PretrainedModelInference`.
3. Set `MODEL_BACKEND` env and map model logits → `InferenceOutput` + keep `explain.py` factual.

/** Shared API contract. Keep in sync with backend/app/schemas.py. */

export interface EventSpan {
  t_start: number;
  t_end: number;
  label: string;
  score: number;
}

export interface AnalyzeResult {
  job_id: string;
  incident_class: string;
  confidence: number;
  events: EventSpan[];
  explanation: string;
  thumbnail_urls: string[];
  duration_sec: number;
  fps: number;
  width: number;
  height: number;
  // Task 3: unified CLIP+YOLO result (all present; arrays may be empty).
  // incident_source: "clip" | "heuristic" | "none" (YOLO never sets this).
  incident_source: string;
  timeline: ObjectTimeline | null;
  objects: ObjectCount[];
  tracks: ObjectTrack[];
  evidence: EvidenceItem[];
  warnings: string[];
}

export interface HealthResponse {
  status: string;
  service: string;
  model_backend: string;
  labels_configured: boolean;
  // Task 3: availability probes (never load weights).
  clip_available: boolean;
  clip_reason: string | null;
  yolo_available: boolean;
  yolo_reason: string | null;
}

// Source of truth for the 12 dataset labels lives in backend/app/config.py.
// DO NOT invent names here. Leave empty until labels.txt is supplied.
export const INCIDENT_CLASSES: string[] = [];

// --- Task 2: object + temporal intelligence (ADDITIVE; mirrors
// backend/app/schemas.py). -----------------------------------------------

export interface ObjectCount {
  class: string;
  count: number;
  track_ids: number[];
}

export interface TrackPoint {
  t: number;
  frame: number;
  cx: number;
  cy: number;
}

export interface ObjectTrack {
  track_id: number;
  class: string;
  first_seen: number;
  last_seen: number;
  duration: number;
  hits: number;
  displacement_px: number;
  trajectory: TrackPoint[];
}

export interface EvidenceItem {
  kind: "pre" | "peak" | "post";
  timestamp: number;
  frame_index: number;
  // Internal server path; never sent over the API (clients use `url`).
  path?: string | null;
  // Task 3: HTTP URL for the frontend (raw filesystem `path` is never
  // fetched directly).
  url: string | null;
}

export interface ObjectTimeline {
  start: number;
  peak: number;
  end: number;
}

export interface IncidentRef {
  class: string;
  confidence: number;
  source: "clip" | "passthrough" | "none";
}

export interface DetectorRef {
  name: string;
  model: string;
  conf: number;
  imgsz: number;
}

export interface ObjectAnalysisResult {
  video_id: string;
  duration_sec: number;
  fps: number;
  frames_analyzed: number;
  sample_fps: number;
  detector: DetectorRef;
  tracker: string;
  incident: IncidentRef;
  timeline: ObjectTimeline;
  objects: ObjectCount[];
  tracks: ObjectTrack[];
  evidence: EvidenceItem[];
}

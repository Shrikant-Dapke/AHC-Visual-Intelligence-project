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
}

export interface HealthResponse {
  status: string;
  service: string;
  model_backend: string;
  labels_configured: boolean;
}

// Source of truth for the 12 dataset labels lives in backend/app/config.py.
// DO NOT invent names here. Leave empty until labels.txt is supplied.
export const INCIDENT_CLASSES: string[] = [];

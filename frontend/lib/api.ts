import type { AnalyzeResult } from "../../shared/types";

export type { AnalyzeResult };

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function asJson(res: Response) {
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail =
      (data as { detail?: string })?.detail || `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return data;
}

export async function getHealth(): Promise<{ status: string; labels_configured: boolean; model_backend: string }> {
  const res = await fetch(`${BASE}/api/health`, { cache: "no-store" });
  return (await asJson(res)) as { status: string; labels_configured: boolean; model_backend: string };
}

export async function analyzeVideo(file: File): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/analyze`, { method: "POST", body: form });
  return (await asJson(res)) as AnalyzeResult;
}

export async function getResult(jobId: string): Promise<AnalyzeResult> {
  const res = await fetch(`${BASE}/api/results/${jobId}`, { cache: "no-store" });
  return (await asJson(res)) as AnalyzeResult;
}

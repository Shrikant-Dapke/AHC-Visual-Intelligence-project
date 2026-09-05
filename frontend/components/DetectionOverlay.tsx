"use client";

import { useEffect, useRef } from "react";
import type { FrameDetections } from "../../shared/types";
import { prettyClass } from "../lib/format";

/** Per-class box colors (restrained; dark-video legible, no neon). */
const CLASS_COLORS: Record<string, string> = {
  car: "#4fb3ff",
  truck: "#4fb3ff",
  bus: "#4fb3ff",
  motorcycle: "#b48cff",
  bicycle: "#b48cff",
  person: "#3ecf8e",
  "traffic light": "#f5a524",
  "stop sign": "#f5a524",
};
const FALLBACK_COLOR = "#9aa7bd";

export function boxColor(cls: string): string {
  return CLASS_COLORS[cls] ?? FALLBACK_COLOR;
}

/** Nearest observation frame at/before t; null when older than tolerance. */
export function findFrameAt(
  detections: FrameDetections[],
  t: number,
  tolerance = 0.6
): FrameDetections | null {
  let best: FrameDetections | null = null;
  for (const f of detections) {
    if (f.timestamp <= t + 1e-6 && (best === null || f.timestamp > best.timestamp)) {
      best = f;
    }
  }
  if (best === null || t - best.timestamp > tolerance) return null;
  return best;
}

export interface ContentRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Rendered-video rect inside an object-fit:contain element box. */
export function contentRect(
  elemW: number,
  elemH: number,
  videoW: number,
  videoH: number
): ContentRect {
  if (elemW <= 0 || elemH <= 0 || videoW <= 0 || videoH <= 0) {
    return { x: 0, y: 0, w: Math.max(0, elemW), h: Math.max(0, elemH) };
  }
  const scale = Math.min(elemW / videoW, elemH / videoH);
  const w = videoW * scale;
  const h = videoH * scale;
  return { x: (elemW - w) / 2, y: (elemH - h) / 2, w, h };
}

type Props = {
  detections: FrameDetections[];
  videoRef: { current: HTMLVideoElement | null };
};

/** Transparent canvas over the video rendering backend track observations.
 *  Pure visualization: reads video.currentTime via rAF (seek/pause/resume
 *  stay synchronized for free), scales normalized bboxes to the displayed
 *  video rect, and never intercepts pointer input.
 */
export default function DetectionOverlay({ detections, videoRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dataRef = useRef(detections);
  dataRef.current = detections;

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(wrap.clientWidth * dpr));
      const h = Math.max(1, Math.round(wrap.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const video = videoRef.current;
      const frames = dataRef.current;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const cw = canvas.width / dpr;
      const ch = canvas.height / dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      if (!video || frames.length === 0 || video.readyState < 2) return;
      const frame = findFrameAt(frames, video.currentTime);
      if (!frame || frame.tracks.length === 0) return;
      const rect = contentRect(
        wrap.clientWidth, wrap.clientHeight,
        video.videoWidth, video.videoHeight
      );
      const fontSize = Math.max(10, Math.min(13, Math.round(rect.w / 60)));
      ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
      for (const t of frame.tracks) {
        const [x1, y1, x2, y2] = t.bbox;
        const px = rect.x + x1 * rect.w;
        const py = rect.y + y1 * rect.h;
        const pw = Math.max(2, (x2 - x1) * rect.w);
        const ph = Math.max(2, (y2 - y1) * rect.h);
        const color = boxColor(t.class);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(px + 0.5, py + 0.5, pw, ph);
        const label = `${prettyClass(t.class)} #${t.id} ${Math.round(t.confidence * 100)}%`;
        const tw = ctx.measureText(label).width;
        const lh = fontSize + 7;
        const ly = py - lh >= rect.y ? py - lh : py;
        ctx.fillStyle = "rgba(8, 11, 16, 0.82)";
        ctx.fillRect(px, ly, tw + 10, lh);
        ctx.fillStyle = color;
        ctx.fillText(label, px + 5, ly + fontSize + 1);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={wrapRef}
      aria-hidden
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
    </div>
  );
}

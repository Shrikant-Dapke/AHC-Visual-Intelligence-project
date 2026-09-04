"use client";

import { useEffect, useRef } from "react";
import type { HistoryPoint } from "../hooks/useRealtimeVideoAnalysis";

type Props = { history: HistoryPoint[]; windowSec?: number };

/** Compact real-time signal graph drawn on canvas. No chart library. */
export default function SignalGraph({ history, windowSec = 12 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const now = history.length > 0 ? history[history.length - 1].t : windowSec;
    const t0 = Math.max(0, now - windowSec);
    const x = (t: number) => ((t - t0) / windowSec) * w;
    const y = (s: number) => h - 6 - (Math.max(0, Math.min(100, s)) / 100) * (h - 14);

    // gridlines at 50
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y(50));
    ctx.lineTo(w, y(50));
    ctx.stroke();
    // anomaly threshold 65
    ctx.strokeStyle = "rgba(242,85,85,0.45)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y(65));
    ctx.lineTo(w, y(65));
    ctx.stroke();
    ctx.setLineDash([]);

    const pts = history.filter((p) => p.t >= t0);
    if (pts.length > 1) {
      // line
      ctx.beginPath();
      pts.forEach((p, i) => {
        if (i === 0) ctx.moveTo(x(p.t), y(p.score));
        else ctx.lineTo(x(p.t), y(p.score));
      });
      ctx.strokeStyle = "#8b95a9";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // hot segments
      ctx.strokeStyle = "#f25555";
      ctx.beginPath();
      let pen = false;
      for (let i = 1; i < pts.length; i++) {
        if (pts[i].score >= 65) {
          if (!pen) {
            ctx.moveTo(x(pts[i - 1].t), y(pts[i - 1].score));
            pen = true;
          }
          ctx.lineTo(x(pts[i].t), y(pts[i].score));
        } else pen = false;
      }
      ctx.stroke();
    } else {
      ctx.fillStyle = "rgba(91,101,117,0.9)";
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText("Signal appears while the video plays", 4, h / 2);
    }
  }, [history, windowSec]);

  return (
    <div className="graph-wrap">
      <canvas ref={ref} aria-label="Live anomaly signal graph" />
      <div className="graph-cap">
        <span>-{windowSec}S</span>
        <span>VISUAL-CHANGE SCORE</span>
        <span>NOW</span>
      </div>
    </div>
  );
}

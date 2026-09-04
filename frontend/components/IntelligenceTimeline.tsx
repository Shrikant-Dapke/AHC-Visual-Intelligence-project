"use client";

import type { EventSpan } from "../../shared/types";
import type { LiveEvent } from "../hooks/useRealtimeVideoAnalysis";

type Props = {
  duration: number;
  currentTime: number;
  liveEvents: LiveEvent[];
  backendEvents: EventSpan[];
  onSeek: (t: number) => void;
};

function fmt(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(0).padStart(2, "0")}`;
}

/** Clean incident timeline: progress, live markers, backend spans. */
export default function IntelligenceTimeline({
  duration,
  currentTime,
  liveEvents,
  backendEvents,
  onSeek,
}: Props) {
  const total = duration > 0 ? duration : 1;

  function seekFromClick(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(f * total);
  }

  return (
    <div className="section">
      <h2 className="section-label">Incident timeline</h2>
      {liveEvents.length === 0 && backendEvents.length === 0 ? (
        <p className="empty-note">
          No anomalies detected
          <small>The live signal remained below the detection threshold.</small>
        </p>
      ) : (
        <>
          <div
            className="itl"
            role="slider"
            aria-label="Incident timeline"
            aria-valuenow={Math.round(currentTime)}
            aria-valuemin={0}
            aria-valuemax={Math.round(total)}
            tabIndex={0}
            onClick={seekFromClick}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft") onSeek(Math.max(0, currentTime - 1));
              if (e.key === "ArrowRight") onSeek(Math.min(total, currentTime + 1));
            }}
          >
            {[0.25, 0.5, 0.75].map((f) => (
              <div key={f} className="itl-grid" style={{ left: `${f * 100}%` }} />
            ))}
            <div
              className="itl-progress"
              style={{ width: `${(Math.min(currentTime, total) / total) * 100}%` }}
            />
            {backendEvents.map((ev, i) => (
              <button
                key={`b${i}`}
                className="itl-ev backend"
                style={{
                  left: `${(ev.t_start / total) * 100}%`,
                  width: `${Math.max(1, ((ev.t_end - ev.t_start) / total) * 100)}%`,
                }}
                title={`${ev.label} ${ev.t_start.toFixed(1)}s–${ev.t_end.toFixed(1)}s`}
                onClick={(e) => {
                  e.stopPropagation();
                  onSeek(ev.t_start);
                }}
              />
            ))}
            {liveEvents.map((ev) => (
              <button
                key={`l${ev.id}`}
                className="itl-ev"
                style={{
                  left: `${(ev.tStart / total) * 100}%`,
                  width: `${Math.max(1, (((ev.tEnd ?? currentTime) - ev.tStart) / total) * 100)}%`,
                }}
                title={`${ev.label} ${ev.tStart.toFixed(1)}s${ev.tEnd ? `–${ev.tEnd.toFixed(1)}s` : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onSeek(ev.tStart);
                }}
              />
            ))}
            <div
              className="itl-playhead"
              style={{ left: `${(Math.min(currentTime, total) / total) * 100}%` }}
            />
          </div>
          <div className="itl-scale">
            <span>{fmt(0)}</span>
            <span>{fmt(total / 2)}</span>
            <span>{fmt(total)}</span>
          </div>
        </>
      )}
    </div>
  );
}

"use client";

import { useState } from "react";
import type { EvidenceItem } from "../../shared/types";
import { evidenceSrc } from "../lib/api";
import { fmtTime } from "./VideoPlayer";
import type { LiveEvent } from "../hooks/useRealtimeVideoAnalysis";

type Props = {
  liveEvents: LiveEvent[];
  backendUrls: string[];
  evidence?: EvidenceItem[];
  onSeek: (t: number) => void;
};

const KIND_LABEL: Record<string, string> = {
  pre: "Pre-incident",
  peak: "Peak",
  post: "Post-incident",
};

/** Evidence gallery: backend pre/peak/post frames (clickable to seek). */
export default function EvidenceStrip({ liveEvents, backendUrls, evidence = [], onSeek }: Props) {
  const [failed, setFailed] = useState<Record<string, boolean>>({});
  const withThumb = liveEvents.filter((e) => e.thumbnail);
  // Prefer structured evidence (has timestamps); fall back to legacy URLs.
  const gallery =
    evidence.length > 0
      ? evidence.map((e) => ({
          key: e.kind,
          src: evidenceSrc(e.url),
          label: KIND_LABEL[e.kind] ?? e.kind,
          t: e.timestamp,
          peak: e.kind === "peak",
        }))
      : backendUrls.map((u, i) => ({
          key: `b${i}`,
          src: evidenceSrc(u),
          label: `Frame ${i + 1}`,
          t: null as number | null,
          peak: false,
        }));
  const visible = gallery.filter((g) => g.src && !failed[g.key]);

  if (withThumb.length === 0 && visible.length === 0) {
    return (
      <div className="section">
        <h2 className="section-label">Evidence</h2>
        <p className="empty-note">
          No evidence yet
          <small>Frames are captured automatically when an anomaly is detected.</small>
        </p>
      </div>
    );
  }
  return (
    <div className="section">
      <h2 className="section-label">Evidence</h2>
      <div className="ev-grid">
        {visible.map((g) => (
          <button
            key={g.key}
            className={`ev-item${g.peak ? " peak" : ""}`}
            onClick={() => {
              if (g.t !== null) onSeek(g.t);
            }}
            title={g.t !== null ? `Seek to ${fmtTime(g.t)}` : g.label}
            disabled={g.t === null}
            style={g.t === null ? { cursor: "default" } : undefined}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={g.src}
              alt={`${g.label}${g.t !== null ? ` at ${fmtTime(g.t)}` : ""}`}
              onError={() => setFailed((f) => ({ ...f, [g.key]: true }))}
            />
            <div className="cap">
              <strong>{g.label}</strong>
              {g.t !== null && <span>{fmtTime(g.t)}</span>}
            </div>
          </button>
        ))}
        {withThumb.map((e) => (
          <button
            key={e.id}
            className="ev-item"
            onClick={() => onSeek(e.tStart)}
            title={`Seek to ${e.tStart.toFixed(1)}s`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={e.thumbnail} alt={`Evidence at ${e.tStart.toFixed(1)} seconds`} />
            <div className="cap">
              <strong>{e.label.toLowerCase()}</strong>
              <span>{e.tStart.toFixed(1)}s</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

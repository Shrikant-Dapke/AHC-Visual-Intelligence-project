"use client";

import type { LiveEvent } from "../hooks/useRealtimeVideoAnalysis";

type Props = {
  liveEvents: LiveEvent[];
  backendUrls: string[];
  onSeek: (t: number) => void;
};

/** Evidence: real captured frames, clickable to seek. */
export default function EvidenceStrip({ liveEvents, backendUrls, onSeek }: Props) {
  const withThumb = liveEvents.filter((e) => e.thumbnail);
  if (withThumb.length === 0 && backendUrls.length === 0) {
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
              {e.tStart.toFixed(1)}s · {e.label.toLowerCase()}
            </div>
          </button>
        ))}
        {backendUrls.map((u, i) => (
          <div key={`b${i}`} className="ev-item" style={{ cursor: "default" }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={u} alt={`Backend evidence ${i + 1}`} />
            <div className="cap">Backend frame</div>
          </div>
        ))}
      </div>
    </div>
  );
}

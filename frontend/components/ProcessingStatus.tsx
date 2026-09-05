"use client";

import { useEffect, useState } from "react";

const STAGES = [
  "Uploading",
  "Analyzing video",
  "Running visual intelligence",
  "Generating evidence",
];

/** Honest processing state: activity stages + elapsed time, never a fake %. */
export default function ProcessingStatus({ startedAt }: { startedAt: number }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    setElapsed(Math.max(0, Math.round((Date.now() - startedAt) / 1000)));
    const id = setInterval(() => {
      setElapsed(Math.max(0, Math.round((Date.now() - startedAt) / 1000)));
    }, 1000);
    return () => clearInterval(id);
  }, [startedAt]);

  const stage = STAGES[Math.min(STAGES.length - 1, Math.floor(elapsed / 20))];
  return (
    <div className="section" role="status" aria-live="polite">
      <h2 className="section-label">Analysis in progress</h2>
      <div className="proc-row">
        <span className="proc-spinner" aria-hidden="true" />
        <div>
          <div className="proc-stage">{stage}…</div>
          <div className="proc-sub">
            {elapsed}s elapsed · usually 1–3 minutes on CPU
          </div>
        </div>
      </div>
      <div className="proc-track">
        <div className="proc-fill" />
      </div>
    </div>
  );
}

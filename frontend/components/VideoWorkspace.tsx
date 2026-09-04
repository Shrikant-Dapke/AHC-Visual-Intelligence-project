"use client";

import { useRef, useState, type RefObject } from "react";
import VideoPlayer from "./VideoPlayer";
import { fmtSize } from "../lib/format";
import type { LiveSnapshot } from "../hooks/useRealtimeVideoAnalysis";

export type FeedPhase = "IDLE" | "VIDEO_READY" | "ANALYZING" | "ANALYSIS_COMPLETE" | "ERROR";

type Props = {
  phase: FeedPhase;
  file: File | null;
  previewUrl: string | null;
  error: string;
  videoRef: RefObject<HTMLVideoElement | null>;
  snapshot: LiveSnapshot;
  analyzing: boolean;
  anomaly: boolean;
  paused: boolean;
  ended: boolean;
  caption: string;
  onFile: (f: File) => void;
  onStart: () => void;
  onStop: () => void;
  onReplay: () => void;
  onReset: () => void;
  onTick: (t: number, dur: number) => void;
};

export default function VideoWorkspace(props: Props) {
  const {
    phase, file, previewUrl, error, videoRef,
    analyzing, anomaly, paused, ended, caption,
    onFile, onStart, onStop, onReplay, onReset, onTick,
  } = props;
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  function pick(f: File | undefined) {
    if (f) onFile(f);
  }

  if (phase === "IDLE" || phase === "ERROR") {
    return (
      <div>
        <div
          className={`hero${drag ? " drag" : ""}`}
          role="button"
          tabIndex={0}
          aria-label="Select MP4 video"
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            pick(e.dataTransfer.files?.[0]);
          }}
        >
          <h1>Analyze a road video</h1>
          <p>Drop your MP4 here, or choose a file. Maximum 200 MB.</p>
          <button className="btn">Choose video</button>
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,.mp4"
            hidden
            onChange={(e) => {
              pick(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>
        {error && <div className="error-box">{error}</div>}
      </div>
    );
  }

  let badge: { cls: string; text: string } = { cls: "idle", text: "Ready" };
  if (anomaly && analyzing && !paused) badge = { cls: "anomaly", text: "● Anomaly detected" };
  else if (analyzing && !paused && !ended) badge = { cls: "analyzing", text: "● Analyzing" };
  else if (ended || phase === "ANALYSIS_COMPLETE") badge = { cls: "done", text: "✓ Analysis complete" };
  else if (paused && analyzing) badge = { cls: "idle", text: "❚❚ Paused" };

  return (
    <div>
      <div className="video-frame">
        {previewUrl && (
          <VideoPlayer
            src={previewUrl}
            videoRef={videoRef}
            onTick={(t) => onTick(t, videoRef.current?.duration ?? 0)}
            onMeta={() => undefined}
          />
        )}
        <div className={`video-badge ${badge.cls}`}>{badge.text}</div>
      </div>

      <div className="caption-line">
        <strong>{file?.name}</strong> · {caption}
        {file && ` · ${fmtSize(file.size)}`}
      </div>

      <div className="control-row">
        {phase === "VIDEO_READY" && (
          <button className="btn" onClick={onStart} disabled={!file}>
            Analyze video
          </button>
        )}
        {phase === "ANALYZING" && (
          <button className="btn" onClick={onStop}>
            Stop analysis
          </button>
        )}
        {phase === "ANALYSIS_COMPLETE" && (
          <button className="btn" onClick={onReplay}>
            Replay analysis
          </button>
        )}
        <button className="btn secondary" onClick={onReset}>
          {phase === "ANALYSIS_COMPLETE" ? "New video" : "Clear"}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}
    </div>
  );
}

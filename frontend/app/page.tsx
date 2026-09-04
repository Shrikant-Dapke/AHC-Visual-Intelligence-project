"use client";

import { useCallback, useEffect, useState } from "react";
import AppShell from "../components/AppShell";
import VideoWorkspace, { type FeedPhase } from "../components/VideoWorkspace";
import LiveIntelligence from "../components/LiveIntelligence";
import IntelligenceTimeline from "../components/IntelligenceTimeline";
import EventLog from "../components/EventLog";
import EvidenceStrip from "../components/EvidenceStrip";
import AnalysisResult from "../components/AnalysisResult";
import { isMp4 } from "../lib/format";
import { useRealtimeVideoAnalysis } from "../hooks/useRealtimeVideoAnalysis";
import { analyzeVideo, getHealth, type AnalyzeResult as BackendResult } from "../lib/api";

export default function Home() {
  const [phase, setPhase] = useState<FeedPhase>("IDLE");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [videoRes, setVideoRes] = useState("");
  const [paused, setPaused] = useState(true);
  const [ended, setEnded] = useState(false);

  const [health, setHealth] = useState<"ok" | "down" | "checking">("checking");
  const [labelsConfigured, setLabelsConfigured] = useState<boolean | null>(null);
  const [modelBackend, setModelBackend] = useState("heuristic");
  const [backendResult, setBackendResult] = useState<BackendResult | null>(null);
  const [backendPending, setBackendPending] = useState(false);

  const live = useRealtimeVideoAnalysis();
  const analyzing = phase === "ANALYZING";
  const anomaly = analyzing && live.snapshot.state === "ANOMALOUS";

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h.status === "ok" ? "ok" : "down");
        setLabelsConfigured(h.labels_configured);
        setModelBackend(h.model_backend || "heuristic");
      })
      .catch(() => setHealth("down"));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    const v = live.videoRef.current;
    if (!v) return;
    const onPlay = () => {
      setPaused(false);
      setEnded(false);
    };
    const onPause = () => setPaused(true);
    const onEnded = () => {
      setEnded(true);
      live.stop();
      setPhase((p) => (p === "ANALYZING" ? "ANALYSIS_COMPLETE" : p));
    };
    const onMeta = () => {
      setVideoDuration(v.duration || 0);
      setVideoRes(v.videoWidth && v.videoHeight ? `${v.videoWidth}×${v.videoHeight}` : "");
    };
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("ended", onEnded);
    v.addEventListener("loadedmetadata", onMeta);
    return () => {
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("ended", onEnded);
      v.removeEventListener("loadedmetadata", onMeta);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewUrl]);

  const handleFile = useCallback((f: File) => {
    if (!isMp4(f)) {
      setError("Only MP4 videos are supported.");
      setPhase("ERROR");
      return;
    }
    setError("");
    setBackendResult(null);
    setBackendPending(false);
    setCurrentTime(0);
    setVideoDuration(0);
    setVideoRes("");
    setPaused(true);
    setEnded(false);
    live.reset();
    setFile(f);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
    setPhase("VIDEO_READY");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runBackend = useCallback((f: File) => {
    setBackendPending(true);
    analyzeVideo(f)
      .then(setBackendResult)
      .catch((e) =>
        setError((prev) =>
          prev || (e instanceof Error ? `Backend: ${e.message}` : "Backend analysis failed.")
        )
      )
      .finally(() => setBackendPending(false));
  }, []);

  const handleStart = useCallback(() => {
    const v = live.videoRef.current;
    if (!v || !file) {
      setError("Load a video first.");
      return;
    }
    setError("");
    setEnded(false);
    setPhase("ANALYZING");
    try {
      v.currentTime = 0;
    } catch {
      /* noop */
    }
    live.start(v);
    v.play().catch(() => undefined);
    runBackend(file);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  const handleStop = useCallback(() => {
    live.stop();
    live.videoRef.current?.pause();
    setPhase("ANALYSIS_COMPLETE");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleReplay = useCallback(() => {
    const v = live.videoRef.current;
    if (!v) return;
    setError("");
    setEnded(false);
    setPhase("ANALYZING");
    try {
      v.currentTime = 0;
    } catch {
      /* noop */
    }
    live.start(v);
    v.play().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleReset = useCallback(() => {
    live.reset();
    setFile(null);
    setBackendResult(null);
    setBackendPending(false);
    setError("");
    setCurrentTime(0);
    setVideoDuration(0);
    setVideoRes("");
    setPaused(true);
    setEnded(false);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setPhase("IDLE");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSeek = useCallback((t: number) => {
    const v = live.videoRef.current;
    if (v) {
      try {
        v.currentTime = t;
      } catch {
        /* noop */
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleTick = useCallback((t: number, dur: number) => {
    setCurrentTime(t);
    if (dur > 0) setVideoDuration(dur);
  }, []);

  const duration = backendResult?.duration_sec || videoDuration || 0;
  const caption =
    duration > 0
      ? `${duration.toFixed(1)} seconds${videoRes ? ` · ${videoRes}` : ""}`
      : "Loading…";
  const showWorkstation = phase !== "IDLE" && phase !== "ERROR";

  return (
    <AppShell
      health={health}
      labelsConfigured={labelsConfigured}
      modelBackend={modelBackend}
      liveActive={analyzing}
    >
      <main className="work">
        <div className="col">
          <VideoWorkspace
            phase={phase}
            file={file}
            previewUrl={previewUrl}
            error={error}
            videoRef={live.videoRef}
            snapshot={live.snapshot}
            analyzing={analyzing}
            anomaly={anomaly}
            paused={paused}
            ended={ended}
            caption={caption}
            onFile={handleFile}
            onStart={handleStart}
            onStop={handleStop}
            onReplay={handleReplay}
            onReset={handleReset}
            onTick={handleTick}
          />
          {showWorkstation && (
            <IntelligenceTimeline
              duration={duration}
              currentTime={currentTime}
              liveEvents={live.events}
              backendEvents={backendResult?.events ?? []}
              onSeek={handleSeek}
            />
          )}
        </div>

        <div className="col">
          {showWorkstation ? (
            <>
              <LiveIntelligence
                snapshot={live.snapshot}
                history={live.history}
                started={analyzing || phase === "ANALYSIS_COMPLETE"}
              />
              <EventLog log={live.log} onSeek={handleSeek} />
            </>
          ) : (
            <div className="section">
              <h2 className="section-label">How it works</h2>
              <p className="empty-note">
                Upload a video to watch live anomaly detection.
                <small>Motion signal, timeline, evidence and classification appear here.</small>
              </p>
            </div>
          )}
        </div>
      </main>

      {(showWorkstation || backendResult) && (
        <div className="work" style={{ paddingTop: 0 }}>
          <div className="col">
            <EvidenceStrip
              liveEvents={live.events}
              backendUrls={backendResult?.thumbnail_urls ?? []}
              onSeek={handleSeek}
            />
          </div>
          <div className="col">
            <AnalysisResult
              result={backendResult}
              pending={backendPending}
              labelsConfigured={labelsConfigured}
            />
            {live.snapshot.samples > 0 && (analyzing || phase === "ANALYSIS_COMPLETE") && (
              <details className="tech-details">
                <summary>Technical details</summary>
                <dl>
                  <dt>Samples</dt>
                  <dd>{live.snapshot.samples}</dd>
                  <dt>Scan rate</dt>
                  <dd>{live.snapshot.scanFps.toFixed(1)} fps</dd>
                  <dt>Signal confidence</dt>
                  <dd>{live.snapshot.confidence}/100</dd>
                  <dt>Sampler</dt>
                  <dd>rVFC + interval fallback</dd>
                </dl>
              </details>
            )}
          </div>
        </div>
      )}
    </AppShell>
  );
}

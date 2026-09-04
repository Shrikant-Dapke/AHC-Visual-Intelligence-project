"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  SignalProcessor,
  frameChangeScore,
  type LiveEvent,
  type LogEntry,
  type SignalSample,
  type SignalState,
} from "../lib/realtime";

export type { LiveEvent, LogEntry, SignalSample, SignalState };

export interface LiveSnapshot {
  score: number;
  state: SignalState;
  motion: number; // 0..100 smoothed visual change
  changeRate: number; // 0..100 instantaneous visual change
  baseline: number; // 0..100 ambient baseline
  confidence: number; // 0..100 sample-coverage confidence
  samples: number;
  frame: number;
  scanFps: number;
}

export interface HistoryPoint {
  t: number;
  score: number;
}

const ANALYSIS_WIDTH = 192;
const EVIDENCE_WIDTH = 320;
const MIN_SAMPLE_INTERVAL_MS = 120;
const HISTORY_CAP = 200;

const IDLE_SNAPSHOT: LiveSnapshot = {
  score: 0,
  state: "NORMAL",
  motion: 0,
  changeRate: 0,
  baseline: 0,
  confidence: 0,
  samples: 0,
  frame: 0,
  scanFps: 0,
};

type VideoLike = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: () => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

/**
 * Owns browser-side real-time visual analysis for one video element.
 * Frame difference -> rolling smooth -> adaptive baseline -> anomaly score.
 * All scores derive from actual video pixels; no randomness, no hardcoding.
 */
export function useRealtimeVideoAnalysis() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const evidenceRef = useRef<HTMLCanvasElement | null>(null);
  const processorRef = useRef(new SignalProcessor());
  const prevRef = useRef<Uint8Array | null>(null);
  const runningRef = useRef(false);
  const rafHandleRef = useRef<number>(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastSampleWallRef = useRef(0);
  const lastTickRef = useRef(0);
  const frameCountRef = useRef(0);
  const openedAtRef = useRef(0);

  const [active, setActive] = useState(false);
  const [snapshot, setSnapshot] = useState<LiveSnapshot>(IDLE_SNAPSHOT);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);

  const ensureCanvases = useCallback((video: HTMLVideoElement) => {
    const vw = video.videoWidth || 16;
    const vh = video.videoHeight || 9;
    if (!canvasRef.current) canvasRef.current = document.createElement("canvas");
    const scale = ANALYSIS_WIDTH / vw;
    canvasRef.current.width = ANALYSIS_WIDTH;
    canvasRef.current.height = Math.max(1, Math.round(vh * scale));
    if (!evidenceRef.current) evidenceRef.current = document.createElement("canvas");
    const escale = EVIDENCE_WIDTH / vw;
    evidenceRef.current.width = EVIDENCE_WIDTH;
    evidenceRef.current.height = Math.max(1, Math.round(vh * escale));
  }, []);

  const captureEvidence = useCallback((video: HTMLVideoElement): string | undefined => {
    try {
      const c = evidenceRef.current;
      if (!c) return undefined;
      const ctx = c.getContext("2d");
      if (!ctx) return undefined;
      ctx.drawImage(video, 0, 0, c.width, c.height);
      return c.toDataURL("image/jpeg", 0.6);
    } catch {
      return undefined;
    }
  }, []);

  const sample = useCallback(() => {
    const video = videoRef.current;
    if (!video || !runningRef.current) return;
    if (video.paused || video.ended || video.readyState < 2) return;
    const nowWall = performance.now();
    if (nowWall - lastSampleWallRef.current < MIN_SAMPLE_INTERVAL_MS) return;
    lastSampleWallRef.current = nowWall;

    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    let gray: Uint8Array;
    try {
      ctx.drawImage(video, 0, 0, c.width, c.height);
      const data = ctx.getImageData(0, 0, c.width, c.height).data;
      gray = new Uint8Array(c.width * c.height);
      for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        gray[j] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
      }
    } catch {
      return; // canvas tainted or unavailable; skip sample honestly
    }

    frameCountRef.current += 1;
    const prev = prevRef.current;
    prevRef.current = gray;
    if (!prev || prev.length !== gray.length) return; // need a pair

    const raw = frameChangeScore(prev, gray);
    const t = video.currentTime;
    const proc = processorRef.current;
    const beforeOpen = proc.openEvent?.id ?? null;
    const beforeLog = proc.log.length;
    const s: SignalSample = proc.push(t, raw);

    // Attach real evidence thumbnail when a new event opens.
    if (proc.openEvent && proc.openEvent.id !== beforeOpen) {
      const thumb = captureEvidence(video);
      if (thumb) proc.openEvent.thumbnail = thumb;
      setEvents([...proc.events]);
    } else if (proc.events.length !== openedAtRef.current) {
      openedAtRef.current = proc.events.length;
      setEvents([...proc.events]);
    }
    if (proc.log.length !== beforeLog) {
      setLog([...proc.log]);
      setEvents([...proc.events]);
    }

    const wallElapsed = Math.max(0.001, (nowWall - (lastTickRef.current || nowWall)) / 1000);
    lastTickRef.current = nowWall;
    const scanFps = Math.round((1 / wallElapsed) * 10) / 10;

    setSnapshot({
      score: s.score,
      state: s.state,
      motion: Math.round(s.smooth * 1000) / 10,
      changeRate: Math.round(s.raw * 1000) / 10,
      baseline: Math.round(s.baseline * 1000) / 10,
      confidence: Math.min(100, Math.round((proc.samples / 40) * 100)),
      samples: proc.samples,
      frame: frameCountRef.current,
      scanFps,
    });
    setHistory((h) => {
      const next = [...h, { t, score: s.score }];
      return next.length > HISTORY_CAP ? next.slice(next.length - HISTORY_CAP) : next;
    });
  }, [captureEvidence]);

  const loopRvfc = useCallback(() => {
    if (!runningRef.current) return;
    sample();
    const video = videoRef.current as VideoLike | null;
    if (video && typeof video.requestVideoFrameCallback === "function") {
      rafHandleRef.current = video.requestVideoFrameCallback(loopRvfc);
    }
  }, [sample]);

  const start = useCallback(
    (video: HTMLVideoElement) => {
      videoRef.current = video;
      processorRef.current.reset();
      prevRef.current = null;
      frameCountRef.current = 0;
      openedAtRef.current = 0;
      lastSampleWallRef.current = 0;
      lastTickRef.current = performance.now();
      setSnapshot(IDLE_SNAPSHOT);
      setHistory([]);
      setEvents([]);
      setLog([]);
      ensureCanvases(video);
      runningRef.current = true;
      setActive(true);
      const v = video as VideoLike;
      if (typeof v.requestVideoFrameCallback === "function") {
        rafHandleRef.current = v.requestVideoFrameCallback(loopRvfc);
      } else if (!intervalRef.current) {
        intervalRef.current = setInterval(sample, MIN_SAMPLE_INTERVAL_MS);
      }
    },
    [ensureCanvases, loopRvfc, sample]
  );

  const stop = useCallback(() => {
    runningRef.current = false;
    setActive(false);
    const video = videoRef.current as VideoLike | null;
    if (video && typeof video.cancelVideoFrameCallback === "function" && rafHandleRef.current) {
      try {
        video.cancelVideoFrameCallback(rafHandleRef.current);
      } catch {
        /* noop */
      }
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    stop();
    processorRef.current.reset();
    prevRef.current = null;
    setSnapshot(IDLE_SNAPSHOT);
    setHistory([]);
    setEvents([]);
    setLog([]);
  }, [stop]);

  useEffect(() => {
    return () => {
      runningRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return { videoRef, active, snapshot, history, events, log, start, stop, reset };
}

/** Pure, deterministic real-time visual-signal math. No DOM, no randomness.
 *  Tested at runtime via realtime.test.mjs (node type-stripping).
 */

export type SignalState = "NORMAL" | "ELEVATED" | "ANOMALOUS";

export interface SignalSample {
  t: number;
  raw: number; // 0..1 instantaneous mean frame difference
  smooth: number; // 0..1 rolling mean of raw
  baseline: number; // 0..1 slow ambient-motion baseline
  score: number; // 0..100 anomaly score
  state: SignalState;
}

export interface LiveEvent {
  id: number;
  tStart: number;
  tEnd: number | null;
  peak: number;
  peakT: number;
  openScore: number;
  label: string;
  thumbnail?: string;
}

export interface LogEntry {
  t: number;
  kind: "ANOMALY" | "PEAK" | "RESOLVED";
  text: string;
  score: number;
}

export interface ProcessorOptions {
  windowSize?: number; // rolling smoothing window (samples)
  baselineAlpha?: number; // EMA rate for ambient baseline
  gain?: number; // deviation -> score gain
  openThreshold?: number; // score to start an event run
  closeThreshold?: number; // score below which a run cools down
  openSustain?: number; // consecutive samples >= openThreshold to open
  closeSustain?: number; // consecutive samples < closeThreshold to close
  peakLogAt?: number; // score that emits a one-time PEAK log per event
}

const DEFAULTS: Required<ProcessorOptions> = {
  windowSize: 8,
  baselineAlpha: 0.05,
  gain: 6,
  openThreshold: 60,
  closeThreshold: 40,
  openSustain: 3,
  closeSustain: 3,
  peakLogAt: 85,
};

/** Mean absolute difference of two equal-length grayscale buffers, 0..1. */
export function frameChangeScore(prev: ArrayLike<number>, cur: ArrayLike<number>): number {
  if (prev.length !== cur.length || prev.length === 0) return 0;
  let acc = 0;
  for (let i = 0; i < prev.length; i++) {
    const d = (cur[i] as number) - (prev[i] as number);
    acc += d < 0 ? -d : d;
  }
  return acc / prev.length / 255;
}

export function classifyScore(score: number): SignalState {
  if (score >= 65) return "ANOMALOUS";
  if (score >= 35) return "ELEVATED";
  return "NORMAL";
}

export function provisionalLabel(peak: number): string {
  return peak >= 80 ? "HIGH MOTION" : "VISUAL ANOMALY";
}

export class SignalProcessor {
  private opts: Required<ProcessorOptions>;
  private window: number[] = [];
  private baseline: number | null = null;
  private aboveRun = 0;
  private belowRun = 0;
  private runStartT = 0;
  private open: LiveEvent | null = null;
  private peakLogged = false;
  private nextId = 1;
  samples = 0;
  events: LiveEvent[] = [];
  log: LogEntry[] = [];

  constructor(opts: ProcessorOptions = {}) {
    this.opts = { ...DEFAULTS, ...opts };
  }

  reset(): void {
    this.window = [];
    this.baseline = null;
    this.aboveRun = 0;
    this.belowRun = 0;
    this.open = null;
    this.peakLogged = false;
    this.samples = 0;
    this.events = [];
    this.log = [];
  }

  push(t: number, raw: number): SignalSample {
    const o = this.opts;
    this.samples += 1;
    this.window.push(raw);
    if (this.window.length > o.windowSize) this.window.shift();
    const smooth = this.window.reduce((a, b) => a + b, 0) / this.window.length;

    if (this.baseline === null) this.baseline = smooth;
    else this.baseline += o.baselineAlpha * (smooth - this.baseline);

    const deviation = Math.max(0, smooth - (this.baseline as number));
    const score = Math.round(Math.min(1, deviation * o.gain) * 100);
    const state = classifyScore(score);

    this.trackEvents(t, score);
    return { t, raw, smooth, baseline: this.baseline as number, score, state };
  }

  private trackEvents(t: number, score: number): void {
    const o = this.opts;
    if (score >= o.openThreshold) {
      this.aboveRun += 1;
      this.belowRun = 0;
      if (this.aboveRun === 1) this.runStartT = t;
      if (!this.open && this.aboveRun >= o.openSustain) {
        this.open = {
          id: this.nextId++,
          tStart: this.runStartT,
          tEnd: null,
          peak: score,
          peakT: t,
          openScore: score,
          label: provisionalLabel(score),
        };
        this.peakLogged = score >= o.peakLogAt;
        this.events = [...this.events, this.open];
        this.log = [
          ...this.log,
          { t, kind: "ANOMALY", text: `${this.open.label} DETECTED`, score },
        ];
        if (this.peakLogged) {
          this.log = [...this.log, { t, kind: "PEAK", text: "SIGNAL PEAK", score }];
        }
      } else if (this.open) {
        if (score > this.open.peak) {
          this.open.peak = score;
          this.open.peakT = t;
          this.open.label = provisionalLabel(score);
        }
        if (!this.peakLogged && score >= o.peakLogAt) {
          this.peakLogged = true;
          this.log = [...this.log, { t, kind: "PEAK", text: "SIGNAL PEAK", score }];
        }
      }
    } else if (score < o.closeThreshold) {
      this.belowRun += 1;
      this.aboveRun = 0;
      if (this.open && this.belowRun >= o.closeSustain) {
        this.open.tEnd = t;
        this.open.label = provisionalLabel(this.open.peak);
        this.log = [
          ...this.log,
          { t, kind: "RESOLVED", text: "SIGNAL NORMALIZED", score },
        ];
        this.open = null;
      }
    } else {
      // Hysteresis band: hold current run counters.
      if (!this.open) this.aboveRun = 0;
      this.belowRun = 0;
    }
  }

  get openEvent(): LiveEvent | null {
    return this.open;
  }
}

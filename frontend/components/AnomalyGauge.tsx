type Props = { value: number; state: "NORMAL" | "ELEVATED" | "ANOMALOUS" };

const cls = { NORMAL: "st-normal", ELEVATED: "st-elevated", ANOMALOUS: "st-anomalous" } as const;

/** Quiet 0–100 gauge. Green at rest, amber/red by state. */
export default function AnomalyGauge({ value, state }: Props) {
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div className={`gauge ${cls[state]}`} role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div className="gauge-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

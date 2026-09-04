type Props = { value: number };

export default function ConfidenceMeter({ value }: Props) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  const band = pct >= 70 ? "high" : pct >= 40 ? "mid" : "";
  return (
    <div>
      <div className="conf-big">
        {(value * 100).toFixed(1)}<small>%</small>
      </div>
      <div className={`meter ${band}`} role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

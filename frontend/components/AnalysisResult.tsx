import type { AnalyzeResult } from "../lib/api";

export function isUnclassified(result: AnalyzeResult): boolean {
  return result.incident_class === "unknown" && result.confidence === 0;
}

/** Backend classification, visually separate from live anomaly detection. */
export default function AnalysisResult({
  result,
  pending,
}: {
  result: AnalyzeResult | null;
  pending: boolean;
  labelsConfigured: boolean | null;
}) {
  return (
    <div className="section">
      <h2 className="section-label">Incident classification</h2>
      {pending && !result && <p className="empty-note">Classifying video…</p>}
      {!pending && !result && (
        <p className="empty-note">
          Not classified yet
          <small>Start analysis to classify this video.</small>
        </p>
      )}
      {result && isUnclassified(result) && (
        <p className="empty-note">
          Classification unavailable
          <small>Incident taxonomy has not been configured yet.</small>
        </p>
      )}
      {result && !isUnclassified(result) && (
        <>
          <p className="class-result">{result.incident_class}</p>
          <div className="class-conf">{Math.round(result.confidence * 100)}% confidence</div>
          {result.events.length > 0 && (
            <div className="class-conf">
              Detected around{" "}
              {result.events
                .slice(0, 2)
                .map((e) => `${e.t_start.toFixed(0)}–${e.t_end.toFixed(0)}s`)
                .join(", ")}
            </div>
          )}
          <p className="explain">{result.explanation}</p>
        </>
      )}
      {result && (
        <details className="tech-details">
          <summary>Technical details</summary>
          <dl>
            <dt>Job</dt>
            <dd>{result.job_id}</dd>
            <dt>Duration</dt>
            <dd>{result.duration_sec.toFixed(1)}s</dd>
            <dt>Source fps</dt>
            <dd>{result.fps.toFixed(1)}</dd>
            <dt>Method</dt>
            <dd>Backend motion heuristic</dd>
          </dl>
        </details>
      )}
    </div>
  );
}

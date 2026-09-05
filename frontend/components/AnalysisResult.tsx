import type { AnalyzeResult } from "../lib/api";
import { objectLabel, prettyClass, sourceLabel } from "../lib/format";
import ConfidenceMeter from "./ConfidenceMeter";

export function isUnclassified(result: AnalyzeResult): boolean {
  return result.incident_class === "unknown" && result.confidence === 0;
}

const VEHICLE_CLASSES = new Set(["car", "truck", "bus", "motorcycle", "bicycle"]);

function trackNoun(classes: string[]): string {
  if (classes.length === 0) return "object";
  if (classes.every((c) => c === "person")) return "person";
  if (classes.every((c) => VEHICLE_CLASSES.has(c))) return "vehicle";
  return "object";
}

/** Incident hero: class + confidence + honest source attribution. */
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
      {result && (
        <>
          {result.warnings.length > 0 && (
            <div className="warn-box" role="status">
              {result.warnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}
          {isUnclassified(result) ? (
            <p className="empty-note">
              No incident classified
              <small>
                {result.incident_source === "none"
                  ? "The classifier reported no incident for this video."
                  : "Incident taxonomy has not been configured yet."}
              </small>
            </p>
          ) : (
            <>
              <span className={`src-badge src-${result.incident_source}`}>
                {sourceLabel(result.incident_source)}
              </span>
              <p className="class-result">{prettyClass(result.incident_class)}</p>
              <div className="hero-conf">
                <ConfidenceMeter value={result.confidence} />
              </div>
              <p className="derived">
                {sourceLabel(result.incident_source)} classified the event as{" "}
                {prettyClass(result.incident_class).toLowerCase()} with{" "}
                {(result.confidence * 100).toFixed(1)}% confidence.
                {result.objects.length > 0 && (
                  <>
                    {" "}Object evidence:{" "}
                    {result.objects.reduce((n, o) => n + o.count, 0)}{" "}
                    {trackNoun(result.objects.map((o) => o.class))} track
                    {result.objects.reduce((n, o) => n + o.count, 0) === 1 ? "" : "s"}{" "}
                    observed during the analyzed window.
                  </>
                )}
              </p>
              {result.events.length > 0 && (
                <div className="class-conf">
                  Detected around{" "}
                  {result.events
                    .slice(0, 2)
                    .map((e) => `${e.t_start.toFixed(0)}–${e.t_end.toFixed(0)}s`)
                    .join(", ")}
                </div>
              )}
            </>
          )}
          <p className="explain">{result.explanation}</p>
          <details className="tech-details">
            <summary>Technical details</summary>
            <dl>
              <dt>Job</dt>
              <dd>{result.job_id}</dd>
              <dt>Source</dt>
              <dd>{sourceLabel(result.incident_source)}</dd>
              <dt>Duration</dt>
              <dd>{result.duration_sec.toFixed(1)}s</dd>
              <dt>Source fps</dt>
              <dd>{result.fps.toFixed(1)}</dd>
              {result.timeline && (
                <>
                  <dt>Window</dt>
                  <dd>
                    {result.timeline.start.toFixed(1)}–{result.timeline.end.toFixed(1)}s
                    (peak {result.timeline.peak.toFixed(1)}s)
                  </dd>
                </>
              )}
              {result.objects.length > 0 && (
                <>
                  <dt>Objects</dt>
                  <dd>
                    {result.objects.map((o) => `${o.count} ${objectLabel(o.class).toLowerCase()}`).join(", ")}
                  </dd>
                </>
              )}
              <dt>Method</dt>
              <dd>
                {result.incident_source === "clip"
                  ? "CLIP ViT-B/32 + linear classifier"
                  : result.incident_source === "heuristic"
                    ? "Backend motion heuristic"
                    : "No model verdict"}
              </dd>
            </dl>
          </details>
        </>
      )}
    </div>
  );
}

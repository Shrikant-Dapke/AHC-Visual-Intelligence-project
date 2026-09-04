import type { LogEntry } from "../hooks/useRealtimeVideoAnalysis";

const kindLabel = { ANOMALY: "Anomaly", PEAK: "Peak", RESOLVED: "Resolved" } as const;

export default function EventLog({ log, onSeek }: { log: LogEntry[]; onSeek: (t: number) => void }) {
  if (log.length === 0) return null;
  const items = [...log].reverse().slice(0, 8);
  return (
    <div className="section">
      <h2 className="section-label">Events</h2>
      <ul className="log-list">
        {items.map((e, i) => (
          <li key={`${e.t}-${i}`}>
            <button
              onClick={() => onSeek(e.t)}
              title="Seek to event"
              style={{
                all: "unset",
                display: "flex",
                gap: 10,
                alignItems: "baseline",
                width: "100%",
                cursor: "pointer",
              }}
            >
              <span className="log-t">{e.t.toFixed(1)}s</span>
              <span className={`log-kind ${e.kind}`}>{kindLabel[e.kind]}</span>
              <span className="log-text">{e.text.toLowerCase()}</span>
              <span className="log-score">{e.score}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

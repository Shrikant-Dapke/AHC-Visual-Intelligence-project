import AnomalyGauge from "./AnomalyGauge";
import SignalGraph from "./SignalGraph";
import type {
  HistoryPoint,
  LiveSnapshot,
} from "../hooks/useRealtimeVideoAnalysis";

type Props = {
  snapshot: LiveSnapshot;
  history: HistoryPoint[];
  started: boolean;
};

const cls = { NORMAL: "st-normal", ELEVATED: "st-elevated", ANOMALOUS: "st-anomalous" } as const;
const label = { NORMAL: "Normal", ELEVATED: "Elevated", ANOMALOUS: "Anomalous" } as const;

export default function LiveIntelligence({ snapshot, history, started }: Props) {
  const state = cls[snapshot.state];
  return (
    <div className="section">
      <h2 className="section-label">Live anomaly</h2>
      <div className={`anomaly-num ${state}`}>{started ? Math.round(snapshot.score) : "–"}</div>
      <div className="anomaly-den">/ 100</div>
      <div>
        <span className={`anomaly-state ${state}`}>{started ? label[snapshot.state] : "Waiting"}</span>
      </div>
      <AnomalyGauge value={started ? snapshot.score : 0} state={snapshot.state} />
      <div className="stat-rows">
        <div className="stat-row">
          <span className="k">Motion</span>
          <span className="v">{snapshot.motion.toFixed(2)}</span>
        </div>
        <div className="stat-row">
          <span className="k">Change</span>
          <span className="v">{snapshot.changeRate.toFixed(2)}</span>
        </div>
        <div className="stat-row">
          <span className="k">Baseline</span>
          <span className="v">{snapshot.baseline.toFixed(2)}</span>
        </div>
      </div>
      <div className="graph-wrap">
        <SignalGraph history={started ? history : []} />
        <div className="graph-cap">
          <span>Live signal</span>
          <span>last 12s</span>
        </div>
      </div>
    </div>
  );
}

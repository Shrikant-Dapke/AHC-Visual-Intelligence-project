type Props = {
  health: "ok" | "down" | "checking";
  labelsConfigured: boolean | null;
  modelBackend: string;
  liveActive: boolean;
};

export default function Header({ health, labelsConfigured, modelBackend, liveActive }: Props) {
  const sysClass = health === "ok" ? "" : health === "down" ? "down" : "checking";
  const sysText = health === "ok" ? "System Online" : health === "down" ? "Backend unreachable" : "Connecting…";
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">A</div>
        <div>
          <div className="brand-name">
            AHC <span>Visual Intelligence</span>
          </div>
          <div className={`sys-online ${sysClass}`}>
            <span className="dot" />
            {sysText}
          </div>
        </div>
      </div>
      <div className="top-meta">
        <span className="hide-sm">
          {liveActive ? <b>Live</b> : "Live"}
        </span>
        <span className="hide-sm">
          Model <b>{modelBackend === "heuristic" ? "Heuristic" : modelBackend}</b>
        </span>
        {labelsConfigured === false && <span className="warn">Taxonomy pending</span>}
      </div>
    </header>
  );
}

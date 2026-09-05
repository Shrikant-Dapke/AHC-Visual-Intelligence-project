import type { ReactNode } from "react";
import Header from "./Header";

type Props = {
  health: "ok" | "down" | "checking";
  labelsConfigured: boolean | null;
  modelBackend: string;
  liveActive: boolean;
  clipAvailable?: boolean | null;
  children: ReactNode;
};

export default function AppShell({ health, labelsConfigured, modelBackend, liveActive, clipAvailable = null, children }: Props) {
  return (
    <div className="shell">
      <Header health={health} labelsConfigured={labelsConfigured} modelBackend={modelBackend} liveActive={liveActive} clipAvailable={clipAvailable} />
      {children}
      <footer className="statusbar">AHC Visual Intelligence · Local analysis</footer>
    </div>
  );
}

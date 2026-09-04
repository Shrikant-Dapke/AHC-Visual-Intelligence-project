import type { ReactNode } from "react";
import Header from "./Header";

type Props = {
  health: "ok" | "down" | "checking";
  labelsConfigured: boolean | null;
  modelBackend: string;
  liveActive: boolean;
  children: ReactNode;
};

export default function AppShell({ health, labelsConfigured, modelBackend, liveActive, children }: Props) {
  return (
    <div className="shell">
      <Header health={health} labelsConfigured={labelsConfigured} modelBackend={modelBackend} liveActive={liveActive} />
      {children}
      <footer className="statusbar">AHC Visual Intelligence · Local analysis</footer>
    </div>
  );
}

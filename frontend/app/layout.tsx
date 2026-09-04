import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AHC Visual Intelligence",
  description: "AI-powered road and incident video intelligence workstation"
};

export const viewport: Viewport = {
  themeColor: "#05080e"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

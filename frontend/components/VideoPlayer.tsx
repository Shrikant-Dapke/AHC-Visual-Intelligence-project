"use client";

type Props = {
  src: string;
  videoRef: { current: HTMLVideoElement | null };
  onTick: (t: number) => void;
  onMeta: (duration: number) => void;
};

export function fmtTime(s: number): string {
  if (!isFinite(s) || s < 0) return "00:00.0";
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${String(m).padStart(2, "0")}:${sec.toFixed(1).padStart(4, "0")}`;
}

export default function VideoPlayer({ src, videoRef, onTick, onMeta }: Props) {
  return (
    <video
      ref={videoRef}
      src={src}
      controls
      preload="metadata"
      playsInline
      onTimeUpdate={(e) => onTick(e.currentTarget.currentTime)}
      onLoadedMetadata={(e) => {
        onMeta(e.currentTarget.duration);
        onTick(0);
      }}
    />
  );
}

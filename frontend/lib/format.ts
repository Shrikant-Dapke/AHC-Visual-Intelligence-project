export function isMp4(file: File): boolean {
  return file.name.toLowerCase().endsWith(".mp4");
}

export function fmtSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** "vehicle_blocking_traffic" -> "Vehicle Blocking Traffic". */
export function prettyClass(raw: string): string {
  return raw
    .split("_")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/** Detector vocabulary -> concise UI plurals. Unknown classes pass through. */
const OBJECT_LABELS: Record<string, string> = {
  car: "Cars",
  person: "People",
  truck: "Trucks",
  bus: "Buses",
  motorcycle: "Motorcycles",
  bicycle: "Bicycles",
  "traffic light": "Traffic lights",
  "stop sign": "Stop signs",
};

export const KNOWN_OBJECT_CLASSES = [
  "car",
  "person",
  "truck",
  "bus",
  "motorcycle",
  "bicycle",
  "traffic light",
  "stop sign",
];

export function objectLabel(cls: string): string {
  return OBJECT_LABELS[cls] ?? prettyClass(cls);
}

/** "clip" | "heuristic" | "none" -> honest human-readable source. */
export function sourceLabel(source: string): string {
  if (source === "clip") return "CLIP classifier";
  if (source === "heuristic") return "Motion heuristic";
  return "No prediction";
}

import type { ObjectTimeline, ObjectTrack, TrackSummary } from "../../shared/types";
import { prettyClass } from "../lib/format";
import { fmtTime } from "./VideoPlayer";

type Props = {
  tracks: ObjectTrack[];
  timeline: ObjectTimeline | null;
  summary?: TrackSummary | null;
};

/** Human-readable tracking insights. Every claim derives from the response —
 *  no raw trajectories, no invented events. */
export default function TrackInsights({ tracks, timeline, summary = null }: Props) {
  if (tracks.length === 0) {
    return (
      <div className="section">
        <h2 className="section-label">Tracking</h2>
        <p className="empty-note">
          No persistent tracks
          <small>Objects were not observed across enough frames to track.</small>
        </p>
      </div>
    );
  }
  const noun = (c: string) =>
    c === "person" ? "person" : ["car", "truck", "bus", "motorcycle", "bicycle"].includes(c) ? "vehicle" : "object";
  const longest = tracks.reduce((a, b) => (b.duration > a.duration ? b : a));
  const moved = tracks.reduce((a, b) => (b.displacement_px > a.displacement_px ? b : a));
  const spanning =
    timeline != null
      ? tracks.filter((t) => t.first_seen <= timeline.start && t.last_seen >= timeline.end)
      : [];
  const shown = tracks.slice(0, 12);

  return (
    <div className="section">
      <h2 className="section-label">Tracking</h2>
      <div className="stat-rows" style={{ marginTop: 0 }}>
        <div className="stat-row">
          <span className="k">Unique tracked objects</span>
          <span className="v">{summary?.unique_count ?? tracks.length}</span>
        </div>
        {summary != null && (
          <div className="stat-row">
            <span className="k">Active tracks</span>
            <span className="v">{summary.active_count}</span>
          </div>
        )}
        <div className="stat-row">
          <span className="k">Longest track</span>
          <span className="v">
            {longest.duration.toFixed(1)}s · {noun(longest.class)} #{longest.track_id}
          </span>
        </div>
        {moved.displacement_px > 0 && (
          <div className="stat-row">
            <span className="k">Largest movement</span>
            <span className="v">
              track #{moved.track_id} · ~{Math.round(moved.displacement_px)}px
            </span>
          </div>
        )}
        {timeline != null && spanning.length > 0 && (
          <div className="stat-row">
            <span className="k">Through window</span>
            <span className="v">
              {spanning.length} track{spanning.length === 1 ? "" : "s"} span{" "}
              {fmtTime(timeline.start)}–{fmtTime(timeline.end)}
            </span>
          </div>
        )}
        {summary != null && summary.fragmented && (
          <div className="stat-row">
            <span className="k">Identity notes</span>
            <span className="v">
              {summary.merged_groups} fragmented group{summary.merged_groups === 1 ? "" : "s"} stitched
            </span>
          </div>
        )}
      </div>
      <details className="tech-details">
        <summary>All tracks</summary>
        <dl>
          {shown.map((t) => (
            <div key={t.track_id} style={{ display: "contents" }}>
              <dt>
                #{t.track_id} {prettyClass(t.class).toLowerCase()}
              </dt>
              <dd>
                {t.first_seen.toFixed(1)}–{t.last_seen.toFixed(1)}s · {t.hits} sightings
              </dd>
            </div>
          ))}
        </dl>
        {tracks.length > shown.length && (
          <div style={{ marginTop: 6 }}>+{tracks.length - shown.length} more</div>
        )}
      </details>
    </div>
  );
}

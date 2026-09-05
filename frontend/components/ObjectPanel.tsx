import type { ObjectCount } from "../../shared/types";
import { KNOWN_OBJECT_CLASSES, objectLabel } from "../lib/format";

/** Unique tracked objects per class — one count per de-duplicated track
 *  identity, never one count per frame detection. Only real backend counts. */
export default function ObjectPanel({ objects }: { objects: ObjectCount[] }) {
  const byClass = new Map(objects.map((o) => [o.class, o.count]));
  const total = objects.reduce((n, o) => n + o.count, 0);
  return (
    <div className="section">
      <h2 className="section-label">Object intelligence</h2>
      {total === 0 ? (
        <p className="empty-note">
          No road objects detected
          <small>Nothing above the detection threshold in the analyzed frames.</small>
        </p>
      ) : (
        <>
          <p className="empty-note" style={{ margin: "0 0 8px" }}>
            {total} unique tracked object{total === 1 ? "" : "s"}
          </p>
          <div className="obj-grid">
          {KNOWN_OBJECT_CLASSES.map((cls) => {
            const n = byClass.get(cls) ?? 0;
            return (
              <div key={cls} className={`obj-cell${n === 0 ? " zero" : ""}`}>
                <span className="obj-num">{n}</span>
                <span className="obj-name">{objectLabel(cls)}</span>
              </div>
            );
          })}
          </div>
        </>
      )}
    </div>
  );
}

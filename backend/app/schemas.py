"""Pydantic schemas. Mirrors shared/types.ts — keep both in sync."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "ahc-visual-intelligence-backend"
    model_backend: str = "heuristic"
    labels_configured: bool = False
    # Task 3: availability probes (cheap; never load weights). Defaults keep
    # old clients working.
    clip_available: bool = False
    clip_reason: str | None = None
    yolo_available: bool = False
    yolo_reason: str | None = None


class EventSpan(BaseModel):
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    label: str
    score: float = Field(ge=0.0, le=1.0)


# --- Task 2: object + temporal intelligence (ADDITIVE; the original models
# above are untouched). Mirrors shared/types.ts — keep both in sync. ------


class ObjectCount(BaseModel):
    object_class: str = Field(alias="class")
    count: int = Field(ge=0)
    track_ids: list[int] = []

    model_config = {"populate_by_name": True}


class TrackPoint(BaseModel):
    t: float = Field(ge=0)
    frame: int = Field(ge=0)
    cx: float
    cy: float


class ObjectTrack(BaseModel):
    track_id: int
    object_class: str = Field(alias="class")
    first_seen: float = Field(ge=0)
    last_seen: float = Field(ge=0)
    duration: float = Field(ge=0)
    hits: int = Field(ge=1)
    displacement_px: float = Field(ge=0)
    trajectory: list[TrackPoint] = []

    model_config = {"populate_by_name": True}


class EvidenceItem(BaseModel):
    kind: str  # "pre" | "peak" | "post"
    timestamp: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    # Internal server path (used for file serving). Excluded from API
    # responses so server directory layout never leaks to clients.
    path: str = Field(default="", exclude=True)
    # Task 3: HTTP URL for the frontend (filled by the backend; clients must
    # use this, never the filesystem `path`).
    url: str | None = None


class ObjectTimeline(BaseModel):
    start: float = Field(ge=0)
    peak: float = Field(ge=0)
    end: float = Field(ge=0)


class IncidentRef(BaseModel):
    """Primary incident prediction (CLIP once trained). This layer passes it
    through verbatim and never invents it: default is unknown/0.0/none."""

    object_class: str = Field(default="unknown", alias="class")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "none"  # "clip" | "passthrough" | "none"

    model_config = {"populate_by_name": True}


class DetectorRef(BaseModel):
    name: str = "yolo"
    model: str = "yolov8n.pt"
    conf: float = 0.35
    imgsz: int = 640


class ObjectAnalysisResult(BaseModel):
    """Unified Task 2 result. A future /api route can return this directly,
    or merge timeline->events + incident into the existing AnalyzeResult."""

    video_id: str
    duration_sec: float = Field(ge=0)
    fps: float = Field(ge=0)
    frames_analyzed: int = Field(ge=0)
    sample_fps: float = Field(gt=0)
    detector: DetectorRef = DetectorRef()
    tracker: str = "iou"
    incident: IncidentRef = IncidentRef()
    timeline: ObjectTimeline
    objects: list[ObjectCount] = []
    tracks: list[ObjectTrack] = []
    track_summary: TrackSummary | None = None
    detections: list[FrameDetections] = []
    evidence: list[EvidenceItem] = []


# --- Task 3: unified CLIP+YOLO response ----------------------------------


class AnalyzeResult(BaseModel):
    job_id: str
    incident_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    events: list[EventSpan] = []
    explanation: str
    thumbnail_urls: list[str] = []
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    # Task 3 additions (all optional/additive; old clients keep working).
    # incident_source: "clip" (trained classifier) | "heuristic" (motion
    # baseline) | "none" (unknown). YOLO evidence never sets this.
    incident_source: str = "none"
    timeline: ObjectTimeline | None = None
    objects: list[ObjectCount] = []
    tracks: list[ObjectTrack] = []
    track_summary: TrackSummary | None = None
    detections: list[FrameDetections] = []
    evidence: list[EvidenceItem] = []
    warnings: list[str] = []


class ErrorResponse(BaseModel):
    detail: str


# --- Task 10: frame-level tracking observations + honest counts ------------
# Additive; all optional so old clients keep working. Mirrors
# shared/types.ts — keep both in sync.


class TrackBox(BaseModel):
    """One tracked object visible in one sampled frame.

    bbox is xyxy in 0..1 fractions of the frame so the frontend overlay
    stays aligned at any display size. id is the canonical (stitched)
    track identity shared with ObjectCount.track_ids.
    """

    id: int
    object_class: str = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)

    model_config = {"populate_by_name": True}


class FrameDetections(BaseModel):
    timestamp: float = Field(ge=0)
    tracks: list[TrackBox] = []


class TrackSummary(BaseModel):
    """Honest tracking statistics.

    unique_count = canonical (de-duplicated) tracks, the basis of the
    dashboard object counts. fragmented = True means the tracker observed
    and stitched identity fragmentation, so even this count is a heuristic
    ("tracked objects"), never a guaranteed physical census.
    """

    unique_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    longest_seconds: float = Field(ge=0)
    largest_movement_px: float = Field(ge=0)
    merged_groups: int = Field(ge=0)
    total_raw_detections: int = Field(ge=0)
    fragmented: bool = False

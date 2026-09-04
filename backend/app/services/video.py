"""Video ingest + probing with OpenCV only. No ffmpeg required."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass
class SampledFrame:
    t_sec: float
    frame_index: int
    # Downscaled grayscale frame kept in memory for motion scoring.
    # Full-res color frames are NOT retained to keep memory bounded.
    gray_small: object


ALLOWED_EXTENSIONS = {".mp4"}
ALLOWED_CONTENT_TYPES = {"video/mp4", "application/octet-stream"}


def ensure_upload_dir(upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def validate_filename(filename: str | None) -> str:
    if not filename or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .mp4 uploads are supported for the demo.")
    return filename


async def save_upload(read_chunk, filename: str, dest_dir: Path, max_mb: int) -> Path:
    """Stream an upload to disk with a size cap. read_chunk() -> bytes."""
    ensure_upload_dir(dest_dir)
    validate_filename(filename)
    job_id = uuid4().hex[:12]
    dest = dest_dir / f"{job_id}_{Path(filename).name}"
    max_bytes = max_mb * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await read_chunk(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise ValueError(f"File exceeds {max_mb} MB limit.")
                f.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    if written == 0:
        dest.unlink(missing_ok=True)
        raise ValueError("Empty file uploaded.")
    return dest


def probe_video(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Unreadable video: cv2.VideoCapture could not open file.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count <= 0:
            # Fallback: count manually for containers that misreport.
            frame_count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                frame_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if frame_count <= 0:
            raise ValueError("Zero-frame video: no decodable frames found.")
        if fps <= 0:
            fps = 30.0  # container did not report fps; assume and continue
        duration = frame_count / fps if fps else 0.0
        return VideoMeta(
            path=path, fps=fps, frame_count=frame_count,
            duration_sec=duration, width=width, height=height,
        )
    finally:
        cap.release()


def sample_frames(meta: VideoMeta, sample_fps: float = 1.0) -> list[SampledFrame]:
    """Sample ~sample_fps frames/sec as small grayscale images (deterministic)."""
    import numpy as np

    cap = cv2.VideoCapture(str(meta.path))
    out: list[SampledFrame] = []
    try:
        if not cap.isOpened():
            raise ValueError("Unreadable video during sampling.")
        native_fps = meta.fps or 30.0
        step = max(1, int(round(native_fps / max(sample_fps, 0.25))))
        idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                out.append(SampledFrame(
                    t_sec=idx / native_fps, frame_index=idx, gray_small=np.asarray(small),
                ))
                kept += 1
                if kept > 1200:  # ~20 min at 1fps; bound memory for demo
                    break
            idx += 1
        if not out:
            raise ValueError("Zero-frame video: sampling produced no frames.")
        return out
    finally:
        cap.release()


def cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass  # best-effort; never fail a request on cleanup

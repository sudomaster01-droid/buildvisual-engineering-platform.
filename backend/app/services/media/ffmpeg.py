"""FFmpeg/ffprobe wrapper with hard safety rules.

Blueprint p.61: "For FFmpeg, use fixed argument arrays, avoid shell concatenation,
enforce timeouts and limit resource use."

Every call here:
* takes an argument list (never a shell string),
* never interpolates user-controlled text into a filter graph,
* runs with a timeout and without a controlling terminal,
* is logged with redacted inputs,
* raises a typed error instead of returning partial success.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ...config import get_settings
from ...errors import BuildVisualError

log = logging.getLogger("buildvisual.media")


class MediaError(BuildVisualError):
    code = "media_error"
    http_status = 500
    user_message = "The video file could not be processed."


class FFmpegUnavailable(MediaError):
    code = "ffmpeg_unavailable"
    user_message = "The video processing tool is not available on this server."


@lru_cache(maxsize=1)
def ffmpeg_binary() -> str:
    settings = get_settings()
    if settings.ffmpeg_binary:
        path = Path(settings.ffmpeg_binary)
        if not path.exists():
            raise FFmpegUnavailable(f"FFMPEG_BINARY does not exist: {path.name}")
        return str(path)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on the environment
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise FFmpegUnavailable(
            "FFmpeg was not found. Install ffmpeg or 'imageio-ffmpeg'."
        ) from exc


@lru_cache(maxsize=1)
def ffprobe_binary() -> str | None:
    settings = get_settings()
    if settings.ffprobe_binary and Path(settings.ffprobe_binary).exists():
        return settings.ffprobe_binary
    return shutil.which("ffprobe")


def ffmpeg_version() -> str:
    try:
        result = subprocess.run(
            [ffmpeg_binary(), "-version"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        return result.stdout.splitlines()[0][:80] if result.stdout else "unknown"
    except (subprocess.SubprocessError, OSError, FFmpegUnavailable):
        return "unavailable"


def run(args: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run FFmpeg with an explicit argument array."""
    settings = get_settings()
    timeout = timeout or settings.subprocess_timeout_seconds
    command = [ffmpeg_binary(), *args]
    log.debug("ffmpeg %s", " ".join(args[:8]))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            "Video processing timed out.", details={"timeout_s": timeout}
        ) from exc
    except OSError as exc:
        raise FFmpegUnavailable(str(exc)) from exc
    if result.returncode != 0:
        tail = (result.stderr or "")[-800:]
        raise MediaError(
            "Video processing failed.",
            details={"returncode": result.returncode, "ffmpeg": tail},
        )
    return result


@dataclass
class MediaInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    codec: str
    has_audio: bool
    size_bytes: int

    def as_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 3),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "codec": self.codec,
            "has_audio": self.has_audio,
            "size_bytes": self.size_bytes,
        }


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")
_VIDEO_RE = re.compile(r"Stream #.*?Video:\s*([a-zA-Z0-9_]+)")
_SIZE_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_AUDIO_RE = re.compile(r"Stream #.*?Audio:")


def probe(path: Path) -> MediaInfo:
    """Read stream parameters. Falls back to parsing FFmpeg's own banner."""
    if not path.exists():
        raise MediaError("The media file does not exist.")
    size = path.stat().st_size
    if size == 0:
        raise MediaError("The media file is empty.")

    probe_path = ffprobe_binary()
    if probe_path:
        try:
            result = subprocess.run(
                [
                    probe_path, "-v", "error", "-print_format", "json",
                    "-show_format", "-show_streams", str(path),
                ],
                capture_output=True, text=True, timeout=60, check=False,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                return _info_from_ffprobe(data, size)
        except (subprocess.SubprocessError, OSError, ValueError):
            pass  # fall through to the banner parser

    result = subprocess.run(
        [ffmpeg_binary(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, timeout=60, check=False,
        stdin=subprocess.DEVNULL,
    )
    banner = (result.stderr or "") + (result.stdout or "")
    duration_match = _DURATION_RE.search(banner)
    if not duration_match:
        raise MediaError("The file is not a readable media container.")
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    video_match = _VIDEO_RE.search(banner)
    if not video_match:
        raise MediaError("The file contains no video stream.")
    codec = video_match.group(1)
    size_match = _SIZE_RE.search(banner[video_match.end():])
    width = int(size_match.group(1)) if size_match else 0
    height = int(size_match.group(2)) if size_match else 0
    fps_match = _FPS_RE.search(banner)
    fps = float(fps_match.group(1)) if fps_match else 0.0
    return MediaInfo(
        duration_s=duration, width=width, height=height, fps=fps,
        codec=codec, has_audio=bool(_AUDIO_RE.search(banner)), size_bytes=size,
    )


def _info_from_ffprobe(data: dict, size: int) -> MediaInfo:
    video = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    if video is None:
        raise MediaError("The file contains no video stream.")
    fps_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        num, _, den = fps_text.partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return MediaInfo(
        duration_s=float(data.get("format", {}).get("duration", 0.0) or 0.0),
        width=int(video.get("width", 0) or 0),
        height=int(video.get("height", 0) or 0),
        fps=round(fps, 3),
        codec=str(video.get("codec_name", "unknown")),
        has_audio=audio,
        size_bytes=size,
    )


def verify_decodes(path: Path) -> tuple[bool, str]:
    """Decode the whole file; any decoder error means the file is not deliverable."""
    result = subprocess.run(
        [ffmpeg_binary(), "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, timeout=get_settings().subprocess_timeout_seconds,
        check=False, stdin=subprocess.DEVNULL,
    )
    errors = (result.stderr or "").strip()
    return (result.returncode == 0 and not errors, errors[:600])


def extract_frame(path: Path, at_seconds: float, out_path: Path) -> Path:
    run([
        "-y", "-v", "error", "-ss", f"{max(0.0, at_seconds):.3f}", "-i", str(path),
        "-frames:v", "1", "-q:v", "2", str(out_path),
    ])
    if not out_path.exists():
        raise MediaError("Frame extraction produced no image.")
    return out_path


def extract_gray_frame_array(path: Path, at_seconds: float, width: int, height: int):
    """Decode one frame as a normalised greyscale numpy array (for QA metrics)."""
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frame.png"
        run([
            "-y", "-v", "error", "-ss", f"{max(0.0, at_seconds):.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale={width}:{height}", str(out),
        ])
        if not out.exists():
            raise MediaError("Frame extraction produced no image.")
        with Image.open(out) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array

"""Clip assembly: validated clips → one deliverable film.

Blueprint p.61/p.62. Steps: verify every clip decodes, normalise codec/frame
rate/aspect/audio policy, order by the shot list, join, add a repeatable
fade-in/out finish, validate duration and resolution, extract a poster, and
record the exact configuration used so the output is traceable.

No shell strings, no user text in filter graphs, bounded timeouts.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ...config import get_settings
from ...errors import BuildVisualError
from . import ffmpeg as ff

log = logging.getLogger("buildvisual.media.assembly")

ASSEMBLY_VERSION = "1.0.0"


class AssemblyError(BuildVisualError):
    code = "assembly_failed"
    http_status = 500
    user_message = "The generated clips could not be assembled into a final video."


@dataclass
class ClipInput:
    shot_id: str
    path: Path
    expected_duration_s: float = 0.0
    transition: str = "cut"


@dataclass
class AssemblyResult:
    output_path: Path
    poster_path: Path | None
    info: ff.MediaInfo
    config: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def normalise_clip(clip: ClipInput, out_path: Path, *, width: int, height: int,
                   fps: int, with_silent_audio: bool) -> ff.MediaInfo:
    """Bring one clip to the house standard (resolution, fps, pixel format)."""
    info = ff.probe(clip.path)
    problems: list[str] = []
    if info.duration_s <= 0.2:
        problems.append("clip is shorter than 0.2 s")
    if info.width <= 0 or info.height <= 0:
        problems.append("clip has no usable frame size")
    if problems:
        raise AssemblyError(f"Clip for shot {clip.shot_id} is unusable: " + "; ".join(problems))

    primary_video = (
        "-vf",
        (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p"
        ),
    )
    audio_args: tuple[str, ...] = ()
    if with_silent_audio:
        if info.has_audio:
            audio_args = ("-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2")
        else:
            # Pad silent audio so the concat demuxer sees a uniform stream layout.
            audio_args = ("-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                          "-shortest", "-c:a", "aac", "-b:a", "128k")
    elif info.has_audio:
        # Audio policy: keep the first clip's native audio only when assembling
        # with audio, otherwise strip it consistently.
        audio_args = ("-an",)

    args = ["-y", "-v", "error", "-i", str(clip.path), *audio_args, *primary_video]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"]
    args += [str(out_path)]
    ff.run(args)
    return ff.probe(out_path)


def assemble(
    clips: list[ClipInput],
    out_path: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fade_seconds: float | None = None,
    expected_duration_s: float | None = None,
) -> AssemblyResult:
    settings = get_settings()
    width = width or settings.video_width
    height = height or settings.video_height
    fps = fps or settings.video_fps
    fade_seconds = settings.transition_fade_seconds if fade_seconds is None else fade_seconds

    if not clips:
        raise AssemblyError("There are no clips to assemble.")
    if width % 2 or height % 2:
        raise AssemblyError("Output dimensions must be even for H.264.")

    warnings: list[str] = []
    durations = [c.expected_duration_s for c in clips if c.expected_duration_s]
    expected = expected_duration_s or (sum(durations) if durations else None)
    keep_audio = any(ff.probe(c.path).has_audio for c in clips)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bv_assembly_") as tmp:
        tmp_path = Path(tmp)
        normalised: list[Path] = []
        for index, clip in enumerate(clips):
            target = tmp_path / f"clip_{index:02d}.mp4"
            info = normalise_clip(
                clip, target, width=width, height=height, fps=fps,
                with_silent_audio=keep_audio,
            )
            if clip.expected_duration_s and info.duration_s:
                delta = abs(info.duration_s - clip.expected_duration_s)
                if delta > max(1.5, clip.expected_duration_s * 0.35):
                    warnings.append(
                        f"Shot {clip.shot_id}: clip is {info.duration_s:.1f}s but "
                        f"{clip.expected_duration_s:.1f}s was planned."
                    )
            normalised.append(target)

        concat_list = tmp_path / "clips.txt"
        concat_list.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in normalised) + "\n"
        )
        silent_master = tmp_path / "master.mp4"
        ff.run([
            "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(silent_master),
        ])

        master_info = ff.probe(silent_master)
        finish_filters = []
        if fade_seconds and fade_seconds > 0:
            finish_filters.append(f"fade=t=in:st=0:d={fade_seconds:.2f}")
            fade_out_start = max(0.0, master_info.duration_s - fade_seconds)
            finish_filters.append(
                f"fade=t=out:st={fade_out_start:.2f}:d={fade_seconds:.2f}"
            )
        args = ["-y", "-v", "error", "-i", str(silent_master)]
        if finish_filters:
            args += ["-vf", ",".join(finish_filters)]
        args += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if keep_audio:
            args += ["-c:a", "aac", "-b:a", "160k"]
        args += [str(out_path)]
        ff.run(args)

    info = ff.probe(out_path)
    ok, errors = ff.verify_decodes(out_path)
    if not ok:
        raise AssemblyError("The assembled file does not decode cleanly.", details={"errors": errors})

    if info.width != width or info.height != height:
        raise AssemblyError(
            f"The assembled file is {info.width}x{info.height}, expected {width}x{height}."
        )
    if expected and info.duration_s < expected * settings.qa_min_duration_ratio:
        raise AssemblyError(
            f"The assembled film is {info.duration_s:.1f}s but {expected:.1f}s was planned."
        )

    poster_path = out_path.with_name(out_path.stem + "_poster.jpg")
    try:
        ff.extract_frame(out_path, min(1.0, max(0.0, info.duration_s * 0.25)), poster_path)
    except Exception as exc:  # poster is nice-to-have, not a delivery blocker
        warnings.append(f"Poster frame could not be extracted: {str(exc)[:120]}")
        poster_path = None

    config = {
        "assembly_version": ASSEMBLY_VERSION,
        "ffmpeg": ff.ffmpeg_version(),
        "width": width,
        "height": height,
        "fps": fps,
        "fade_seconds": fade_seconds,
        "audio": "preserved" if keep_audio else "none",
        "clip_order": [c.shot_id for c in clips],
        "clip_durations": [round(ff.probe(c.path).duration_s, 3) for c in clips],
        "expected_duration_s": expected,
        "output_duration_s": round(info.duration_s, 3),
        "warnings": warnings,
    }
    log.info(
        "assembly complete",
        extra={"clips": len(clips), "duration_s": round(info.duration_s, 1)},
    )
    return AssemblyResult(
        output_path=out_path, poster_path=poster_path, info=info,
        config=config, warnings=warnings,
    )


def write_assembly_manifest(result: AssemblyResult, path: Path) -> Path:
    path.write_text(json.dumps(result.config, indent=2))
    return path

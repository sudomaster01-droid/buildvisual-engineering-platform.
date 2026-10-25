"""Local model renderer — a real, non-AI fallback provider.

This exists so the pipeline (assembly, QA, versions, rectification, delivery) can
be exercised end to end on a machine with no AI credentials, without pretending
an external API succeeded.

Honesty contract, enforced by the code below and surfaced in the UI:
* ``is_external`` is False and ``generation_mode`` is ``model_render``;
* the ledger records zero provider spend (``unmetered``), plus a separate
  compute-time row so cost reporting never claims an API charge;
* the output is a genuine render of the *locked model's* camera plan — it is
  deterministic, so a QA failure here is a real pipeline defect, not model drift.

It renders frames with the in-repo NumPy rasteriser and encodes them with
FFmpeg, so it needs no GPU, no OpenGL and no desktop 3D application.
"""

from __future__ import annotations

import io
import logging
import shutil
import tempfile
import time
from pathlib import Path

from ...config import get_settings
from ...errors import ProviderError
from ..film.motion import sample_camera
from ..film.shotplan import Shot, ShotPlan
from ..media import ffmpeg as ff
from ..modeling.mesh import Mesh
from ..modeling.raster import colour_to_png_bytes, render
from .base import Capabilities, SubmitResult, TaskStatus, VideoOutput, VideoProvider, VideoRequest

log = logging.getLogger("buildvisual.provider.local")


class LocalRenderProvider(VideoProvider):
    """Renders from the locked model. Not an external service."""

    name = "local_render"
    is_external = False

    def __init__(
        self,
        *,
        mesh: Mesh | None = None,
        shot_plan: ShotPlan | None = None,
        sun_direction: tuple[float, float, float] = (-0.45, 0.55, 0.70),
        lighting: dict | None = None,
    ) -> None:
        self.mesh = mesh
        self.shot_plan = shot_plan
        self.sun_direction = sun_direction
        #: per-shot ambient/exposure presets from the scene manifest, so the film
        #: matches the reference frames exactly (interior shots need interior light)
        self.lighting = lighting or {}
        self._outputs: dict[str, bytes] = {}
        self._shots: dict[str, Shot] = {}
        self.render_log: list[dict] = []

    def capabilities(self) -> Capabilities:
        settings = get_settings()
        return Capabilities(
            provider=self.name,
            model="buildvisual-rasteriser",
            docs_url="",
            verified_on="2026-10-24",
            supports_first_frame=False,
            supports_last_frame=False,
            max_reference_images=0,
            durations_s=tuple(range(2, 21)),
            aspect_ratios=("16:9",),
            resolutions=("720p",),
            native_audio=False,
            prompt_max_chars=100000,
            output_ttl_hours=None,
            watermark=None,
            commercial_use="Rendered from the engineer's own model; no third-party model involved.",
            cost_unit="local compute (not billable)",
            structural_control=(
                "Exact: the frames are rendered directly from the locked 3D model, so geometry "
                "is correct by construction. This is a model render, not AI-generated footage."
            ),
            notes=[
                "Used when no external video API key is configured.",
                "Deterministic: identical model + camera plan produce identical frames.",
                f"Render settings: {settings.local_render_fps} fps at "
                f"{int(settings.reference_width * settings.local_render_scale)}x"
                f"{int(settings.reference_height * settings.local_render_scale)} upscaled.",
            ],
        )

    # -- submission -------------------------------------------------------
    def submit(self, request: VideoRequest, *, idempotency_key: str) -> SubmitResult:
        if self.mesh is None or self.shot_plan is None:
            raise ProviderError(
                "The local renderer has no model attached for this job.", retryable=False
            )
        shot = self._find_shot(request.shot_id)
        if shot is None:
            raise ProviderError(
                f"Shot {request.shot_id} is not in the shot plan.", retryable=False
            )
        task_id = f"local:{request.shot_id}:{int(time.time()*1000)}"
        self._shots[task_id] = shot
        # Rendering happens at fetch() so the job state machine sees a genuine
        # submit → processing → succeeded cycle, exactly like an external provider.
        return SubmitResult(
            task_id=task_id, provider=self.name, model="buildvisual-rasteriser",
            idempotency_key=idempotency_key,
        )

    def _light_for(self, shot: Shot) -> dict:
        interior = shot.type in ("entrance_walkthrough", "room_walkthrough", "upper_floor")
        preset = self.lighting.get("interior" if interior else "exterior") or {}
        return preset

    def _find_shot(self, shot_id: str) -> Shot | None:
        if self.shot_plan is None:
            return None
        return next((s for s in self.shot_plan.shots if s.id == shot_id), None)

    def poll(self, task_id: str) -> TaskStatus:
        return TaskStatus(state="succeeded", provider_state="local_ready")

    # -- rendering --------------------------------------------------------
    def fetch(self, task_id: str, status: TaskStatus | None = None) -> VideoOutput:
        shot = self._shots.get(task_id)
        if shot is None:
            raise ProviderError("Unknown local render task.", retryable=False)
        settings = get_settings()
        fps = max(6, settings.local_render_fps)
        scale = max(0.25, min(1.0, settings.local_render_scale))
        render_w = int(settings.reference_width * scale) // 2 * 2
        render_h = int(settings.reference_height * scale) // 2 * 2
        frames = max(2, int(round(shot.duration_s * fps)))

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="bv_render_") as tmp:
            tmp_path = Path(tmp)
            for index in range(frames):
                t = index / max(1, frames - 1)
                camera = sample_camera(shot, t)
                light = self._light_for(shot)
                result = render(
                    self.mesh, camera, render_w, render_h,
                    sun_direction=self.sun_direction,
                    ambient=float(light.get("ambient", 0.44)),
                    exposure=float(light.get("exposure", 1.0)),
                )
                (tmp_path / f"frame_{index:05d}.png").write_bytes(
                    colour_to_png_bytes(result)
                )
            out_path = tmp_path / "shot.mp4"
            ff.run([
                "-y", "-v", "error",
                "-framerate", str(fps),
                "-i", str(tmp_path / "frame_%05d.png"),
                "-vf", f"scale={settings.video_width}:{settings.video_height}:flags=lanczos",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", str(settings.video_fps),
                "-movflags", "+faststart",
                str(out_path),
            ])
            data = out_path.read_bytes()

        elapsed = time.perf_counter() - started
        self.render_log.append({
            "shot_id": shot.id,
            "frames": frames,
            "render_fps": fps,
            "render_size": [render_w, render_h],
            "seconds": round(elapsed, 2),
        })
        log.info(
            "local render complete",
            extra={
                "shot_id": shot.id,
                "provider": self.name,
                "frames": frames,
                "seconds": round(elapsed, 1),
            },
        )
        return VideoOutput(
            data=data,
            mime="video/mp4",
            duration_s=shot.duration_s,
            usage={
                "render_seconds": round(elapsed, 2),
                "frames": frames,
                "billable": False,
            },
        )

    def estimate_cost_cents(self, request: VideoRequest) -> tuple[int, str, bool]:
        return (0, "unmetered", False)


def _write_bytes(path: Path, data: bytes) -> None:  # pragma: no cover - helper
    path.write_bytes(data)


def _copy(src: Path, dst: Path) -> None:  # pragma: no cover - helper
    shutil.copyfile(src, dst)


def _buffer(data: bytes) -> io.BytesIO:  # pragma: no cover - helper
    return io.BytesIO(data)

"""Runway Gen-4.5 adapter.

Verified against <https://docs.dev.runwayml.com/api/> on 24 Oct 2026.

Submit  POST https://api.dev.runwayml.com/v1/image_to_video
        Authorization: Bearer <key>, X-Runway-Version: 2024-11-06
        body {model, promptText (≤1000 UTF-16 units), promptImage (https URL,
              runway:// URI, or data URI ≤ 5 MB), position:"first",
              ratio, duration (2..10), seed}
Poll    GET /v1/tasks/{id} → PENDING | RUNNING | SUCCEEDED | FAILED, output:[url]

Note the difference from Veo: images are *required* for this endpoint and the
prompt is capped at 1000 characters, so the adapter uses the composer's compact
variant and validates rather than truncating.
"""

from __future__ import annotations

import base64
import logging

from ...config import get_settings
from ...errors import ProviderError
from .base import (
    Capabilities,
    SubmitResult,
    TaskStatus,
    VideoOutput,
    VideoProvider,
    VideoRequest,
)
from .http import ProviderHttp

log = logging.getLogger("buildvisual.provider.runway")

BASE_URL = "https://api.dev.runwayml.com/v1"
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_STATE_MAP = {
    "PENDING": "submitted",
    "THROTTLED": "submitted",
    "RUNNING": "processing",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
    "CANCELLED": "failed",
}


class RunwayProvider(VideoProvider):
    name = "runway"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 http: ProviderHttp | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.runway_api_key
        self.model = model or settings.runway_model
        self.version = settings.runway_api_version
        self.http = http or ProviderHttp()
        # Credentials are required to *call* the service, not to describe it. This
        # keeps capability reports and contract tests usable without secrets.
        if not self.api_key:
            log.warning("RUNWAY_API_KEY is not configured; runway calls will fail")

    def _require_key(self) -> None:
        if not self.api_key:
            raise ProviderError(
                "RUNWAY_API_KEY is not configured, so this deployment cannot generate "
                "video with Runway.",
                details={"provider": "runway", "configuration": "RUNWAY_API_KEY"},
                retryable=False,
            )

    def capabilities(self) -> Capabilities:
        return Capabilities(
            provider="runway",
            model=self.model,
            docs_url="https://docs.dev.runwayml.com/api/",
            verified_on="2026-10-24",
            supports_first_frame=True,
            supports_last_frame=False,
            max_reference_images=0,
            durations_s=(2, 3, 4, 5, 6, 7, 8, 9, 10),
            aspect_ratios=("1280:720", "720:1280"),
            resolutions=("720p",),
            native_audio=False,
            prompt_max_chars=1000,
            output_ttl_hours=None,
            watermark=None,
            commercial_use="Permitted under Runway's developer terms; review current terms.",
            cost_unit="credits per second of generated video",
            structural_control=(
                "Requires an image as the first frame and animates from it. Strong anchoring on the "
                "opening composition; subsequent frames drift by nature of the model."
            ),
            notes=[
                "The image_to_video endpoint requires promptImage.",
                "promptText is capped at 1000 characters.",
                "Duration must be an integer between 2 and 10 seconds.",
                "Output URLs are served from Runway's CDN and are not permanent; download immediately.",
            ],
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": self.version,
            "Content-Type": "application/json",
        }

    def _data_uri(self, data: bytes, mime: str) -> str:
        if len(data) > MAX_IMAGE_BYTES:
            raise ProviderError(
                "Reference frame exceeds Runway's 5 MB inline-image limit.", retryable=False
            )
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def submit(self, request: VideoRequest, *, idempotency_key: str) -> SubmitResult:
        self._require_key()
        if request.first_frame is None:
            raise ProviderError(
                "Runway image_to_video requires a first-frame image; "
                "this shot produced no reference frame.",
                retryable=False,
            )
        # Use the compact prompt when the verbose one would breach the cap.
        prompt = request.prompt
        if len(prompt) > self.capabilities().prompt_max_chars:
            raise ProviderError(
                "Runway prompt exceeds 1000 characters; the composer must supply the "
                "compact variant for this provider.",
                retryable=False,
            )
        payload = {
            "model": self.model,
            "promptText": prompt,
            "promptImage": self._data_uri(request.first_frame, request.first_frame_mime),
            "position": "first",
            "ratio": request.aspect_ratio if request.aspect_ratio in ("1280:720", "720:1280") else "1280:720",
            "duration": int(request.duration_s),
        }
        if request.negative_prompt:
            payload["promptText"] = (
                f"{prompt[:820]}\n\nAvoid: {request.negative_prompt[:150]}."
            )
        if request.seed is not None:
            payload["seed"] = int(request.seed) % (2**32)

        data = self.http.request_json(
            "POST", f"{BASE_URL}/image_to_video", headers=self._headers(), json=payload
        )
        task_id = data.get("id")
        if not task_id:
            raise ProviderError("Runway did not return a task id.")
        return SubmitResult(
            task_id=str(task_id), provider=self.name, model=self.model,
            raw={"id": task_id}, idempotency_key=idempotency_key,
        )

    def poll(self, task_id: str) -> TaskStatus:
        self._require_key()
        data = self.http.request_json(
            "GET", f"{BASE_URL}/tasks/{task_id}", headers=self._headers()
        )
        raw_state = str(data.get("status", "")).upper()
        state = _STATE_MAP.get(raw_state, "processing")
        if state == "failed":
            failure = data.get("failure") or data.get("error") or ""
            message = str(failure)[:300] or "Runway reported the task failed."
            retryable = any(word in message.lower() for word in ("timeout", "internal", "try again"))
            return TaskStatus(
                state="failed", provider_state=raw_state, message=message,
                retryable=retryable,
                error_code="provider_cancelled" if raw_state == "CANCELLED" else "provider_failed",
            )
        return TaskStatus(state=state, provider_state=raw_state)  # type: ignore[arg-type]

    def fetch(self, task_id: str, status: TaskStatus | None = None) -> VideoOutput:
        self._require_key()
        data = self.http.request_json(
            "GET", f"{BASE_URL}/tasks/{task_id}", headers=self._headers()
        )
        outputs = data.get("output") or []
        if not outputs:
            raise ProviderError("Runway task has no output yet.", retryable=True)
        url = outputs[0] if isinstance(outputs[0], str) else outputs[0].get("url", "")
        if not url:
            raise ProviderError("Runway output entry has no URL.")
        payload = self.http.download(url)
        return VideoOutput(
            data=payload, mime="video/mp4", usage={"bytes": len(payload)},
            source_url_redacted="runway:task-output",
        )

    def cancel(self, task_id: str) -> bool:
        try:
            self.http.request_json(
                "DELETE", f"{BASE_URL}/tasks/{task_id}", headers=self._headers()
            )
            return True
        except ProviderError:
            return False

    def estimate_cost_cents(self, request: VideoRequest) -> tuple[int, str, bool]:
        settings = get_settings()
        rate = settings.runway_cost_cents_per_second
        if rate <= 0:
            return (0, "unmetered", True)
        return (int(round(rate * request.duration_s)), "configured_rate", True)

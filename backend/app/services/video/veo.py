"""Google Gemini Omni Flash video adapter.

The previous Veo 3.1 Preview endpoint has been retired. Omni uses the
Interactions API and returns generated video bytes in the completed response.
"""

from __future__ import annotations

import base64
import logging
import uuid

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

log = logging.getLogger("buildvisual.provider.veo")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class VeoProvider(VideoProvider):
    name = "gemini_omni"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 http: ProviderHttp | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.veo_model
        self.http = http or ProviderHttp()
        self._outputs: dict[str, bytes] = {}
        # Credentials are required to *call* the service, not to describe it. This
        # keeps capability reports and contract tests usable without secrets.
        if not self.api_key:
            log.warning("GEMINI_API_KEY is not configured; veo calls will fail")

    #: Provider failures translated into plain language for the engineer.
    FAILURE_MESSAGES = {
        "provider_failed": "Google could not complete this scene.",
        "provider_empty_output": "Google returned no video footage for this scene.",
        "provider_safety_filter": "Google declined this scene under its safety policy.",
        "provider_output_expired": "The generated scene is no longer available.",
        "provider_quota": "The Google video generation quota is exhausted.",
    }

    def describe_failure(self, code: str) -> str:
        """Plain-language message for a provider error code."""
        return self.FAILURE_MESSAGES.get(code, "The video service reported a failure.")

    def _require_key(self) -> None:
        if not self.api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not configured, so this deployment cannot generate "
                "video with Veo.",
                details={"provider": "veo", "configuration": "GEMINI_API_KEY"},
                retryable=False,
            )

    # -- capability discovery -------------------------------------------
    def capabilities(self) -> Capabilities:
        return Capabilities(
            provider=self.name,
            model=self.model,
            docs_url="https://ai.google.dev/gemini-api/docs/omni",
            verified_on="2026-10-25",
            supports_first_frame=True,
            supports_last_frame=False,
            max_reference_images=1,
            durations_s=(8,),
            aspect_ratios=("16:9", "9:16"),
            resolutions=("720p", "1080p"),
            native_audio=False,
            prompt_max_chars=4000,
            output_ttl_hours=None,
            watermark=None,
            commercial_use="Review the current Google Gemini API terms.",
            cost_unit="per second of generated video",
            structural_control=(
                "Accepts a reference image with a text prompt; it does not document "
                "hard geometric constraints."
            ),
            notes=[
                "The Interactions API returns video output with the completed interaction.",
                "The provider's documented video request does not expose clip duration.",
                "The API output contains generated video bytes; no provider URL is needed.",
            ],
        )

    def submit(self, request: VideoRequest, *, idempotency_key: str) -> SubmitResult:
        self._require_key()
        problems = self.validate(request)
        if problems:
            raise ProviderError("Gemini video request rejected before submission: " + "; ".join(problems),
                                retryable=False)
        content: list[dict] = []
        if request.first_frame:
            content.append({
                "type": "image",
                "data": base64.b64encode(request.first_frame).decode("ascii"),
                "mime_type": request.first_frame_mime,
            })
        prompt = request.prompt
        if request.negative_prompt:
            prompt += f"\n\nAvoid: {request.negative_prompt}."
        content.append({"type": "text", "text": prompt})

        data = self.http.request_json(
            "POST", f"{BASE_URL}/interactions",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "model": self.model,
                "input": content,
                "response_format": {
                    "type": "video",
                    "aspect_ratio": request.aspect_ratio,
                    "resolution": request.resolution,
                },
            },
            # This POST generates billable output and has no documented
            # idempotency header. Never retry it implicitly at the HTTP layer.
            max_attempts=1,
            retry_on=(),
        )
        video_bytes = self._video_bytes(data)
        if not video_bytes:
            raise ProviderError(
                "Google completed the interaction without returning video bytes.",
                details={"interaction_status": data.get("status")},
                retryable=False,
            )
        interaction_id = str(data.get("id") or uuid.uuid4().hex)
        self._outputs[interaction_id] = video_bytes
        return SubmitResult(
            task_id=interaction_id,
            provider=self.name,
            model=self.model,
            raw={"interaction_id": interaction_id},
            idempotency_key=idempotency_key,
        )

    def poll(self, task_id: str) -> TaskStatus:
        if task_id in self._outputs:
            return TaskStatus(state="succeeded", provider_state="completed")
        return TaskStatus(
            state="failed",
            provider_state="missing_local_output",
            message="The completed Gemini interaction output is no longer available.",
            retryable=False,
            error_code="provider_output_missing",
        )

    @staticmethod
    def _video_bytes(data: dict) -> bytes | None:
        direct = data.get("output_video")
        if isinstance(direct, dict) and direct.get("data"):
            return _decode_video_data(direct["data"])
        for step in data.get("steps") or []:
            if step.get("type") != "model_output":
                continue
            for item in step.get("content") or []:
                if item.get("type") == "video" and item.get("data"):
                    return _decode_video_data(item["data"])
        return None

    def fetch(self, task_id: str, status: TaskStatus | None = None) -> VideoOutput:
        payload = self._outputs.pop(task_id, None)
        if payload is None:
            raise ProviderError(
                "The completed Gemini interaction output is no longer available.",
                retryable=False,
            )
        return VideoOutput(
            data=payload,
            mime="video/mp4",
            usage={"bytes": len(payload)},
            source_url_redacted="gemini:interaction",
        )


def _decode_video_data(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ProviderError(
            "Google returned video data in an unreadable format.",
            retryable=False,
        ) from exc

    def estimate_cost_cents(self, request: VideoRequest) -> tuple[int, str, bool]:
        settings = get_settings()
        rate = settings.veo_cost_cents_per_second
        if rate <= 0:
            return (0, "unmetered", True)
        return (int(round(rate * request.duration_s)), "configured_rate", True)

"""Google Veo 3.1 adapter (Gemini API).

Verified against <https://ai.google.dev/gemini-api/docs/veo> on 24 Oct 2026.

Submit   POST {base}/models/{model}:predictLongRunning      (x-goog-api-key)
         body: {"instances":[{"prompt", "image":{"inlineData":{...}},
                              "lastFrame":{"inlineData":{...}}}],
                "parameters":{"aspectRatio","durationSeconds","resolution",
                              "personGeneration","seed"}}
Poll     GET  {base}/{operation.name}          → done, response.generateVideoResponse…
Fetch    GET  {video.uri} with x-goog-api-key  (Veo output URLs require the key)

Documented constraints encoded below: 4/6/8 s clips, 24 fps, 720p/1080p/4k,
16:9 or 9:16, up to 3 reference images, first+last frame keyframes, natively
generated audio, output deleted after 2 days, SynthID watermark.

Honest limitation: Veo accepts ordinary images as anchors. It does not document
depth-map or hard geometric constraint input, so BuildVisual does not claim a
structural lock.
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

log = logging.getLogger("buildvisual.provider.veo")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class VeoProvider(VideoProvider):
    name = "veo"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 http: ProviderHttp | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.veo_model
        self.http = http or ProviderHttp()
        # Credentials are required to *call* the service, not to describe it. This
        # keeps capability reports and contract tests usable without secrets.
        if not self.api_key:
            log.warning("GEMINI_API_KEY is not configured; veo calls will fail")

    #: Provider failures translated into plain language for the engineer.
    FAILURE_MESSAGES = {
        "provider_failed": "The video service could not complete this scene. It will be retried.",
        "provider_empty_output": "The video service returned a scene with no footage in it.",
        "provider_safety_filter": (
            "The video service declined this scene under its safety policy. "
            "Try a different description for the property."
        ),
        "provider_output_expired": (
            "The video service had already deleted this scene (its outputs expire after "
            "two days), so it has to be generated again."
        ),
        "provider_quota": "The video service quota is exhausted. Add quota or wait and retry.",
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
            provider="veo",
            model=self.model,
            docs_url="https://ai.google.dev/gemini-api/docs/veo",
            verified_on="2026-10-24",
            supports_first_frame=True,
            supports_last_frame=True,
            max_reference_images=3,
            durations_s=(4, 6, 8),
            aspect_ratios=("16:9", "9:16"),
            resolutions=("720p", "1080p", "4k"),
            native_audio=True,
            prompt_max_chars=4000,
            output_ttl_hours=48.0,
            watermark="SynthID (invisible provenance watermark)",
            commercial_use="Permitted under Google's Generative AI API terms; review current terms.",
            cost_unit="per second of generated video",
            structural_control=(
                "Accepts a first-frame image and an optional last frame (interpolation) plus up to "
                "three reference images. These guide composition and subject appearance; they are not "
                "documented as hard geometric constraints."
            ),
            notes=[
                "8-second clips are required when using 1080p/4k or reference images.",
                "Generated videos are removed from Google's servers after 2 days.",
                "Request latency is documented as 11 s to 6 minutes at peak.",
                "Native audio is always generated.",
            ],
        )

    # -- request shape ---------------------------------------------------
    def _instance(self, request: VideoRequest) -> dict:
        instance: dict = {"prompt": request.prompt}
        if request.first_frame:
            instance["image"] = {
                "inlineData": {
                    "mimeType": request.first_frame_mime,
                    "data": base64.b64encode(request.first_frame).decode("ascii"),
                }
            }
        if request.last_frame and request.first_frame:
            instance["lastFrame"] = {
                "inlineData": {
                    "mimeType": request.last_frame_mime,
                    "data": base64.b64encode(request.last_frame).decode("ascii"),
                }
            }
        return instance

    def submit(self, request: VideoRequest, *, idempotency_key: str) -> SubmitResult:
        self._require_key()
        problems = self.validate(request)
        if problems:
            raise ProviderError("Veo request rejected before submission: " + "; ".join(problems),
                                retryable=False)
        payload: dict = {
            "instances": [self._instance(request)],
            "parameters": {
                "aspectRatio": request.aspect_ratio,
                "durationSeconds": str(request.duration_s),
                "resolution": request.resolution,
                "personGeneration": "allow_adult",
            },
        }
        if request.seed is not None:
            payload["parameters"]["seed"] = int(request.seed)
        if request.negative_prompt:
            # Veo has no separate negative field: fold constraints into the prompt
            # inside the adapter rather than leaking provider syntax upward.
            payload["instances"][0]["prompt"] = (
                f"{request.prompt}\n\nAvoid: {request.negative_prompt}."
            )

        url = f"{BASE_URL}/models/{self.model}:predictLongRunning"
        data = self.http.request_json(
            "POST", url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )
        operation = data.get("name")
        if not operation:
            raise ProviderError("Veo did not return an operation name.",
                                details={"keys": sorted(data)[:8]})
        return SubmitResult(
            task_id=operation, provider=self.name, model=self.model,
            raw={"operation": operation}, idempotency_key=idempotency_key,
        )

    def poll(self, task_id: str) -> TaskStatus:
        self._require_key()
        url = f"{BASE_URL}/{task_id}"
        data = self.http.request_json("GET", url, headers={"x-goog-api-key": self.api_key})
        done = bool(data.get("done"))
        if data.get("error"):
            error = data["error"]
            message = str(error.get("message", ""))[:300]
            code = error.get("code")
            retryable = code in (429, 500, 503) or "internal" in message.lower()
            return TaskStatus(
                state="failed", provider_state=f"error:{code}", message=message,
                retryable=retryable, error_code="provider_failed",
            )
        if not done:
            metadata = data.get("metadata") or {}
            hint = metadata.get("progressPercent")
            return TaskStatus(
                state="processing", provider_state="running",
                progress_hint=f"{hint}%" if hint is not None else None,
            )
        sample = self._first_sample(data)
        if sample is None:
            return TaskStatus(
                state="failed", provider_state="done_without_output",
                message="Veo reported completion without a video.",
                retryable=True, error_code="provider_empty_output",
            )
        blocked = sample.get("raiMediaFilteredReasons")
        if blocked:
            return TaskStatus(
                state="failed", provider_state="filtered",
                message="The provider's safety filter blocked this shot.",
                retryable=False, error_code="provider_safety_filter",
            )
        return TaskStatus(state="succeeded", provider_state="done")

    @staticmethod
    def _first_sample(data: dict) -> dict | None:
        response = data.get("response") or {}
        for key in ("generateVideoResponse", "generateVideoResponse_v2"):
            bucket = response.get(key)
            if isinstance(bucket, dict):
                samples = bucket.get("generatedSamples") or bucket.get("generatedVideos")
                if samples:
                    return samples[0]
        for key in ("generatedSamples", "generatedVideos"):
            if response.get(key):
                return response[key][0]
        return None

    def fetch(self, task_id: str, status: TaskStatus | None = None) -> VideoOutput:
        self._require_key()
        data = self.http.request_json(
            "GET", f"{BASE_URL}/{task_id}", headers={"x-goog-api-key": self.api_key}
        )
        sample = self._first_sample(data)
        if sample is None:
            if data.get("done"):
                raise ProviderError("Veo output is no longer available (expired).",
                                    details={"reason": "expired"})
            raise ProviderError("Veo has not finished this shot yet.", retryable=True)
        video = sample.get("video") or {}
        uri = video.get("uri")
        if not uri:
            inline = video.get("videoBytes")
            if inline:
                return VideoOutput(data=base64.b64decode(inline), mime="video/mp4",
                                   usage={"bytes": len(inline)})
            raise ProviderError("Veo returned no downloadable URI.")
        if "expired" in uri or "error" in uri.lower():
            raise ProviderError("Veo output link has expired.", details={"reason": "expired"})
        payload = self.http.download(uri, headers={"x-goog-api-key": self.api_key})
        return VideoOutput(
            data=payload,
            mime=video.get("mimeType", "video/mp4"),
            usage={"bytes": len(payload)},
            source_url_redacted="veo:generatedSample",
        )

    def estimate_cost_cents(self, request: VideoRequest) -> tuple[int, str, bool]:
        settings = get_settings()
        rate = settings.veo_cost_cents_per_second
        if rate <= 0:
            return (0, "unmetered", True)
        return (int(round(rate * request.duration_s)), "configured_rate", True)

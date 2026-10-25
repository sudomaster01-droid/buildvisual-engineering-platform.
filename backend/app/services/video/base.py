"""Canonical video-provider interface.

Blueprint p.26/p.58: the application talks to *this* interface only. Provider
request shapes, status vocabularies, error formats and cost units are translated
inside each adapter. A new provider is a new file, not a rewrite.

Canonical states an adapter must map into:

    ``submitted`` → accepted, no output yet
    ``processing`` → running
    ``succeeded`` → output available
    ``failed`` → terminal failure (may be retryable)
    ``expired`` → output existed but is no longer retrievable
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

from ...config import get_settings
from ...errors import ProviderError, ProviderRejected

CanonicalState = Literal["submitted", "processing", "succeeded", "failed", "expired"]


@dataclass(frozen=True)
class Capabilities:
    """What this provider can actually do — verified against documentation."""

    provider: str
    model: str
    docs_url: str
    verified_on: str
    supports_first_frame: bool
    supports_last_frame: bool
    max_reference_images: int
    durations_s: tuple[int, ...]
    aspect_ratios: tuple[str, ...]
    resolutions: tuple[str, ...]
    native_audio: bool
    prompt_max_chars: int
    output_ttl_hours: float | None
    watermark: str | None
    commercial_use: str
    cost_unit: str
    #: honest statement about geometric control
    structural_control: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self, *, hide_model: bool = False) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": None if hide_model else self.model,
            "docs_url": self.docs_url,
            "verified_on": self.verified_on,
            "supports_first_frame": self.supports_first_frame,
            "supports_last_frame": self.supports_last_frame,
            "max_reference_images": self.max_reference_images,
            "durations_s": list(self.durations_s),
            "aspect_ratios": list(self.aspect_ratios),
            "resolutions": list(self.resolutions),
            "native_audio": self.native_audio,
            "prompt_max_chars": self.prompt_max_chars,
            "output_ttl_hours": self.output_ttl_hours,
            "watermark": self.watermark,
            "commercial_use": self.commercial_use,
            "cost_unit": self.cost_unit,
            "structural_control": self.structural_control,
            "notes": list(self.notes),
        }


@dataclass
class VideoRequest:
    shot_id: str
    prompt: str
    negative_prompt: str = ""
    duration_s: int = 8
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    seed: int | None = None
    first_frame: bytes | None = None
    first_frame_mime: str = "image/png"
    last_frame: bytes | None = None
    last_frame_mime: str = "image/png"
    reference_images: list[tuple[bytes, str]] = field(default_factory=list)

    def has_image_anchor(self) -> bool:
        return self.first_frame is not None


@dataclass
class SubmitResult:
    task_id: str
    provider: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""


@dataclass
class TaskStatus:
    state: CanonicalState
    provider_state: str = ""
    message: str = ""
    retryable: bool = False
    error_code: str | None = None
    progress_hint: str | None = None


@dataclass
class VideoOutput:
    data: bytes
    mime: str = "video/mp4"
    duration_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    source_url_redacted: str | None = None


class VideoProvider(abc.ABC):
    """Contract every provider adapter implements."""

    name: str = "unknown"

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    def validate(self, request: VideoRequest) -> list[str]:
        """Problems that make this request impossible *before* spending money."""
        caps = self.capabilities()
        problems: list[str] = []
        if not request.prompt.strip():
            problems.append("prompt is empty")
        if len(request.prompt) > caps.prompt_max_chars:
            problems.append(
                f"prompt is {len(request.prompt)} chars, provider limit is {caps.prompt_max_chars}"
            )
        if request.duration_s not in caps.durations_s:
            problems.append(
                f"duration {request.duration_s}s is not supported "
                f"(allowed: {sorted(caps.durations_s)})"
            )
        if request.aspect_ratio not in caps.aspect_ratios:
            problems.append(f"aspect ratio {request.aspect_ratio} is not supported")
        if request.resolution not in caps.resolutions:
            problems.append(f"resolution {request.resolution} is not supported")
        if request.first_frame and not caps.supports_first_frame:
            problems.append("provider does not accept a first-frame image")
        if request.last_frame and not caps.supports_last_frame:
            problems.append("provider does not accept a last-frame image")
        if len(request.reference_images) > caps.max_reference_images:
            problems.append(
                f"{len(request.reference_images)} reference images supplied, "
                f"provider accepts {caps.max_reference_images}"
            )
        return problems

    @abc.abstractmethod
    def submit(self, request: VideoRequest, *, idempotency_key: str) -> SubmitResult: ...

    @abc.abstractmethod
    def poll(self, task_id: str) -> TaskStatus: ...

    @abc.abstractmethod
    def fetch(self, task_id: str, status: TaskStatus | None = None) -> VideoOutput: ...

    def cancel(self, task_id: str) -> bool:
        """Optional. Returns True when the provider confirmed cancellation."""
        return False

    def estimate_cost_cents(self, request: VideoRequest) -> tuple[int, str, bool]:
        """(cents, basis, is_estimate) — never reported as a metered charge."""
        return (0, "unmetered", True)

    # -- error helpers ---------------------------------------------------
    def normalise_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        settings = get_settings()
        message = str(exc)[:400]
        if "401" in message or "403" in message or "unauthor" in message.lower():
            return ProviderError(
                f"{self.name} rejected the credentials.",
                details={"provider": self.name, "hint": "check the API key"},
                retryable=False,
            )
        return ProviderError(f"{self.name} call failed: {message}",
                             details={"provider": self.name})

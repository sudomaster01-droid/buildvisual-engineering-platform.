"""Voice requirements → text.

Uses an OpenAI-compatible ``/v1/audio/transcriptions`` endpoint (multipart,
``model``, optional ``language``, ``response_format=verbose_json`` for
confidence). Only advertised in the UI when a key is configured; if it is not,
the endpoint returns a clear capability error rather than pretending to listen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from ...config import get_settings
from ...errors import CapabilityUnavailable, ProviderError

log = logging.getLogger("buildvisual.speech")

ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=60.0, pool=10.0)


@dataclass
class TranscriptResult:
    text: str
    language: str | None
    duration_seconds: float | None
    provider: str
    model: str
    confidence: float | None = None


def transcription_available() -> bool:
    return get_settings().transcription_available


def transcribe(data: bytes, filename: str, *, language: str | None = None) -> TranscriptResult:
    settings = get_settings()
    if not settings.transcription_available:
        raise CapabilityUnavailable(
            "Voice transcription is not configured on this deployment. "
            "Type the requirements instead, or set TRANSCRIPTION_API_KEY."
        )
    files = {"file": (filename or "audio.webm", data, "application/octet-stream")}
    form = {
        "model": settings.transcription_model,
        "response_format": "verbose_json",
    }
    if language:
        form["language"] = language.split("-")[0]
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {settings.transcription_key}"},
                files=files,
                data=form,
            )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise ProviderError(
            "The speech service could not be reached.", details={"error": str(exc)[:160]}
        ) from exc

    if response.status_code == 401 or response.status_code == 403:
        raise CapabilityUnavailable(
            "The speech service rejected the configured credentials."
        )
    if response.status_code == 429:
        raise ProviderError("The speech service is rate limiting requests.", retryable=True)
    if response.status_code >= 400:
        raise ProviderError(
            "The speech service could not transcribe this recording.",
            details={"status": response.status_code},
        )
    payload = response.json()
    text = (payload.get("text") or "").strip()
    if not text:
        raise ProviderError(
            "The recording produced no recognisable speech. Try again in a quieter room."
        )
    segments = payload.get("segments") or []
    confidence = None
    if segments:
        values = [s.get("avg_logprob") for s in segments if isinstance(s.get("avg_logprob"), (int, float))]
        if values:
            # Map mean log-probability to a rough 0–1 confidence for display only.
            confidence = round(min(1.0, max(0.0, 1.0 + sum(values) / len(values) / 4.0)), 3)
    return TranscriptResult(
        text=text[: get_settings().max_text_chars],
        language=payload.get("language"),
        duration_seconds=payload.get("duration"),
        provider="openai-compatible",
        model=settings.transcription_model,
        confidence=confidence,
    )

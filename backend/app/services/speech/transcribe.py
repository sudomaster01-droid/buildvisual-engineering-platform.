"""Local Faster-Whisper transcription; weights must already exist on disk."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ...config import get_settings
from ...errors import CapabilityUnavailable, ProviderError

log = logging.getLogger("buildvisual.speech")


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


@lru_cache(maxsize=2)
def _load_model(model_path: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise CapabilityUnavailable(
            "Local voice transcription is unavailable because Faster-Whisper is not installed."
        ) from exc
    try:
        return WhisperModel(model_path, device=device, compute_type=compute_type)
    except Exception as exc:
        raise ProviderError(
            "The local speech model could not be loaded.",
            details={"model_path": Path(model_path).name},
            retryable=False,
        ) from exc


def transcribe(data: bytes, filename: str, *, language: str | None = None) -> TranscriptResult:
    settings = get_settings()
    if not transcription_available():
        raise CapabilityUnavailable(
            "Local voice transcription is not ready. Install Faster-Whisper and its "
            "model weights on the server, or type the requirements instead."
        )
    model_path = settings.transcription_model_path.resolve()
    model = _load_model(
        str(model_path),
        settings.transcription_device,
        settings.transcription_compute_type,
    )
    suffix = Path(filename or "audio.webm").suffix.lower()
    if suffix not in {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".mp4"}:
        suffix = ".audio"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as audio_file:
            audio_file.write(data)
            audio_file.flush()
            segments, info = model.transcribe(
                audio_file.name,
                language=(language or "").split("-")[0] or None,
                vad_filter=True,
            )
            segments = list(segments)
    except Exception as exc:
        raise ProviderError(
            "The local speech model could not transcribe this recording.",
            retryable=False,
        ) from exc

    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise ProviderError(
            "The recording produced no recognisable speech. Try again in a quieter room."
        )
    values = [
        segment.avg_logprob
        for segment in segments
        if isinstance(segment.avg_logprob, (int, float))
    ]
    confidence = (
        round(min(1.0, max(0.0, 1.0 + sum(values) / len(values) / 4.0)), 3)
        if values else None
    )
    return TranscriptResult(
        text=text[: settings.max_text_chars],
        language=getattr(info, "language", None),
        duration_seconds=getattr(info, "duration", None),
        provider="faster-whisper",
        model=model_path.name,
        confidence=confidence,
    )

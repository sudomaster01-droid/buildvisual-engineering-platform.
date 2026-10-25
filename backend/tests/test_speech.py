from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


def test_transcription_uses_local_model_without_network(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.services.speech import transcribe as speech

    settings = get_settings()
    monkeypatch.setattr(settings, "transcription_enabled", True)
    monkeypatch.setattr(settings, "transcription_model_path", tmp_path)
    monkeypatch.setattr(settings, "transcription_device", "cpu")
    monkeypatch.setattr(settings, "transcription_compute_type", "int8")
    monkeypatch.setattr(speech, "transcription_available", lambda: True)
    speech._load_model.cache_clear()
    calls = {}

    class Model:
        def transcribe(self, audio_path, *, language, vad_filter):
            from pathlib import Path

            calls.update(
                bytes=Path(audio_path).read_bytes(),
                language=language,
                vad_filter=vad_filter,
            )
            return [
                SimpleNamespace(text=" A two-bedroom house ", avg_logprob=-0.4),
                SimpleNamespace(text=" with a flat roof. ", avg_logprob=-0.2),
            ], SimpleNamespace(language="en", duration=3.2)

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = lambda path, **kwargs: (
        calls.update(model_path=path, **kwargs) or Model()
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    result = speech.transcribe(b"local audio fixture", "brief.wav", language="en-US")

    assert result.text == "A two-bedroom house with a flat roof."
    assert result.provider == "faster-whisper"
    assert result.model == tmp_path.name
    assert result.language == "en"
    assert result.duration_seconds == 3.2
    assert calls["bytes"] == b"local audio fixture"
    assert calls["language"] == "en"
    assert calls["vad_filter"] is True
    assert calls["device"] == "cpu"
    assert calls["compute_type"] == "int8"
    speech._load_model.cache_clear()


def test_transcription_is_unavailable_without_local_model(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.errors import CapabilityUnavailable
    from app.services.speech import transcribe as speech

    settings = get_settings()
    monkeypatch.setattr(settings, "transcription_enabled", True)
    monkeypatch.setattr(settings, "transcription_model_path", tmp_path / "missing")
    assert not speech.transcription_available()

    try:
        speech.transcribe(b"audio", "brief.wav")
    except CapabilityUnavailable as exc:
        assert "model weights" in str(exc).lower()
    else:
        raise AssertionError("Missing local model weights must not be treated as available")

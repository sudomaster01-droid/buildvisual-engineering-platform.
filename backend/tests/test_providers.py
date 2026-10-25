"""Provider adapter contract.

Every adapter must satisfy the same interface so the pipeline never needs to know
which vendor is configured — and must declare honestly what it cannot do.
"""

from __future__ import annotations

import pytest

from app.services.video.base import VideoRequest
from app.services.video.factory import build_provider, is_external, provider_names


def _request(**overrides) -> VideoRequest:
    base = dict(
        shot_id="shot_01_aerial",
        prompt="Cinematic aerial reveal of a modern bungalow",
        duration_s=8,
        aspect_ratio="16:9",
        resolution="720p",
    )
    base.update(overrides)
    return VideoRequest(**base)


@pytest.mark.parametrize("name", provider_names())
def test_each_adapter_exposes_a_complete_capability_report(name):
    provider = build_provider(name)
    caps = provider.capabilities()
    assert caps.provider == name
    assert caps.model
    assert caps.durations_s
    assert caps.aspect_ratios
    assert caps.resolutions
    # Honesty: a provider must state whether it offers structural control.
    assert isinstance(caps.structural_control, str) and caps.structural_control.strip()
    data = caps.as_dict(hide_model=True)
    assert "model" not in data or not data.get("model")


@pytest.mark.parametrize("name", provider_names())
def test_adapter_implements_the_full_contract(name):
    provider = build_provider(name)
    for attribute in ("capabilities", "validate", "submit", "poll", "fetch",
                      "estimate_cost_cents", "normalise_error"):
        assert callable(getattr(provider, attribute)), attribute
    cents, basis, estimated = provider.estimate_cost_cents(_request())
    assert isinstance(cents, int) and cents >= 0
    assert basis in {"unmetered", "configured_rate", "provider_reported"}
    assert isinstance(estimated, bool)


def test_external_providers_declare_costs_and_local_does_not():
    local = build_provider("local_render")
    assert local.is_external is False
    assert local.estimate_cost_cents(_request())[0] == 0
    assert is_external("local_render") is False
    assert is_external("veo") and is_external("runway")


def test_validation_rejects_impossible_requests_before_spending_money():
    provider = build_provider("local_render")
    caps = provider.capabilities()
    assert provider.validate(_request()) == []
    assert provider.validate(_request(prompt="   ")), "empty prompt must be rejected"
    assert provider.validate(_request(duration_s=999)), "unsupported duration must be rejected"
    assert provider.validate(_request(aspect_ratio="9:16")), "unsupported ratio must be rejected"
    if not caps.supports_first_frame:
        problems = provider.validate(_request(first_frame=b"\x89PNG"))
        assert any("first-frame" in p for p in problems)


def test_capability_driven_requests_never_claim_unsupported_inputs(monkeypatch):
    """The pipeline must degrade to prompt-only guidance, not fail the shot."""
    from app.pipeline.steps.generate import _build_request

    class _Ctx:
        settings = type("S", (), {"resolved_video_provider": "veo",
                                  "video_aspect_ratio": "16:9"})()

    caps = build_provider("local_render").capabilities()  # no first/last frame support
    request = _build_request(_Ctx(), type("Shot", (), {
        "id": "s1", "duration_s": 5.0, "type": "aerial_reveal",
    })(), {"prompt": "hello world"}, type("Row", (), {"id": "plan1"})(), caps)
    assert request.first_frame is None and request.last_frame is None
    # Duration is clamped into what the provider accepts.
    assert request.duration_s in caps.durations_s


def test_unknown_provider_fails_loudly():
    from app.errors import CapabilityUnavailable

    with pytest.raises(CapabilityUnavailable):
        build_provider("no-such-vendor")


def test_capability_report_declares_no_structural_lock():
    from app.services.video.factory import capability_report

    report = capability_report()
    assert report["structural_lock_supported"] is False
    assert "reference" in report["structural_lock_note"].lower()


def test_veo_error_codes_map_to_user_readable_failures():
    from app.services.video.veo import VeoProvider

    provider = VeoProvider()
    mapped = {code: provider.describe_failure(code) for code in
              ("provider_failed", "provider_empty_output", "provider_safety_filter")}
    for code, message in mapped.items():
        assert message and message != code, code


def test_gemini_video_uses_interactions_api_and_reads_rest_video_output(monkeypatch):
    import base64

    from app.services.video.veo import VeoProvider

    class FakeHttp:
        def __init__(self):
            self.request = None

        def request_json(self, method, url, **kwargs):
            self.request = (method, url, kwargs)
            return {
                "id": "interaction-123",
                "status": "completed",
                "steps": [{
                    "type": "model_output",
                    "content": [{
                        "type": "video",
                        "mime_type": "video/mp4",
                        "data": base64.b64encode(b"clip-bytes").decode("ascii"),
                    }],
                }],
            }

    provider = VeoProvider(api_key="", model="gemini-omni-1.1-flash", http=FakeHttp())
    monkeypatch.setattr(provider, "_require_key", lambda: None)
    request = _request(first_frame=b"reference-png")
    submitted = provider.submit(request, idempotency_key="local-test-id")

    method, url, options = provider.http.request
    assert method == "POST"
    assert url.endswith("/v1beta/interactions")
    assert options["max_attempts"] == 1
    assert options["retry_on"] == ()
    body = options["json"]
    assert body["model"] == "gemini-omni-1.1-flash"
    assert body["response_format"] == {
        "type": "video",
        "aspect_ratio": "16:9",
        "resolution": "720p",
    }
    assert body["input"][0] == {
        "type": "image",
        "data": "cmVmZXJlbmNlLXBuZw==",
        "mime_type": "image/png",
    }
    assert provider.poll(submitted.task_id).state == "succeeded"
    output = provider.fetch(submitted.task_id)
    assert output.data == b"clip-bytes"
    assert provider.poll(submitted.task_id).state == "failed"


def test_retired_veo_preview_model_is_upgraded_to_current_gemini_model():
    from app.config import Settings

    settings = Settings(
        app_env="test",
        veo_model="veo-3.1-fast-generate-preview",
    )
    assert settings.veo_model == "gemini-omni-1.1-flash"
    assert settings.resolved_video_provider == "local_render"


def test_auto_video_provider_never_uses_configured_hosted_credentials():
    from app.config import Settings

    settings = Settings(
        app_env="test",
        video_provider="auto",
        gemini_api_key="not-a-real-key",
        runway_api_key="not-a-real-key",
    )
    assert settings.resolved_video_provider == "local_render"


def test_production_settings_reject_hosted_video_and_vision_providers():
    import pytest

    from app.config import Settings

    secure = {
        "app_env": "production",
        "app_secret": "s" * 40,
        "engineer_password": "not-a-default-password",
    }
    with pytest.raises(ValueError, match="Hosted video providers"):
        Settings(**secure, video_provider="gemini_omni")
    with pytest.raises(ValueError, match="Hosted interpretation providers"):
        Settings(**secure, llm_provider="openai")


def test_video_credentials_do_not_implicitly_enable_hosted_interpretation():
    from app.config import Settings

    settings = Settings(
        app_env="test",
        llm_provider="none",
        gemini_api_key="configured-video-credential",
        llm_api_key="configured-llm-credential",
    )
    assert settings.resolved_llm_provider == "none"
    assert not settings.capabilities()["drawing_interpretation"]


def test_generation_step_does_not_retry_every_scene_as_a_group():
    from app.pipeline.runner import STEP_BY_NAME

    assert STEP_BY_NAME["generate_clips"].max_attempts == 1


def test_rejected_unaccepted_video_request_is_not_retried_or_charged():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from app.errors import ProviderRejected
    from app.models import ProviderTask, UsageLedger
    from app.pipeline.steps.generate import _generate_one_shot

    class RejectingProvider:
        calls = 0

        def validate(self, request):
            return []

        def estimate_cost_cents(self, request):
            return (120, "configured_rate", True)

        def submit(self, request, *, idempotency_key):
            self.calls += 1
            raise ProviderRejected(
                "The video service rejected this request.",
                details={"provider_status": 404},
            )

    db = Mock()
    ctx = SimpleNamespace(
        job=SimpleNamespace(id="job-id", spent_cents=0),
        db=db,
        settings=SimpleNamespace(
            max_provider_attempts_per_shot=2,
            max_job_spend_cents=0,
            daily_spend_ceiling_cents=0,
        ),
        heartbeat=lambda: None,
    )
    provider = RejectingProvider()
    caps = SimpleNamespace(model="gemini-omni-1.1-flash")
    shot = SimpleNamespace(id="shot_01", type="approach", duration_s=8)

    outcome = _generate_one_shot(
        ctx, shot, _request(), provider, "gemini_omni", caps
    )

    assert provider.calls == 1
    assert outcome.clip_id is None
    assert outcome.qa_passed is False
    assert outcome.qa_report["provider_status"] == 404
    added = [call.args[0] for call in db.add.call_args_list]
    assert len(added) == 1 and isinstance(added[0], ProviderTask)
    assert not any(isinstance(row, UsageLedger) for row in added)

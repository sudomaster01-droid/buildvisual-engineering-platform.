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

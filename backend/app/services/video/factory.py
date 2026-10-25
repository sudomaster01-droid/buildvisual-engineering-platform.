"""Provider registry and capability reporting.

The orchestrator asks this module for "the configured provider". It is the only
place that knows which concrete adapter is in play, which is what makes the
provider swappable (blueprint p.26).
"""

from __future__ import annotations

import logging
from typing import Any

from ...config import get_settings
from ...errors import CapabilityUnavailable, ProviderError

log = logging.getLogger("buildvisual.provider")


def provider_names() -> list[str]:
    return ["gemini_omni", "runway", "local_render"]


def is_external(name: str) -> bool:
    return name in ("veo", "gemini_omni", "runway")


def build_provider(name: str | None = None, **kwargs):
    """Instantiate a provider adapter by name (default: configured)."""
    settings = get_settings()
    resolved = name or settings.resolved_video_provider
    if resolved in ("veo", "gemini_omni"):
        from .veo import VeoProvider

        return VeoProvider()
    if resolved == "runway":
        from .runway import RunwayProvider

        return RunwayProvider()
    if resolved == "local_render":
        from .local_render import LocalRenderProvider

        return LocalRenderProvider(
            mesh=kwargs.get("mesh"),
            shot_plan=kwargs.get("shot_plan"),
            sun_direction=kwargs.get("sun_direction", (-0.45, 0.55, 0.70)),
            # Interior shots must be lit like their reference frames, otherwise the
            # clip never matches the frame it was supposed to continue.
            lighting=kwargs.get("lighting"),
        )
    raise CapabilityUnavailable(f"Unknown video provider: {resolved}")


def capability_report() -> dict[str, Any]:
    """Credential-free capability matrix for the UI and diagnostics."""
    settings = get_settings()
    report: dict[str, Any] = {
        "configured_provider": settings.video_provider,
        "resolved_provider": settings.resolved_video_provider,
        "external": is_external(settings.resolved_video_provider),
        "available": {},
    }
    for name in provider_names():
        try:
            provider = build_provider(name)
            report["available"][name] = provider.capabilities().as_dict(hide_model=True)
        except ProviderError as exc:
            report["available"][name] = {"available": False, "reason": exc.user_message}
        except CapabilityUnavailable as exc:
            report["available"][name] = {"available": False, "reason": exc.user_message}
        except Exception as exc:  # pragma: no cover - defensive
            report["available"][name] = {"available": False, "reason": str(exc)[:200]}
    report["structural_lock_supported"] = False
    report["structural_lock_note"] = (
        "No configured provider documents hard geometric constraint input. Model-derived "
        "reference frames anchor composition; accuracy is verified by the QA stage rather "
        "than guaranteed by the provider."
    )
    return report

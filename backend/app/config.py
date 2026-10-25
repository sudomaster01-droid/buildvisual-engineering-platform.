"""Typed application configuration.

Every value that could plausibly differ between environments lives here and is
read from the environment (or a local `.env`). Nothing in this module invents a
credential: missing secrets resolve to ``None`` and the capability they gate is
reported as unavailable rather than silently faked.
"""

from __future__ import annotations

import json
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvName = Literal["development", "test", "production"]
VideoProviderName = Literal["auto", "veo", "runway", "local_render"]
LlmProviderName = Literal["none", "gemini", "openai", "anthropic"]
StorageBackendName = Literal["local", "s3"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Env var names are the upper-case field names."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- application -----------------------------------------------------
    app_env: EnvName = "development"
    app_name: str = "BuildVisual"
    app_secret: str = ""  # signs share tokens / CSRF; required in production
    public_base_url: str = "http://localhost:8000"
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/buildvisual.db"

    # ---- the single engineer account (seeded on first boot) ---------------
    #: The configured address is the source of truth for this one account.
    engineer_email: str = "engineer@example.com"
    engineer_password: str = ""
    engineer_name: str = "Engineer"
    #: When true (default), a changed ENGINEER_PASSWORD is applied to the stored
    #: account at startup. Environment variables are already root-equivalent on
    #: this host, so this cannot weaken security — it only stops an operator
    #: being locked out by a stale password hash after rotating the secret.
    engineer_password_sync: bool = True

    # ---- login protection -------------------------------------------------
    login_max_attempts: int = 8          # failures per window, per email + IP
    login_window_minutes: int = 15
    login_lockout_minutes: int = 15

    # ---- session / link policy ------------------------------------------
    session_ttl_hours: int = 24 * 14
    download_url_ttl_seconds: int = 300
    share_default_expiry_days: int = 7
    share_max_expiry_days: int = 90
    #: None -> derive from the public_base_url scheme
    cookie_secure_override: bool | None = None

    # ---- input limits ----------------------------------------------------
    max_upload_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_pdf_pages: int = 12
    max_audio_bytes: int = 20 * 1024 * 1024
    max_audio_seconds: int = 180
    max_text_chars: int = 4000
    allowed_drawing_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".png", ".jpg", ".jpeg", ".webp"]
    )
    allowed_audio_extensions: list[str] = Field(
        default_factory=lambda: [".webm", ".wav", ".mp3", ".m4a", ".ogg", ".mp4"]
    )

    # ---- language / vision models ---------------------------------------
    llm_provider: LlmProviderName = "none"
    llm_api_key: str = ""
    llm_model: str = ""  # provider-specific; left blank -> adapter default
    transcription_enabled: bool = True
    transcription_api_key: str = ""  # falls back to llm/openai key
    transcription_model: str = "whisper-1"

    # ---- video provider --------------------------------------------------
    video_provider: VideoProviderName = "auto"
    gemini_api_key: str = ""
    veo_model: str = "veo-3.1-fast-generate-preview"
    runway_api_key: str = ""
    runway_model: str = "gen4.5"
    runway_api_version: str = "2024-11-06"

    #: configured list prices (cents/second). VERIFY against current provider
    #: pricing; the ledger marks these as estimates, never as metered charges.
    veo_cost_cents_per_second: int = 15
    runway_cost_cents_per_second: int = 12

    video_aspect_ratio: str = "16:9"
    video_width: int = 1280
    video_height: int = 720
    video_fps: int = 24
    max_shots: int = 4
    shot_seconds: int = 8
    reference_width: int = 960
    reference_height: int = 540
    #: local model render (non-AI fallback): fps and internal scale before upscale
    local_render_fps: int = 12
    local_render_scale: float = 0.7
    transition_fade_seconds: float = 0.0

    # ---- budgets / retries ----------------------------------------------
    max_provider_attempts_per_shot: int = 2
    max_job_spend_cents: int = 1500
    daily_spend_ceiling_cents: int = 5000
    job_max_runtime_seconds: int = 45 * 60
    step_timeout_seconds: int = 20 * 60
    provider_poll_interval_seconds: int = 10
    max_active_jobs_per_user: int = 3
    max_jobs_per_hour: int = 12

    # ---- QA thresholds (versioned via QA_EVALUATOR_VERSION) --------------
    qa_evaluator_version: str = "1.0.0"
    qa_min_duration_ratio: float = 0.85
    qa_black_frame_ratio_max: float = 0.12
    qa_blur_variance_min: float = 18.0
    qa_exposure_clip_ratio_max: float = 0.30
    qa_reference_similarity_min: float = 0.18
    qa_blocking_severities: list[str] = Field(
        default_factory=lambda: ["critical"]
    )

    # ---- worker ----------------------------------------------------------
    run_worker_inline: bool = True
    worker_concurrency: int = 1
    worker_poll_interval_seconds: float = 1.0
    worker_lease_seconds: int = 120
    inline_worker_threads: int = 1

    # ---- storage ---------------------------------------------------------
    storage_backend: StorageBackendName = "local"
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_endpoint_url: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""

    # ---- media tooling ---------------------------------------------------
    ffmpeg_binary: str = ""
    ffprobe_binary: str = ""
    subprocess_timeout_seconds: int = 300

    # ---- retention -------------------------------------------------------
    retention_source_days: int = 365
    retention_intermediate_days: int = 14
    retention_video_days: int = 365

    # ---- derived helpers -------------------------------------------------
    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_path(cls, v):
        if isinstance(v, (str, Path)):
            return Path(str(v)).expanduser()
        return v

    @field_validator("allowed_drawing_extensions", "allowed_audio_extensions")
    @classmethod
    def _normalise_ext(cls, v: list[str]) -> list[str]:
        return sorted({e.lower() if e.startswith(".") else "." + e.lower() for e in v})

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.app_env == "production":
            if not self.app_secret or len(self.app_secret) < 32:
                raise ValueError(
                    "APP_SECRET must be set to at least 32 characters in production"
                )
            if not self.engineer_password:
                raise ValueError("ENGINEER_PASSWORD must be set in production")
            if self.storage_backend == "local":
                # Allowed, but it must be a persistent volume. Warned at startup.
                pass
        if self.video_width % 2 or self.video_height % 2:
            raise ValueError("video width/height must be even for H.264 encoding")
        if self.max_shots < 1 or self.max_shots > 12:
            raise ValueError("MAX_SHOTS must be between 1 and 12")
        if self.shot_seconds not in (4, 6, 8):
            raise ValueError("SHOT_SECONDS must be 4, 6 or 8 (Veo clip lengths)")
        return self

    # ---- capability reporting -------------------------------------------
    @property
    def resolved_secret(self) -> str:
        """Signing secret. Falls back to a per-process random value in dev."""
        if self.app_secret:
            return self.app_secret
        return _ephemeral_secret()

    @property
    def cookie_secure(self) -> bool:
        if self.cookie_secure_override is not None:
            return self.cookie_secure_override
        return self.public_base_url.startswith("https://")

    @property
    def llm_key(self) -> str:
        return self.llm_api_key

    @property
    def resolved_llm_provider(self) -> LlmProviderName:
        if self.llm_provider != "none":
            return self.llm_provider
        if self.gemini_api_key:
            return "gemini"
        return "none"

    @property
    def transcription_key(self) -> str:
        return self.transcription_api_key or self.llm_api_key

    @property
    def transcription_available(self) -> bool:
        return bool(self.transcription_enabled and self.transcription_key)

    @property
    def resolved_video_provider(self) -> str:
        """Which adapter the pipeline will actually use right now."""
        if self.video_provider == "auto":
            if self.gemini_api_key:
                return "veo"
            if self.runway_api_key:
                return "runway"
            return "local_render"
        if self.video_provider == "veo" and not self.gemini_api_key:
            return "local_render"
        if self.video_provider == "runway" and not self.runway_api_key:
            return "local_render"
        return self.video_provider

    def capabilities(self) -> dict:
        """Safe, credential-free capability report for the UI."""
        provider = self.resolved_video_provider
        return {
            "llm_provider": self.resolved_llm_provider,
            "drawing_interpretation": self.resolved_llm_provider != "none",
            "transcription": self.transcription_available,
            "video_provider": provider,
            "video_provider_is_external": provider in {"veo", "runway"},
            "generation_mode": (
                "external_ai" if provider in {"veo", "runway"} else "model_render"
            ),
            "accepted_drawing_types": self.allowed_drawing_extensions,
            "accepted_audio_types": self.allowed_audio_extensions,
            "max_upload_mb": round(self.max_upload_bytes / (1024 * 1024), 1),
            "max_audio_seconds": self.max_audio_seconds,
            "max_text_chars": self.max_text_chars,
            "shot_seconds": self.shot_seconds,
            "max_shots": self.max_shots,
            "resolution": f"{self.video_width}x{self.video_height}",
            "fps": self.video_fps,
            "structural_lock_supported": False,
            "structural_lock_note": (
                "Reference images guide the video model; no configured provider "
                "documents hard geometric constraint input, so structural accuracy "
                "is checked after generation rather than guaranteed."
            ),
        }


@lru_cache(maxsize=1)
def _ephemeral_secret() -> str:
    return secrets.token_urlsafe(48)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def reset_settings_cache() -> None:
    """Used by tests that need to re-read a patched environment."""
    get_settings.cache_clear()
    global settings
    settings = get_settings()


def describe_settings() -> dict:
    """Structured, redacted configuration snapshot for the diagnostics endpoint."""
    s = get_settings()
    return json.loads(s.model_dump_json(exclude={
        "app_secret", "engineer_password", "llm_api_key", "gemini_api_key",
        "runway_api_key", "transcription_api_key", "s3_access_key_id",
        "s3_secret_access_key",
    }))

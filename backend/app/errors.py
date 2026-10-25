"""Central error taxonomy.

Every error that can reach a user carries a stable machine code, an
HTTP status (when raised from the API) and a plain-language message that never
leaks internals or credentials.
"""

from __future__ import annotations

from typing import Any


class BuildVisualError(Exception):
    code: str = "internal_error"
    http_status: int = 500
    user_message: str = (
        "Something went wrong while processing this request. "
        "The job has been kept and can be retried."
    )
    retryable: bool = False

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message or self.user_message)
        self.message = message or self.user_message
        if message:
            # Instance-level message wins so callers see the specific reason.
            self.user_message = message
        self.details = details or {}
        if retryable is not None:
            self.retryable = retryable

    def to_public(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.user_message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }


# ---- input / validation ---------------------------------------------------
class ValidationError(BuildVisualError):
    code = "invalid_input"
    http_status = 400
    user_message = "The information supplied cannot be processed. Check the file or text and try again."


class UnsupportedFileType(ValidationError):
    code = "unsupported_file_type"
    user_message = "That file type is not supported."


class FileTooLarge(ValidationError):
    code = "file_too_large"
    user_message = "That file is larger than the configured limit."


class CorruptFile(ValidationError):
    code = "corrupt_file"
    user_message = "That file could not be read. It may be damaged or password-protected."


class MissingInput(ValidationError):
    code = "missing_input"
    user_message = "Add a drawing, or type or speak the house requirements first."


# ---- auth -----------------------------------------------------------------
class AuthError(BuildVisualError):
    code = "not_authenticated"
    http_status = 401
    user_message = "Please sign in to continue."


class PermissionDenied(BuildVisualError):
    code = "forbidden"
    http_status = 403
    user_message = "You do not have access to that resource."


class RateLimited(BuildVisualError):
    code = "rate_limited"
    http_status = 429
    user_message = "Too many requests. Wait a moment before trying again."
    retryable = True


class NotFound(BuildVisualError):
    code = "not_found"
    http_status = 404
    user_message = "That item could not be found."


class Conflict(BuildVisualError):
    code = "conflict"
    http_status = 409
    user_message = "That request conflicts with the current state."


# ---- capability / configuration ------------------------------------------
class CapabilityUnavailable(BuildVisualError):
    code = "capability_unavailable"
    http_status = 503
    user_message = "A required service is not configured on this deployment."
    retryable = False


# ---- pipeline -------------------------------------------------------------
class PipelineError(BuildVisualError):
    code = "pipeline_failed"
    http_status = 500


class InterpretationError(PipelineError):
    code = "interpretation_failed"
    user_message = (
        "The house could not be understood from the supplied input. "
        "Try a clearer drawing or describe the house in text."
    )


class GeometryError(PipelineError):
    code = "geometry_failed"
    user_message = (
        "The house description produced geometry that failed structural checks."
    )


class ProviderError(BuildVisualError):
    code = "provider_error"
    http_status = 502
    user_message = "The video service could not complete this request."
    retryable = True


class ProviderRejected(ProviderError):
    code = "provider_rejected"
    user_message = "The video service rejected the request as submitted."
    retryable = False


class ProviderRateLimited(ProviderError):
    code = "provider_rate_limited"
    user_message = "The video service is rate limiting requests. The job will retry."


class ProviderOutputExpired(ProviderError):
    code = "provider_output_expired"
    user_message = "The video service output expired before it could be retrieved."


class QualityGateFailed(PipelineError):
    code = "quality_gate_failed"
    user_message = "The generated video did not pass the required quality checks."


class BudgetExceeded(BuildVisualError):
    code = "budget_exceeded"
    http_status = 402
    user_message = "This job would exceed the configured spending limit."


class ClarificationRequired(BuildVisualError):
    code = "clarification_required"
    http_status = 200  # not an HTTP error: it is a legitimate job state
    user_message = "One detail needs to be confirmed before the model can be built."


class StepTimeout(PipelineError):
    code = "step_timeout"
    retryable = True

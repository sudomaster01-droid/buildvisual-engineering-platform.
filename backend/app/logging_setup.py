"""Structured logging with secret redaction.

Logs are JSON lines so they can be shipped and queried. A redaction pass runs
over every record: anything that looks like a credential, a signed query string
or a storage key is masked before it is written. The blueprint (p.82) makes
credential leakage an explicit threat.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

REDACTED = "***redacted***"

_SENSITIVE_KEYS = re.compile(
    r"(api[_-]?key|secret|password|token|authorization|cookie|credential|signature)",
    re.IGNORECASE,
)
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(x-goog-api-key|api[_-]?key|key)=([^&\s\"']+)", re.I), r"\1=" + REDACTED),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", re.I), r"\1" + REDACTED),
    (re.compile(r"(X-Amz-Signature=)[A-Fa-f0-9]+"), r"\1" + REDACTED),
    (re.compile(r"(X-Goog-Signature=)[A-Fa-f0-9]+"), r"\1" + REDACTED),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"), REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), REDACTED),
]


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively mask secrets in a log payload."""
    if _depth > 6:
        return "***depth***"
    if isinstance(value, dict):
        return {
            k: (REDACTED if _SENSITIVE_KEYS.search(str(k)) else redact(v, _depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v, _depth + 1) for v in value]
    if isinstance(value, str):
        out = value
        for pattern, repl in _PATTERNS:
            out = pattern.sub(repl, out)
        return out
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("job_id", "project_id", "step", "provider", "shot_id", "duration_ms", "event"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if isinstance(record.args, dict):
            payload["ctx"] = redact(record.args)
        return json.dumps(redact(payload), default=str)


class DevFormatter(logging.Formatter):
    """Readable console output for local development."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for key in ("job_id", "project_id", "step", "provider"):
            if hasattr(record, key):
                extras.append(f"{key}={getattr(record, key)}")
        return base + ("  [" + " ".join(extras) + "]" if extras else "")


def configure_logging(level: str | None = None, *, json_logs: bool | None = None) -> None:
    from .config import get_settings

    settings = get_settings()
    lvl = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    use_json = settings.app_env == "production" if json_logs is None else json_logs

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if use_json
        else DevFormatter("%(levelname)-7s %(name)-28s %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(lvl)
    for noisy in ("uvicorn.access", "httpx", "httpcore", "PIL"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(lvl)))
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def job_logger(job_id: str, step: str | None = None) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(
        logging.getLogger("buildvisual.pipeline"),
        {"job_id": job_id, "step": step or ""},
    )

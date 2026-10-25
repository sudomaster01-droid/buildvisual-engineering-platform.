"""Hardened outbound HTTP for provider adapters.

* allow-listed hosts only (blocks SSRF and metadata endpoints);
* bounded timeouts and a single retry policy for 429/5xx with jitter;
* response bodies are truncated before they can reach a log;
* every failure is translated into a normalised provider error.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from ...errors import ProviderError, ProviderRateLimited, ProviderRejected

log = logging.getLogger("buildvisual.http")

ALLOWED_HOSTS = {
    "generativelanguage.googleapis.com",
    "api.dev.runwayml.com",
    "storage.googleapis.com",
    "cdn.runwayml.com",
    "d3phaj0sisr2ct.cloudfront.net",
}

MAX_BODY_PREVIEW = 600
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0)


def check_host(url: str, *, extra_allowed: tuple[str, ...] = ()) -> None:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        raise ProviderError("provider returned a URL without a host")
    allowed = ALLOWED_HOSTS | set(extra_allowed)
    if host not in allowed:
        raise ProviderError(
            "refusing to contact a host outside the provider allow-list",
            details={"host": host},
        )


class ProviderHttp:
    def __init__(self, *, timeout: httpx.Timeout | None = None) -> None:
        self._timeout = timeout or DEFAULT_TIMEOUT

    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        max_attempts: int = 3,
        retry_on: tuple[int, ...] = (429, 500, 502, 503, 504),
        extra_allowed_hosts: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        check_host(url, extra_allowed=extra_allowed_hosts)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.request(method, url, headers=headers, json=json)
                if response.status_code in retry_on and attempt < max_attempts:
                    delay = _backoff(attempt)
                    log.warning(
                        "provider http retry in %.1fs",
                        delay,
                        extra={
                            "provider_event": "http_retry",
                            "status": response.status_code,
                            "attempt": attempt,
                        },
                    )
                    time.sleep(delay)
                    continue
                if response.status_code == 429:
                    raise ProviderRateLimited(
                        "The video service is rate limiting requests.",
                        details={"provider_status": 429},
                    )
                if response.status_code >= 500:
                    raise ProviderError(
                        "The video service is temporarily unavailable.",
                        details={"provider_status": response.status_code},
                    )
                if response.status_code == 401 or response.status_code == 403:
                    raise ProviderError(
                        "The video service rejected the API credentials.",
                        details={"provider_status": response.status_code},
                        retryable=False,
                    )
                if response.status_code >= 400:
                    raise ProviderRejected(
                        "The video service rejected this request.",
                        details={
                            "provider_status": response.status_code,
                            "detail": response.text[:MAX_BODY_PREVIEW],
                        },
                    )
                try:
                    return response.json()
                except ValueError as exc:
                    raise ProviderError("The video service returned malformed JSON.") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(_backoff(attempt))
                    continue
                raise ProviderError(
                    "Could not reach the video service.",
                    details={"error": str(exc)[:200]},
                ) from exc
        raise ProviderError(f"provider call failed: {last_error}")

    def download(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        max_bytes: int = 200 * 1024 * 1024,
        extra_allowed_hosts: tuple[str, ...] = (),
    ) -> bytes:
        check_host(url, extra_allowed=extra_allowed_hosts)
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 404 or response.status_code == 410:
                        raise ProviderError(
                            "The generated video is no longer available from the provider.",
                            details={"provider_status": response.status_code},
                        )
                    if response.status_code >= 400:
                        raise ProviderError(
                            "Could not download the generated video.",
                            details={"provider_status": response.status_code},
                        )
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes(1024 * 256):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ProviderError("generated video exceeds the download limit")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(
                "Downloading the generated video failed.",
                details={"error": str(exc)[:200]},
            ) from exc


def _backoff(attempt: int, base: float = 1.5, cap: float = 20.0) -> float:
    delay = min(cap, base ** attempt)
    return delay * (0.7 + random.random() * 0.6)

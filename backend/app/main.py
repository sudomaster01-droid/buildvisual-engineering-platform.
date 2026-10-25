"""FastAPI application: routing, security middleware, error mapping, lifespan.

Everything user-facing passes through here, so the cross-cutting rules live in
one file: security headers, CSRF on mutations, request ids in logs, consistent
error envelopes, and the single-page app shell.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .api import auth, capabilities, jobs, media, projects, rectify, share
from .config import get_settings
from .db import migrate, session_scope, utcnow
from .errors import AuthError, BuildVisualError
from .logging_setup import configure_logging
from .security import CSRF_COOKIE, CSRF_HEADER, ensure_engineer_account

log = logging.getLogger("buildvisual.app")
STATIC_DIR = Path(__file__).parent / "static"
PUBLIC_AUTH_MUTATIONS = {
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/password-reset",
    "/api/auth/password-reset/confirm",
}
PUBLIC_PAGES = {"/signin", "/signup", "/reset-password"}
PUBLIC_PATHS = PUBLIC_PAGES | {
    "/favicon.svg",
    "/healthz",
    "/readyz",
}


class SecurityMiddleware(BaseHTTPMiddleware):
    """Request id, security headers and double-submit CSRF protection."""

    async def dispatch(self, request: Request, call_next):
        from .errors import PermissionDenied
        from .security import constant_time_equals, current_user_optional

        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        path = request.url.path
        if (
            request.method == "GET"
            and path not in PUBLIC_PATHS
            and not path.startswith(("/api/", "/static/", "/share/"))
        ):
            with session_scope() as db:
                user = current_user_optional(request, db)
            if user is None:
                attempted = path + (f"?{request.url.query}" if request.url.query else "")
                return RedirectResponse(
                    f"/signin?next={quote(attempted, safe='')}", status_code=303
                )

        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if path.startswith("/api/"):
                csrf_cookie = request.cookies.get(CSRF_COOKIE, "")
                header = request.headers.get(CSRF_HEADER, "")
                is_public_auth = path in PUBLIC_AUTH_MUTATIONS
                if not is_public_auth and not request.cookies.get("bv_session"):
                    return _error_response(
                        AuthError("Please sign in to continue."), request_id
                    )
                if not csrf_cookie or not header or not constant_time_equals(header, csrf_cookie):
                    return _error_response(
                        PermissionDenied(
                            "Your session token was missing. Reload the page and try again."
                        ),
                        request_id,
                    )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(self), geolocation=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'self'; base-uri 'self'; "
            "form-action 'self'",
        )
        if request.url.path.startswith("/api"):
            response.headers.setdefault("Cache-Control", "no-store")
        elif request.url.path.startswith("/static/") or request.url.path in (
            "/", *PUBLIC_PAGES
        ):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response


def _error_response(exc: BuildVisualError, request_id: str) -> JSONResponse:
    payload = exc.to_public()
    payload["error"]["request_id"] = request_id
    if exc.details and exc.http_status < 500:
        payload["error"]["details"] = exc.details
    return JSONResponse(status_code=exc.http_status, content=payload)


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover - startup wiring
    configure_logging()
    migrate()
    with session_scope() as db:
        user = ensure_engineer_account(db)
        if user is not None:
            log.info("engineer account ready: %s", user.email)
    settings = get_settings()
    if settings.run_worker_inline:
        import threading

        from .pipeline.runner import Worker

        for index in range(max(1, settings.inline_worker_threads)):
            worker = Worker(worker_id=f"inline-{index}")
            thread = threading.Thread(target=worker.run_forever, name=f"bv-worker-{index}", daemon=True)
            thread.start()
        log.info("started %s inline worker thread(s)", settings.inline_worker_threads)
    yield
    log.info("shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="BuildVisual",
        version="1.0.0",
        docs_url="/api/docs" if settings.app_env != "production" else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )
    app.add_middleware(SecurityMiddleware)

    for module in (auth, capabilities, projects, jobs, rectify, media):
        app.include_router(module.router, prefix="/api")
    # Share links are public and live at the top level so the URL a client clicks
    # is short and does not look like an API endpoint.
    app.include_router(share.router)

    @app.exception_handler(BuildVisualError)
    async def _handle_buildvisual_error(request: Request, exc: BuildVisualError):
        request_id = getattr(request.state, "request_id", "-")
        level = logging.WARNING if exc.http_status < 500 else logging.ERROR
        log.log(level, "%s %s → %s: %s", request.method, request.url.path, exc.code, exc.message)
        if exc.code == "clarification_required":
            # Not an error: the job is parked waiting for one detail.
            return JSONResponse(
                status_code=200,
                content={
                    "status": "needs_input",
                    "question": exc.details.get("question") or exc.user_message,
                    "options": exc.details.get("options", []),
                    "reason": exc.details.get("reason"),
                    "supported": exc.details.get("supported", []),
                    "request_id": request_id,
                },
            )
        return _error_response(exc, request_id)

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        log.exception("unhandled error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Something went wrong. The failure has been logged.",
                    "retryable": True,
                    "request_id": request_id,
                }
            },
        )

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok", "time": utcnow().isoformat() + "Z"}

    @app.get("/readyz", include_in_schema=False)
    def readyz():
        from .services.video.factory import capability_report

        return {
            "status": "ready",
            "capabilities": {
                "video_provider": get_settings().resolved_video_provider,
                "provider_report": capability_report(),
            },
        }

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/signin", include_in_schema=False)
        @app.get("/signup", include_in_schema=False)
        @app.get("/reset-password", include_in_schema=False)
        def index():
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/favicon.svg", include_in_schema=False)
        def favicon():
            return FileResponse(STATIC_DIR / "favicon.svg")

    return app


app = create_app()

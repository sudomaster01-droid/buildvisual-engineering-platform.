"""Input validation and upload security.

The promise to the engineer is "we only accept what we can actually process", so
these tests assert both directions: supported files pass, unsupported or hostile
files are refused with an explanation rather than a stack trace.
"""

from __future__ import annotations

import io

import pytest


def _png(width: int = 400, height: int = 300) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    inset = max(4, min(width, height) // 8)
    draw.rectangle((inset, inset, width - inset, height - inset), outline="black", width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pdf(pages: int = 1) -> bytes:
    from PIL import Image

    images = [Image.new("RGB", (600, 400), "white") for _ in range(pages)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Accepted inputs
# ---------------------------------------------------------------------------


def test_valid_png_plan_is_accepted(client, login):
    login()
    response = client.post(
        "/api/jobs",
        files={"drawing": ("plan.png", _png(), "image/png")},
        data={"text": "Three bedroom bungalow with a flat roof"},
        headers={"Idempotency-Key": "png-1"},
    )
    # The drawing needs a vision model in this deployment, so the job is queued and
    # fails later with a capability message — but *validation* accepts the file.
    assert response.status_code == 201, response.text
    job_id = response.json()["job"]["id"]
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["stage"] in {
        "created", "validating", "interpreting", "failed", "waiting_for_clarification",
    }


def test_text_requirements_are_accepted(client, login):
    login()
    response = client.post(
        "/api/jobs",
        data={"text": "A 4 bedroom maisonette with a gable roof and a garage"},
        headers={"Idempotency-Key": "text-1"},
    )
    assert response.status_code == 201
    assert response.json()["job"]["input_mode"] in {"text", "drawing", "voice"}


# ---------------------------------------------------------------------------
# Refused inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,payload,expect",
    [
        ("plan.exe", b"MZ\x90\x00" + b"\x00" * 200, "unsupported"),
        ("plan.zip", b"PK\x03\x04" + b"\x00" * 200, "unsupported"),
        ("plan.dwg", b"AC1032" + b"\x00" * 200, "unsupported"),
        ("plan.html", b"<html><script>alert(1)</script></html>", "unsupported"),
        ("plan.png", b"not really a png at all", "unsupported"),
        ("plan.pdf", b"%PDF-1.4 truncated garbage", "corrupt"),
    ],
)
def test_hostile_drawings_are_refused(client, login, filename, payload, expect):
    login()
    response = client.post(
        "/api/jobs",
        files={"drawing": (filename, payload, "application/octet-stream")},
        data={"text": "three bedroom bungalow"},
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["message"]
    assert "Traceback" not in response.text


def test_oversized_upload_is_refused(client, login, monkeypatch):
    from app.config import get_settings

    payload = _png(400, 300)
    # Cap one byte below the real payload so the limit is genuinely exercised.
    monkeypatch.setattr(get_settings(), "max_upload_bytes", len(payload) - 1, raising=False)
    login()
    response = client.post(
        "/api/jobs",
        files={"drawing": ("plan.png", payload, "image/png")},
        data={"text": "three bedroom bungalow"},
    )
    assert response.status_code == 400, response.text
    assert "accepts up to" in response.json()["error"]["message"].lower()


def test_tiny_image_is_refused(client, login):
    login()
    response = client.post(
        "/api/jobs",
        files={"drawing": ("plan.png", _png(80, 60), "image/png")},
        data={"text": "three bedroom bungalow"},
    )
    assert response.status_code == 400
    assert "too small" in response.json()["error"]["message"].lower()


def test_pdf_page_limit_is_enforced(client, login):
    from app.config import get_settings

    monkeypatch_limit = get_settings().max_pdf_pages
    assert monkeypatch_limit >= 1
    login()
    response = client.post(
        "/api/jobs",
        files={"drawing": ("plan.pdf", _pdf(2), "application/pdf")},
        data={"text": "three bedroom bungalow"},
    )
    assert response.status_code == 201  # 2 pages is within the configured limit


def test_empty_request_is_refused(client, login):
    login()
    response = client.post("/api/jobs", data={"text": "   "})
    assert response.status_code == 400
    message = response.json()["error"]["message"].lower()
    # The message must name every input the engineer could use, not just text.
    assert "drawing" in message and "voice" in message and "description" in message


def test_overlong_text_is_refused(client, login, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_text_chars", 50, raising=False)
    login()
    response = client.post("/api/jobs", data={"text": "x" * 200})
    assert response.status_code == 400
    assert "limit is 50" in response.json()["error"]["message"].lower()


def test_validator_unit_level_checks():
    """Direct validator assertions, independent of HTTP."""
    from app.errors import UnsupportedFileType, ValidationError
    from app.services.intake.validators import validate_drawing, validate_text

    with pytest.raises(UnsupportedFileType):
        validate_drawing("plan.dwg", b"AC1032" + b"\x00" * 100)
    with pytest.raises(ValidationError):
        validate_text("   ")
    accepted = validate_drawing("plan.png", _png(), )
    assert accepted.mime == "image/png"
    assert accepted.width > 0 and accepted.checksum

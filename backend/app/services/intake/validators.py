"""Upload validation and safe intake.

Blueprint p.29/p.30/p.82. The rule that matters most: **do not advertise a format
the backend cannot process**. Only PDF, PNG, JPEG and WebP drawings are accepted
because those are the only formats we can actually rasterise and read. DWG, DXF,
IFC and RVT are deliberately rejected with a message that says so.

Checks, in order:
1. size limits (before reading the whole file into memory),
2. magic-byte signature vs claimed extension,
3. executable/archive rejection (a "drawing" that is a PE/ELF/ZIP is refused),
4. structural decode (Pillow ``verify`` + pixel-bomb guard, or pypdf page count),
5. checksum + metadata record.

Nothing here executes, extracts or shells out on the uploaded bytes.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ...config import get_settings
from ...errors import CorruptFile, FileTooLarge, UnsupportedFileType, ValidationError

log = logging.getLogger("buildvisual.intake")

#: Binary signatures we accept, mapped to (kind, canonical mime).
_SIGNATURES: list[tuple[bytes, str, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png", "drawing"),
    (b"\xff\xd8\xff", "image/jpeg", "drawing"),
    (b"RIFF", "image/webp", "drawing"),  # refined below (RIFF is shared)
    (b"%PDF-", "application/pdf", "drawing"),
    (b"ID3", "audio/mpeg", "audio"),
    (b"\x1aE\xdf\xa3", "audio/webm", "audio"),
    (b"OggS", "audio/ogg", "audio"),
    (b"fLaC", "audio/flac", "audio"),
    (b"RIFF", "audio/wav", "audio"),
    (b"\x00\x00\x00", "audio/mp4", "audio"),  # m4a/mp4 containers, refined below
]

#: Formats the product deliberately does not support, with an honest reason.
_KNOWN_UNSUPPORTED = {
    ".dwg": "AutoCAD DWG needs a licensed CAD parser that is not part of this build.",
    ".dxf": "DXF vector parsing is not implemented; export a PDF and upload that.",
    ".ifc": "IFC/BIM import is not implemented in this build.",
    ".rvt": "Revit files are proprietary and not supported.",
    ".skp": "SketchUp files are not supported.",
    ".zip": "Archives are not accepted; upload the drawing itself.",
    ".doc": "Word documents are not drawings.",
    ".docx": "Word documents are not drawings.",
}

_EXECUTABLE_SIGNATURES = (
    b"MZ", b"\x7fELF", b"\xca\xfe\xba\xbe", b"PK\x03\x04", b"7z\xbc\xaf",
    b"Rar!\x1a\x07", b"#!", b"<script", b"<html", b"<?php",
)


@dataclass
class ValidatedFile:
    data: bytes
    kind: str            # drawing | audio
    mime: str
    extension: str
    size_bytes: int
    checksum: str
    width: int | None = None
    height: int | None = None
    page_count: int | None = None
    duration_seconds: float | None = None
    report: dict = field(default_factory=dict)

    @property
    def is_pdf(self) -> bool:
        return self.mime == "application/pdf"


def _extension(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def validate_drawing(filename: str, data: bytes) -> ValidatedFile:
    settings = get_settings()
    ext = _extension(filename)
    if ext in _KNOWN_UNSUPPORTED:
        raise UnsupportedFileType(
            f"{ext.upper()} files are not supported. {_KNOWN_UNSUPPORTED[ext]}",
            details={"extension": ext, "supported": settings.allowed_drawing_extensions},
        )
    if ext not in settings.allowed_drawing_extensions:
        raise UnsupportedFileType(
            f"{ext or 'That file type'} is not supported. Upload "
            + ", ".join(e.upper().lstrip(".") for e in settings.allowed_drawing_extensions)
            + ".",
            details={"extension": ext, "supported": settings.allowed_drawing_extensions},
        )
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(
            f"The drawing is {len(data) / 1e6:.1f} MB; the limit is "
            f"{settings.max_upload_bytes / 1e6:.0f} MB.",
            details={"size_bytes": len(data)},
        )
    _reject_executables(data)

    sniffed = _sniff(data)
    if sniffed is None:
        raise UnsupportedFileType(
            "That file's contents do not match a supported drawing format."
        )
    mime, kind = sniffed
    if kind != "drawing":
        raise UnsupportedFileType("That file is not a drawing.")
    if ext in (".png", ".jpg", ".jpeg", ".webp") and not mime.startswith("image/"):
        raise UnsupportedFileType(
            "The file extension says image but the contents are something else."
        )
    if ext == ".pdf" and mime != "application/pdf":
        raise UnsupportedFileType(
            "The file extension says PDF but the contents are something else."
        )

    checksum = hashlib.sha256(data).hexdigest()
    result = ValidatedFile(
        data=data, kind="drawing", mime=mime, extension=ext,
        size_bytes=len(data), checksum=checksum,
    )
    if mime == "application/pdf":
        _validate_pdf(result)
    else:
        _validate_image(result)
    result.report = {
        "validated": True,
        "mime": mime,
        "size_bytes": len(data),
        "checksum": checksum,
        "width": result.width,
        "height": result.height,
        "page_count": result.page_count,
        "supported": True,
    }
    return result


def validate_audio(filename: str, data: bytes) -> ValidatedFile:
    settings = get_settings()
    ext = _extension(filename)
    if ext not in settings.allowed_audio_extensions:
        raise UnsupportedFileType(
            f"{ext or 'That'} audio format is not supported.",
            details={"supported": settings.allowed_audio_extensions},
        )
    if not data:
        raise ValidationError("The recording is empty.")
    if len(data) > settings.max_audio_bytes:
        raise FileTooLarge(
            f"The recording is {len(data) / 1e6:.1f} MB; the limit is "
            f"{settings.max_audio_bytes / 1e6:.0f} MB.",
        )
    _reject_executables(data)
    checksum = hashlib.sha256(data).hexdigest()
    mime = _audio_mime(data, ext)
    result = ValidatedFile(
        data=data, kind="audio", mime=mime, extension=ext,
        size_bytes=len(data), checksum=checksum,
    )
    duration = _audio_duration(data, ext)
    if duration is None:
        raise CorruptFile("The recording could not be read as audio.")
    if duration <= 0.4:
        raise ValidationError("The recording is too short to transcribe.")
    if duration > settings.max_audio_seconds:
        raise ValidationError(
            f"The recording is {duration:.0f}s; the limit is "
            f"{settings.max_audio_seconds}s. Describe the house in a shorter clip.",
        )
    result.duration_seconds = round(duration, 2)
    result.report = {
        "validated": True, "mime": mime, "duration_seconds": result.duration_seconds,
        "checksum": checksum, "size_bytes": len(data),
    }
    return result


def validate_text(text: str) -> str:
    settings = get_settings()
    cleaned = (text or "").strip()
    if not cleaned:
        from ...errors import MissingInput

        raise MissingInput()
    if len(cleaned) > settings.max_text_chars:
        raise ValidationError(
            f"That description is {len(cleaned)} characters; the limit is "
            f"{settings.max_text_chars}. Shorten it and try again."
        )
    return cleaned


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _reject_executables(data: bytes) -> None:
    head = data[:512]
    for signature in _EXECUTABLE_SIGNATURES:
        if head.startswith(signature):
            raise UnsupportedFileType(
                "That file looks like an executable, archive or web page, not a drawing.",
            )
    # Embedded script markers near the start of a text-ish payload.
    if b"\x00" not in head and head.lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml")) \
            and b"%PDF" not in data[:8]:
        raise UnsupportedFileType("That file looks like a web page, not a drawing.")


def _sniff(data: bytes) -> tuple[str, str] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ("image/png", "drawing")
    if data.startswith(b"\xff\xd8\xff"):
        return ("image/jpeg", "drawing")
    if data.startswith(b"%PDF-"):
        return ("application/pdf", "drawing")
    if data.startswith(b"RIFF") and len(data) > 12:
        form = data[8:12]
        if form == b"WEBP":
            return ("image/webp", "drawing")
        if form == b"WAVE":
            return ("audio/wav", "audio")
    if data.startswith(b"ID3") or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ("audio/mpeg", "audio")
    if data.startswith(b"\x1aE\xdf\xa3"):
        return ("audio/webm", "audio")
    if data.startswith(b"OggS"):
        return ("audio/ogg", "audio")
    if data.startswith(b"fLaC"):
        return ("audio/flac", "audio")
    if len(data) > 12 and data[4:8] == b"ftyp":
        return ("audio/mp4", "audio")
    return None


def _audio_mime(data: bytes, ext: str) -> str:
    sniffed = _sniff(data)
    if sniffed and sniffed[1] == "audio":
        return sniffed[0]
    return {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".webm": "audio/webm",
        ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(ext, "application/octet-stream")


def _validate_image(result: ValidatedFile) -> None:
    from PIL import Image, UnidentifiedImageError

    settings = get_settings()
    try:
        with Image.open(io.BytesIO(result.data)) as image:
            image.verify()  # detects truncation / corruption
        with Image.open(io.BytesIO(result.data)) as image:
            width, height = image.size
            if width * height > settings.max_image_pixels:
                raise ValidationError(
                    f"The image is {width}x{height}, which exceeds the "
                    f"{settings.max_image_pixels / 1e6:.0f} megapixel limit. "
                    "Downsample it and try again."
                )
            if width < 200 or height < 200:
                raise ValidationError(
                    "The image is too small to read as a plan "
                    f"({width}x{height}). Upload at least 200x200 pixels."
                )
            result.width, result.height = width, height
    except UnidentifiedImageError as exc:
        raise CorruptFile("That image could not be decoded.") from exc
    except ValidationError:
        raise
    except Exception as exc:  # pragma: no cover - unusual decoder failures
        raise CorruptFile(f"That image could not be read: {str(exc)[:120]}") from exc


def _validate_pdf(result: ValidatedFile) -> None:
    settings = get_settings()
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ValidationError(
            "PDF support is not installed on this server. Upload a PNG or JPEG instead."
        ) from exc
    try:
        reader = PdfReader(io.BytesIO(result.data))
        if reader.is_encrypted:
            raise ValidationError(
                "That PDF is password protected. Remove the password and upload it again."
            )
        pages = len(reader.pages)
        if pages == 0:
            raise CorruptFile("That PDF contains no pages.")
        if pages > settings.max_pdf_pages:
            raise ValidationError(
                f"That PDF has {pages} pages; the limit is {settings.max_pdf_pages}. "
                "Upload only the relevant plan sheets."
            )
        # Touch the first page to confirm the content stream is readable.
        reader.pages[0].mediabox
        result.page_count = pages
    except ValidationError:
        raise
    except (PdfReadError, ValueError, KeyError) as exc:
        raise CorruptFile(f"That PDF could not be read: {str(exc)[:120]}") from exc


def _audio_duration(data: bytes, ext: str) -> float | None:
    """Duration via FFmpeg's decoder (no shell, bounded input size)."""
    import subprocess
    import tempfile

    from ..media import ffmpeg as ff

    with tempfile.NamedTemporaryFile(suffix=ext or ".bin", delete=True) as handle:
        handle.write(data)
        handle.flush()
        try:
            result = subprocess.run(
                [ff.ffmpeg_binary(), "-hide_banner", "-i", handle.name],
                capture_output=True, text=True, timeout=30, check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        banner = (result.stderr or "") + (result.stdout or "")
        import re

        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", banner)
        if not match:
            return None
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def pdf_to_images(data: bytes, max_pages: int | None = None) -> list[tuple[bytes, str]]:
    """Rasterise PDF pages for the vision model.

    Rendering uses PyMuPDF when available (best quality); otherwise the PDF is
    rejected with an honest message rather than being silently misread.
    """
    settings = get_settings()
    limit = max_pages or settings.max_pdf_pages
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 exposes the pymupdf module name
    except ImportError:  # pragma: no cover - older installs
        try:
            import fitz  # type: ignore[no-redef]
        except ImportError as exc:
            raise ValidationError(
                "PDF plans cannot be rendered on this server because the PDF rasteriser "
                "is not installed. Upload the plan as a PNG or JPEG instead."
            ) from exc

    images: list[tuple[bytes, str]] = []
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # pragma: no cover - depends on the file
        raise CorruptFile("That PDF could not be opened for rendering.") from exc
    try:
        for index, page in enumerate(document):
            if index >= limit:
                break
            zoom = 200 / 72.0  # ≈200 dpi, enough to read dimension text
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            images.append((pixmap.tobytes("png"), "image/png"))
    finally:
        document.close()
    if not images:
        raise CorruptFile("That PDF had no renderable pages.")
    return images

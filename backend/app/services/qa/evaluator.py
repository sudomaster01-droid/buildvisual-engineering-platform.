"""Quality evaluation — deterministic technical + reference checks.

Blueprint p.64–p.66/p.71. Deliberate design choices:

* A **vector of signals**, not one unexplained score.
* Technical checks first (cheap, deterministic), then reference comparison.
* Every defect carries a segment, timestamp, severity and evidence so a human
  can adjudicate; every decision logs a reason code so false alarms can be
  analysed later.
* The evaluator is *measured*, not assumed: see ``qa/evaluation.py`` for the
  precision/recall harness that runs against a labelled set.
* It never claims geometric proof. A generated frame is a projection from a
  camera that the video model invented; similarity to a reference frame is
  evidence of drift, not proof of correctness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ...config import get_settings
from ..media import ffmpeg as ff

EVALUATOR_VERSION = "1.0.0"

SEVERITY_ORDER = {"none": 0, "info": 1, "minor": 2, "major": 3, "critical": 4}


@dataclass
class Defect:
    code: str
    severity: str
    message: str
    segment: str = ""
    timestamp_s: float | None = None
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "segment": self.segment,
            "timestamp_s": self.timestamp_s,
            "evidence": self.evidence,
        }


@dataclass
class QualityReport:
    target: str
    target_id: str
    passed: bool
    severity: str
    signals: dict
    defects: list[Defect]
    reason_code: str = ""
    evaluator_version: str = EVALUATOR_VERSION

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "target_id": self.target_id,
            "passed": self.passed,
            "severity": self.severity,
            "signals": self.signals,
            "defects": [d.as_dict() for d in self.defects],
            "reason_code": self.reason_code,
            "evaluator_version": self.evaluator_version,
        }


# ---------------------------------------------------------------------------
# Frame metrics
# ---------------------------------------------------------------------------


def luminance(frame: np.ndarray) -> np.ndarray:
    return (0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2])


def laplacian_variance(gray: np.ndarray) -> float:
    """Classic focus measure: high variance = crisp detail.

    Computed on a 0–255 scale (the convention for published thresholds) so the
    configured minimum is comparable with literature values.
    """
    gray = (gray * 255.0).astype(np.float32) if gray.max(initial=0.0) <= 1.0 else gray.astype(np.float32)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        kernel[0, 1] * padded[0:h, 1:w + 1]
        + kernel[1, 0] * padded[1:h + 1, 0:w]
        + kernel[1, 1] * padded[1:h + 1, 1:w + 1]
        + kernel[1, 2] * padded[1:h + 1, 2:w + 2]
        + kernel[2, 1] * padded[2:h + 2, 1:w + 1]
    )
    return float(laplacian.var())


def exposure_clip_ratio(gray: np.ndarray) -> tuple[float, float]:
    return (float((gray > 0.985).mean()), float((gray < 0.015).mean()))


def edge_mask(gray: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.maximum(gx, gy) > threshold


def reference_similarity(video_frame: np.ndarray, reference: np.ndarray) -> dict:
    """Compare a generated frame to a model-rendered reference frame.

    Two signals, because either alone is noisy:
      * ``edge_iou``   — structural overlap of boundaries (windows, roof lines);
      * ``tone_corr``  — normalised correlation of the luminance distribution.
    """
    size = (192, 108)
    vid = np.asarray(Image.fromarray((video_frame * 255).astype(np.uint8)).convert("L").resize(size), dtype=np.float32) / 255.0
    ref = np.asarray(Image.fromarray((reference * 255).astype(np.uint8)).convert("L").resize(size), dtype=np.float32) / 255.0

    ve, re_ = edge_mask(vid), edge_mask(ref)
    union = np.logical_or(ve, re_).sum()
    inter = np.logical_and(ve, re_).sum()
    edge_iou = float(inter / union) if union else 0.0

    a = vid.ravel() - vid.mean()
    b = ref.ravel() - ref.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    tone = float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0
    return {
        "edge_iou": round(edge_iou, 4),
        "tone_corr": round(tone, 4),
        "combined": round(0.65 * edge_iou + 0.35 * max(0.0, tone), 4),
    }


def continuity_delta(frames: list[np.ndarray]) -> float:
    """Mean absolute difference between consecutive samples (frozen-shot check)."""
    if len(frames) < 2:
        return 0.0
    deltas = [
        float(np.abs(frames[i] - frames[i - 1]).mean())
        for i in range(1, len(frames))
    ]
    return round(float(np.mean(deltas)), 5)


# ---------------------------------------------------------------------------
# Clip evaluation
# ---------------------------------------------------------------------------


def evaluate_clip(
    video_path: Path,
    *,
    clip_id: str,
    expected_duration_s: float | None = None,
    reference_first: Path | None = None,
    reference_last: Path | None = None,
    samples: int = 5,
) -> QualityReport:
    settings = get_settings()
    defects: list[Defect] = []
    signals: dict = {}

    info = ff.probe(video_path)
    signals["technical"] = info.as_dict()

    decodes, decode_errors = ff.verify_decodes(video_path)
    if not decodes:
        defects.append(Defect(
            "decode_error", "critical",
            "The clip does not decode cleanly.", segment=clip_id,
            evidence={"errors": decode_errors[:300]},
        ))

    if expected_duration_s:
        ratio = info.duration_s / expected_duration_s if expected_duration_s else 0
        signals["duration_ratio"] = round(ratio, 3)
        if ratio < 0.75:
            defects.append(Defect(
                "short_clip", "major",
                f"Clip is {info.duration_s:.1f}s but {expected_duration_s:.1f}s was planned.",
                segment=clip_id,
            ))

    if info.width < 640 or info.height < 360:
        defects.append(Defect(
            "low_resolution", "minor",
            f"Clip resolution {info.width}x{info.height} is below the delivery minimum.",
            segment=clip_id,
        ))

    # Frame sampling across the clip (avoid the very first/last frames which may
    # be fades and would inflate black-frame counts).
    sample_times = [
        info.duration_s * (0.12 + 0.76 * i / max(1, samples - 1)) for i in range(samples)
    ]
    frames: list[np.ndarray] = []
    for index, at in enumerate(sample_times):
        try:
            frames.append(ff.extract_gray_frame_array(video_path, at, 240, 135))
        except ff.MediaError:
            defects.append(Defect(
                "frame_extraction_failed", "major",
                "A frame could not be decoded for inspection.",
                segment=clip_id, timestamp_s=round(at, 2),
            ))
            continue

    if not frames:
        return _finalise("clip", clip_id, signals, defects, blocking=("critical",))

    luminance_values = [float(luminance(f).mean()) for f in frames]
    signals["luminance_mean"] = round(float(np.mean(luminance_values)), 4)
    signals["luminance_range"] = round(float(np.max(luminance_values) - np.min(luminance_values)), 4)

    black = [i for i, value in enumerate(luminance_values) if value < 0.05]
    signals["black_frame_ratio"] = round(len(black) / len(frames), 3)
    if signals["black_frame_ratio"] > settings.qa_black_frame_ratio_max:
        defects.append(Defect(
            "black_frames", "critical",
            "Too many sampled frames are essentially black.",
            segment=clip_id,
            evidence={"black_ratio": signals["black_frame_ratio"]},
        ))

    sharpness = [laplacian_variance(luminance(f)) for f in frames]
    signals["sharpness"] = round(float(np.median(sharpness)), 2)
    if signals["sharpness"] < settings.qa_blur_variance_min:
        defects.append(Defect(
            "blurred", "major",
            "Frames are soft/blurred beyond the delivery threshold.",
            segment=clip_id, evidence={"sharpness": signals["sharpness"]},
        ))

    highlights = [exposure_clip_ratio(luminance(f))[0] for f in frames]
    shadows = [exposure_clip_ratio(luminance(f))[1] for f in frames]
    signals["highlight_clip_ratio"] = round(float(np.mean(highlights)), 4)
    signals["shadow_clip_ratio"] = round(float(np.mean(shadows)), 4)
    if max(signals["highlight_clip_ratio"], signals["shadow_clip_ratio"]) > settings.qa_exposure_clip_ratio_max:
        defects.append(Defect(
            "exposure_clipped", "minor",
            "Highlights or shadows are clipped; check exposure consistency.",
            segment=clip_id,
            evidence={"highlight": signals["highlight_clip_ratio"],
                      "shadow": signals["shadow_clip_ratio"]},
        ))

    signals["continuity_delta"] = continuity_delta(frames)
    if signals["continuity_delta"] < 0.0009:
        defects.append(Defect(
            "frozen_motion", "major",
            "The camera appears static: consecutive samples are nearly identical.",
            segment=clip_id, evidence={"delta": signals["continuity_delta"]},
        ))
    if signals["continuity_delta"] > 0.30:
        defects.append(Defect(
            "motion_spike", "minor",
            "Large frame-to-frame change detected; possible cut or flash inside a shot.",
            segment=clip_id, evidence={"delta": signals["continuity_delta"]},
        ))

    # --- reference comparison -------------------------------------------
    if reference_first and reference_first.exists():
        similarity = _compare(video_path, sample_times[0], frames[0], reference_first)
        signals["reference_first"] = similarity
        if similarity["combined"] < settings.qa_reference_similarity_min:
            defects.append(Defect(
                "reference_drift", "major",
                "The first frame does not resemble the model-rendered reference.",
                segment=clip_id, timestamp_s=round(sample_times[0], 2),
                evidence=similarity,
            ))
    if reference_last and reference_last.exists():
        similarity = _compare(video_path, sample_times[-1], frames[-1], reference_last)
        signals["reference_last"] = similarity
        if similarity["combined"] < settings.qa_reference_similarity_min:
            defects.append(Defect(
                "reference_drift_end", "minor",
                "The final frame has drifted from the planned end reference.",
                segment=clip_id, timestamp_s=round(sample_times[-1], 2),
                evidence=similarity,
            ))

    return _finalise("clip", clip_id, signals, defects, blocking=("critical", "major"))


def _compare(video_path: Path, at: float, frame: np.ndarray, reference: Path) -> dict:
    try:
        with Image.open(reference) as image:
            ref_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return reference_similarity(frame, ref_array)
    except (OSError, ValueError):
        return {"edge_iou": 0.0, "tone_corr": 0.0, "combined": 0.0, "error": "reference unreadable"}


# ---------------------------------------------------------------------------
# Final film evaluation
# ---------------------------------------------------------------------------


def evaluate_final(
    video_path: Path,
    *,
    video_id: str,
    expected_duration_s: float | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_fps: int | None = None,
) -> QualityReport:
    settings = get_settings()
    defects: list[Defect] = []
    signals: dict = {}

    info = ff.probe(video_path)
    signals["technical"] = info.as_dict()

    decodes, decode_errors = ff.verify_decodes(video_path)
    if not decodes:
        defects.append(Defect("decode_error", "critical",
                              "The final file does not decode cleanly.",
                              evidence={"errors": decode_errors[:300]}))

    if expected_width and expected_height:
        if abs(info.width - expected_width) > 1 or abs(info.height - expected_height) > 1:
            defects.append(Defect(
                "resolution_mismatch", "major",
                f"Output is {info.width}x{info.height}, expected "
                f"{expected_width}x{expected_height}.",
            ))
    if expected_fps and info.fps and abs(info.fps - expected_fps) > 0.6:
        defects.append(Defect(
            "fps_mismatch", "minor",
            f"Output runs at {info.fps:.2f} fps, expected {expected_fps}.",
        ))
    if expected_duration_s:
        ratio = info.duration_s / expected_duration_s if expected_duration_s else 0
        signals["duration_ratio"] = round(ratio, 3)
        if ratio < settings.qa_min_duration_ratio:
            defects.append(Defect(
                "duration_short", "critical",
                f"Final film is {info.duration_s:.1f}s against a planned "
                f"{expected_duration_s:.1f}s.",
            ))
        if ratio > 1.25:
            defects.append(Defect(
                "duration_long", "minor",
                f"Final film is {info.duration_s:.1f}s against a planned "
                f"{expected_duration_s:.1f}s.",
            ))

    if info.size_bytes < 20_000:
        defects.append(Defect("file_tiny", "critical",
                              "The output file is implausibly small."))

    # Sample the whole film including the opening and closing beats.
    samples = 12
    durations = info.duration_s or 1.0
    frames: list[np.ndarray] = []
    for index in range(samples):
        at = durations * (index + 0.5) / samples
        try:
            frames.append(ff.extract_gray_frame_array(video_path, at, 240, 135))
        except ff.MediaError:
            continue

    if frames:
        lums = [float(luminance(f).mean()) for f in frames]
        signals["luminance_mean"] = round(float(np.mean(lums)), 4)
        signals["luminance_range"] = round(float(np.max(lums) - np.min(lums)), 4)
        signals["black_frame_ratio"] = round(
            sum(1 for value in lums if value < 0.045) / len(frames), 3
        )
        if signals["black_frame_ratio"] > settings.qa_black_frame_ratio_max:
            defects.append(Defect(
                "black_segments", "critical",
                "The film contains black or empty segments.",
                evidence={"black_ratio": signals["black_frame_ratio"]},
            ))
        signals["sharpness"] = round(float(np.median([laplacian_variance(luminance(f)) for f in frames])), 2)
        if signals["sharpness"] < settings.qa_blur_variance_min:
            defects.append(Defect(
                "film_soft", "major",
                "The film is soft throughout; check the source clips.",
                evidence={"sharpness": signals["sharpness"]},
            ))
        signals["continuity_delta"] = continuity_delta(frames)
        if signals["continuity_delta"] < 0.0008:
            defects.append(Defect(
                "film_static", "major",
                "Consecutive samples are nearly identical; the film may be frozen.",
            ))
        # Exposure consistency across the film: a large swing between shots is a
        # visible defect even when each shot is individually acceptable.
        if signals["luminance_range"] > 0.42:
            defects.append(Defect(
                "exposure_inconsistent", "minor",
                "Brightness varies strongly between shots.",
                evidence={"range": signals["luminance_range"]},
            ))

    report = _finalise("final", video_id, signals, defects,
                       blocking=tuple(settings.qa_blocking_severities) + ("major",))
    return report


def _finalise(
    target: str, target_id: str, signals: dict, defects: list[Defect],
    blocking: tuple[str, ...],
) -> QualityReport:
    severities = [d.severity for d in defects]
    worst = max(severities, key=lambda s: SEVERITY_ORDER.get(s, 0), default="none")
    threshold = min(
        (SEVERITY_ORDER.get(b, 4) for b in blocking), default=4
    )
    passed = all(SEVERITY_ORDER.get(s, 0) < threshold for s in severities)
    codes = [d.code for d in defects]
    if not defects:
        reason = "ok"
    elif passed:
        reason = "accepted_with_minor_defects"
    else:
        reason = codes[0] if codes else "quality_gate_failed"
    signals["defect_codes"] = codes
    signals["worst_severity"] = worst
    return QualityReport(
        target=target,
        target_id=target_id,
        passed=passed,
        severity=worst,
        signals=signals,
        defects=defects,
        reason_code=reason,
    )


def summarise(reports: list[QualityReport]) -> dict:
    """Aggregate view used by the job record and the UI."""
    defects = [d.as_dict() for r in reports for d in r.defects]
    passed = all(r.passed for r in reports)
    worst = max((r.severity for r in reports), key=lambda s: SEVERITY_ORDER.get(s, 0), default="none")
    return {
        "passed": passed,
        "worst_severity": worst,
        "evaluator_version": EVALUATOR_VERSION,
        "checked": len(reports),
        "failed": [r.target_id for r in reports if not r.passed],
        "defects": defects[:40],
        "reason_codes": sorted({r.reason_code for r in reports}),
    }

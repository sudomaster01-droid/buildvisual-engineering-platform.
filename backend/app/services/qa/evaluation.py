"""Measure the quality evaluator instead of assuming it works.

Blueprint p.71/p.97: thresholds must be set empirically and the checker's
precision/recall tracked, with failures kept in the evaluation set so the system
does not suffer from survivorship bias.

This module builds a small labelled corpus of *synthetic but real* video files
(with honest defect labels), runs the evaluator over them and reports per-class
precision/recall. It is used by ``tests/test_qa_evaluation.py`` and can be run
from the CLI:

    python -m app.services.qa.evaluation --out data/qa_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..media import ffmpeg as ff
from .evaluator import EVALUATOR_VERSION, evaluate_clip

#: label -> (generator, expected defect codes that must be found)
LABELS: dict[str, dict] = {
    "clean": {"expect": [], "severity": "none"},
    "black": {"expect": ["black_frames"], "severity": "critical"},
    "frozen": {"expect": ["frozen_motion"], "severity": "major"},
    "blurred": {"expect": ["blurred"], "severity": "major"},
    "overexposed": {"expect": ["exposure_clipped"], "severity": "minor"},
}

SOURCES = {
    # Deterministic synthetic sources: a moving pattern with structure so the
    # sharpness and continuity measures are meaningful.
    "clean": "testsrc2=size=640x360:rate=24:duration=4",
    "black": "color=c=black:size=640x360:rate=24:duration=4",
    "frozen": "color=c=0x224466:size=640x360:rate=24:duration=4",
    "blurred": "testsrc2=size=640x360:rate=24:duration=4",
    "overexposed": "color=c=0xFFFFFF:size=640x360:rate=24:duration=4",
}

EXTRA_FILTERS = {
    "blurred": "gblur=sigma=14",
    # A bright scene with light structure: most pixels sit above the highlight
    # clipping point, which is what an over-exposed architectural shot looks like.
    "overexposed": (
        "noise=alls=4:allf=t+u,"
        "drawbox=x=40:y=40:w=200:h=120:color=0xD8D2C6:t=fill"
    ),
    "frozen": "",
}

_GRAY_NOISE = "noise=alls=6:allf=t+u"


@dataclass
class Sample:
    label: str
    path: Path
    expected_codes: list[str]
    detected_codes: list[str]
    passed: bool
    severity: str


def build_corpus(directory: Path, *, per_label: int = 2) -> list[tuple[str, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
    corpus: list[tuple[str, Path]] = []
    for label, source in SOURCES.items():
        for index in range(per_label):
            path = directory / f"{label}_{index}.mp4"
            filters = EXTRA_FILTERS.get(label, "")
            if label == "clean":
                filters = _GRAY_NOISE
            args = [
                "-y", "-v", "error",
                "-f", "lavfi", "-i", f"{source},{_GRAY_NOISE}" if label != "frozen" else source,
            ]
            if filters:
                args += ["-vf", filters]
            args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                     "-pix_fmt", "yuv420p", str(path)]
            ff.run(args)
            corpus.append((label, path))
    return corpus


def run_evaluation(directory: Path | None = None, *, per_label: int = 2) -> dict:
    tmp_context = None
    if directory is None:
        tmp_context = tempfile.TemporaryDirectory(prefix="bv_qa_eval_")
        directory = Path(tmp_context.name)
    try:
        corpus = build_corpus(directory, per_label=per_label)
        samples: list[Sample] = []
        for label, path in corpus:
            report = evaluate_clip(path, clip_id=f"{label}:{path.stem}",
                                   expected_duration_s=4.0, samples=5)
            samples.append(Sample(
                label=label,
                path=path,
                expected_codes=list(LABELS[label]["expect"]),
                detected_codes=[d.code for d in report.defects],
                passed=report.passed,
                severity=report.severity,
            ))

        # Metric definitions (stated explicitly so the numbers cannot be
        # over-read):
        #   recall       — of the samples that genuinely contain this defect,
        #                  how many were flagged for it;
        #   precision    — of the samples flagged for this defect, how many are
        #                  "clean" footage (a deliberately strict definition:
        #                  co-occurring real defects on a defective clip are not
        #                  counted as false alarms).
        classes = sorted({code for spec in LABELS.values() for code in spec["expect"]})
        clean = [s for s in samples if s.label == "clean"]
        per_class: dict[str, dict] = {}
        for code in classes:
            positives = [s for s in samples if code in s.expected_codes]
            true_pos = sum(1 for s in positives if code in s.detected_codes)
            false_neg = len(positives) - true_pos
            flagged_clean = [s for s in clean if code in s.detected_codes]
            precision = (
                (len(clean) - len(flagged_clean)) / len(clean) if clean else None
            )
            recall = true_pos / len(positives) if positives else None
            per_class[code] = {
                "labelled_positive": len(positives),
                "true_positive": true_pos,
                "false_negative": false_neg,
                "flagged_clean": len(flagged_clean),
                "precision_on_clean": round(precision, 3) if precision is not None else None,
                "recall": round(recall, 3) if recall is not None else None,
            }

        false_alarm_rate = (
            sum(1 for s in clean if not s.passed) / len(clean) if clean else None
        )
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "samples": len(samples),
            "metric_definitions": {
                "recall": "flagged / labelled-positive samples for that defect class",
                "precision_on_clean": "share of clean samples NOT flagged with that defect",
                "false_alarm_rate_on_clean": "share of clean samples that failed the gate",
            },
            "per_class": per_class,
            "false_alarm_rate_on_clean": (
                round(false_alarm_rate, 3) if false_alarm_rate is not None else None
            ),
            "rows": [
                {
                    "label": s.label,
                    "file": s.path.name,
                    "expected": s.expected_codes,
                    "detected": s.detected_codes,
                    "passed": s.passed,
                    "severity": s.severity,
                }
                for s in samples
            ],
        }
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="Measure the BuildVisual QA evaluator")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--per-label", type=int, default=2)
    args = parser.parse_args()
    report = run_evaluation(per_label=args.per_label)
    payload = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f"wrote {args.out}")
    print(payload)


if __name__ == "__main__":  # pragma: no cover
    main()

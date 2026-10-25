# Implementation Map — blueprint requirement → code

Legend: **I** implemented, **T** tested, **P** partial, **U** unverified (needs credentials/evidence).

| # | Blueprint requirement (page) | Implementation | Status |
|---|---|---|---|
| A | Accepted input types, limits, signatures (29,30) | `services/intake/validators.py` | I/T |
| A | Original stored immutably, checksum, derived artifacts distinct (29,78) | `services/storage.py`, `api/uploads.py`, `source_artifacts` | I/T |
| A | Voice transcription as alternative input (31) | `services/speech/transcribe.py` | I/U (needs key) |
| A | Unique project/source/job ids, provenance preserved (A) | `db/models.py` | I/T |
| B | Typed house spec with units, provenance, confidence (34) | `services/spec/schema.py` | I/T |
| B | Drawing understanding → structured facts (32) | `services/interpret/` (vision adapters + rules) | I/U (needs key) |
| B | Merge drawing vs stated requirements, conflicts, no silent invention (35) | `services/interpret/merge.py` | I/T |
| B | Entropy/information-gain clarification (36,37) | `services/interpret/clarify.py` | I/T |
| C | Deterministic geometry builder, openings hosted on walls (40) | `services/modeling/geometry.py` | I/T |
| C | Topology/geometry validation before spend (42) | `services/modeling/validate.py` | I/T |
| C | Immutable, versioned, hashed model artifact + GLB (43) | `services/modeling/exporter.py`, `model_versions` | I/T |
| C | Lock semantics: styling cannot mutate geometry (43) | `services/design/compose.py` writes scene manifest only | I/T |
| D | Automatic materials/furniture/lighting/landscaping (44-47) | `services/design/*`, `services/design/library/*.json` | I/T |
| D | Deterministic placement constraints + clearance (46) | `services/design/placement.py` | I/T |
| E | Model-derived references: colour/depth/edge/segmentation (57) | `services/modeling/raster.py`, pipeline `render_references` | I/T |
| E | Capability-aware adapter; no false "hard lock" claim (57,26) | `services/video/capabilities.py`, UI disclosure | I/T |
| F | Hidden, versioned prompt composer, provider-independent (56) | `services/film/prompts.py` + `prompts/*.json` | I/T |
| F | Untrusted document text never becomes instructions (56) | `services/film/prompts.py` (sanitise + fenced data) | I/T |
| G | Provider interface: discover/validate/submit/status/fetch/error/cost/retry (26,58) | `services/video/base.py`, `veo.py`, `runway.py`, `local_render.py` | I/T |
| G | Real async external job, idempotent submit, URL expiry handling (58) | `services/video/*`, `pipeline/steps/generate.py` | I/U (needs key) |
| H | Cinematic sequence adapted to the actual house (50-53) | `services/film/shotplan.py`, `routing.py` | I/T |
| H | Room-graph route, collision-validated camera paths (51,53) | `services/film/routing.py`, `services/modeling/validate.py` | I/T |
| I | FFmpeg assembly, normalisation, poster, integrity (61,62) | `services/media/assembly.py`, `ffmpeg.py` | I/T |
| I | Technical + visual QA, vector score, reason codes (64-66,71) | `services/qa/` | I/T |
| I | Targeted regeneration within retry budget (67) | `pipeline/steps/generate.py` + `qa_gate` | I/T |
| J | Plain-language rectify, 4 categories (68,70) | `services/rectify/classify.py`, `apply.py` | I/T |
| J | Presentation keeps model version; structural rebuilds it (43,68) | `services/rectify/apply.py` | I/T |
| J | New immutable version, previous preserved (76) | `video_versions`, `api/videos.py` | I/T |
| K | Play/download/save/share/version history (17,63) | `api/videos.py`, `api/shares.py`, frontend ResultPanel | I/T |
| K | Private by default, revocable expiring share (78) | `api/shares.py`, `share_links` | I/T |
| 8 | Job state machine + failure paths + recovery (72) | `pipeline/state_machine.py`, `pipeline/recovery.py` | I/T |
| 8 | Idempotency, rate limits, spend caps (85) | `idempotency_keys`, `services/limits.py` | I/T |
| 8 | API contracts create/status/rectify (74-76) | `api/jobs.py`, `schemas.py` | I/T |
| 8 | Browser refresh recovers job (15) | `GET /api/jobs/active`, frontend bootstrap | I/T |
| 9 | Percentile latency, queue metrics, cost per video (87,89) | `api/diagnostics.py`, `usage_ledger` | I/P |
| 10 | Security: upload safety, authz, redaction, SSRF allowlist (82) | `services/intake/validators.py`, `security.py`, `logging.py` | I/T |
| 13 | Unit/integration/API/provider-contract/state/e2e tests (95,96) | `backend/tests/*` | I/T |
| — | Vector/interior semantic QA with vision model | Not implemented — deterministic CV only | **Unimplemented** |

## Honest gaps (also listed in the delivery report)

1. Drawing→spec accuracy is bounded by the vision model; no CAD/vector extraction of PDF linework.
   PDF pages are rasterised and read as images. DWG/IFC/RVT/DXF are **not** advertised or accepted.
2. No semantic (vision-LLM) QA pass; QA is deterministic image statistics + reference comparison.
3. Real provider calls and real cost figures are **unverified** here (no credentials in this
   environment). Adapter contract tests use recorded/mock transports.
4. Audio: provider-native audio is preserved; no music library, no licensed score.

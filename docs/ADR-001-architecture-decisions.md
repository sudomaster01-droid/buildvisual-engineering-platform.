# BuildVisual — Architecture Decision Records

Prepared 24 Oct 2026. Source: `uploads/BuildVisual_Software_Architecture_Blueprint_100_Pages.pdf`
(100 pages, read in full). This file records the consequential decisions, the evidence behind
them, and the fallback if the primary choice fails.

Labels follow the blueprint vocabulary: **AGREED** (decided in the blueprint), **PROPOSED**
(implementation recommendation), **VERIFIED** (checked against live documentation/code here),
**UNVERIFIED** (needs a real credentialed run).

---

## ADR-1 — Locked workflow (AGREED, non-negotiable)

```
Lock 3D model → complete design → generate structural references →
generate cinematic video → review & rectify → deliver
```

The model is the source of truth. Presentation corrections never mutate it. Structural
corrections create a **new** house-spec version *and* a new model version before any footage is
regenerated. This is enforced in code (`services/rectify/apply.py`, `pipeline/steps/*`) and
covered by tests, not just described.

---

## ADR-2 — Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | React 18 + TypeScript + Vite | Blueprint allows React/TS. Vite build is emitted to static files and served by the same FastAPI process, so there is **one deployable unit, one origin, no CORS surface** for a single-engineer product. |
| Backend | Python 3.13 + FastAPI + Pydantic v2 | Blueprint permits Python/FastAPI. The heavy parts (geometry, NUMPY rasterisation, image QA, PDF/vision orchestration) are Python-native; one backend language only. |
| ORM/DB | SQLAlchemy 2.0, SQLite default / PostgreSQL in compose & prod | Blueprint recommends Postgres. SQLite is used as the zero-infrastructure local default so the app starts reproducibly without Docker; the schema is portable and Postgres is a `DATABASE_URL` change. |
| Queue | Durable DB-backed queue (`pipeline_steps`) with leases + heartbeats | Avoids adding Redis/Valkey as a required dependency for a one-engineer product (Rule 3: no unnecessary infrastructure). Claiming is a conditional UPDATE (`WHERE state='queued' AND ...`), which is correct on both SQLite (`BEGIN IMMEDIATE`) and Postgres. Valkey/Celery is documented as the scale-up path, not a prerequisite. |
| Video assembly | FFmpeg 7.0.2 via `imageio-ffmpeg` | Blueprint names FFmpeg. The `imageio-ffmpeg` wheel ships a static binary, so no system package or GPU is needed; `FFMPEG_BINARY` overrides it. Subprocesses use argument arrays only (no shell). |
| 3D geometry | In-repo deterministic builder + NumPy z-buffer rasteriser | Blender/IfcOpenShell are the wrong shape here: we do not ingest IFC/RVT/DWG, we *construct* from a typed spec, and we must not depend on a GPU or a desktop app. A deterministic builder gives byte-stable geometry hashes and a pure-CPU renderer for references. Interface (`services/modeling/`) allows a Blender/IfcOpenShell backend later. |
| Vision/LLM | Provider-agnostic `interpretation` interface: Gemini, OpenAI, Anthropic | Blueprint requires model-specific prompts behind an interface. Adapters live in `services/interpret/`. |
| Speech | Whisper via OpenAI `/v1/audio/transcriptions` (same key as LLM path) | VERIFIED endpoint shape (multipart, `model`, `language`, `response_format`). Only advertised in the UI when configured. |
| Video AI | Gemini API **Veo 3.1** primary, **Runway Gen-4.5** secondary | Both VERIFIED against official docs on 24 Oct 2026 (below). |

---

## ADR-3 — Video provider selection (blueprint "provider selection gate")

**Required capability:** image-conditioned generation (model-derived frame as anchor) + async task
lifecycle + commercial-use output + server-side only credentials.

### Veo 3.1 via Gemini API — VERIFIED (primary)
Source: <https://ai.google.dev/gemini-api/docs/veo> (read 24 Oct 2026).

- Submit: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:predictLongRunning`,
  header `x-goog-api-key`, body `{"instances":[{"prompt","image":{"inlineData":{"mimeType","data"}},"lastFrame":…,"referenceImages":[…]}],"parameters":{"aspectRatio","durationSeconds","resolution","personGeneration","seed"}}`.
- Model ids: `veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview`.
- Poll: `GET /v1beta/{operation.name}` → `done`; output path
  `response.generateVideoResponse.generatedSamples[0].video.uri`, downloaded with `x-goog-api-key`.
- Limits (documented): 4/6/8 s clips, 24 fps, 720p/1080p/4k, 16:9|9:16, up to **3 reference
  images**, **first+last frame** control, natively generated audio, videos deleted **after 2 days**,
  request latency **11 s – 6 min**, SynthID watermark.
- Consequence in code: 8 s is mandatory with reference images/1080p; we therefore plan 8 s shots,
  download immediately, and never rely on the vendor URL later.

### Runway — VERIFIED (secondary)
Source: <https://docs.dev.runwayml.com/api/> (read 24 Oct 2026).

- `POST https://api.dev.runwayml.com/v1/image_to_video`, `Authorization: Bearer …`,
  `X-Runway-Version: 2024-11-06`, body `{model:"gen4.5"|…, promptText (≤1000 chars), promptImage
  (https URL or data URI ≤5 MB), position:"first", ratio, duration 2..10}` → `{id}`.
- Poll `GET /v1/tasks/{id}` → `PENDING|RUNNING|SUCCEEDED|FAILED`, `output:[url]`.
- Consequence: prompt text is capped at 1000 UTF-16 units → the composer emits a compact variant
  and the adapter validates it rather than truncating silently.

### What neither provider documents: depth maps / edge maps / geometry locks
Both accept **ordinary images** as anchors. Neither documents arbitrary depth-map or hard geometric
constraint input. Therefore BuildVisual does **not** claim a structural lock. Depth/edge/mask
references are generated as *internal* artifacts and used for **QA comparison** and as visual
anchors only. This is exposed honestly in the UI and the capability matrix
(`services/video/capabilities.py`).

### Fallback: `local_render`
When no video-provider credential is configured, the pipeline uses `local_render`, which renders
the locked model's camera plan with the in-repo rasteriser + FFmpeg. It is **not** an external API
call and never pretends to be one: the video version is tagged `provider=local_render`,
`generation_mode=model_render`, and the UI shows a persistent banner saying the footage was rendered
locally from the 3D model and is not AI-generated footage. This exercises assembly, QA, versions,
rectification and delivery genuinely, without faking a vendor success.

---

## ADR-4 — Model & reference generation without a GPU

`services/modeling/geometry.py` turns a validated `HouseSpec` into a triangle mesh (walls with real
openings, slabs, roof, stairs). `raster.py` is a NumPy z-buffer renderer producing, per camera:
shaded colour, **depth** (linear metres), **segmentation/object-id**, and **edge** maps. Everything
is CPU, deterministic, and hashable (`geometry_hash`). GLB export is written by an in-repo glTF 2.0
binary writer so the locked model is a real, inspectable artifact.

Determinism rule from the blueprint: styling/variation is seeded from the model version, never from
wall-clock randomness, so the same spec + builder version produces the same geometry hash.

---

## ADR-5 — Honest progress and state

No invented percentages. The job state machine has named stages; the UI shows stage completion
(done / current / pending) plus elapsed time measured from `job_events`. Every stage transition is
written transactionally with timestamps. Overdue non-terminal jobs are reaped by a recovery sweep
that re-queues or fails them with a reason code — a job can never sit "processing" forever.

---

## ADR-6 — Cost accounting

`usage_ledger` records **actual** reported cost when the provider returns usage, otherwise
`cost_basis='unmetered'` with the configured per-second/per-request list price marked
`estimated=True`. The UI/report never mixes the two. Budget caps are enforced *before* submission.

---

## ADR-7 — Degradation matrix (what happens with no credentials)

| Credential | Missing behaviour |
|---|---|
| `GEMINI_API_KEY` / `RUNWAY_API_KEY` | `local_render` provider, explicit banner; provider-marked metadata |
| LLM vision/interpretation key | Deterministic rules parser for typed/transcribed text; drawing interpretation unavailable with an explicit error (never a fabricated spec) |
| Speech key | Mic button hidden; "Speak" explains that transcription is not configured |

Drawing interpretation requires a vision model. Without one the job fails **before** any expensive
stage, with a clear message — it does not guess a house from pixels.

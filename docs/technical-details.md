# Audio Typewriter - Technical Details (Codebase Deep Dive)

## Scope and Method
- Scope: this document is based on all tracked source/config/docs files in the repository (`git ls-files`) plus runtime queue metadata (`data/asr_task_queue.json`).
- Included: backend API/services/tests, frontend code, scripts, package stubs, Docker compose, project metadata.
- Excluded from line-level explanation: generated lock/build artifacts (`apps/web/package-lock.json`, `apps/web/tsconfig.tsbuildinfo`), model binary assets/content under `models/`, and runtime audio assets under `audio_base/`.

## Repository Layout
- Root: `README.md`, `pyproject.toml`, `requirements.txt`, startup scripts.
- API: `apps/api/app/*` plus tests in `apps/api/tests/*`.
- Web: `apps/web/*` (React + TypeScript + Vite).
- Shared contracts: `packages/core/audio_typewriter_core/*`.
- Worker placeholder: `packages/workers/*`.
- Data/runtime: `data/asr_task_queue.json`, SQLite database (`data/audio_typewriter.sqlite3` at runtime).
- Infra: `infra/docker/docker-compose.yml`.

## Runtime Architecture
- Process model: single FastAPI process with an in-process background queue worker (`TaskQueueService`) started at import-time in router module.
- Persistence:
  - SQLite for core metadata/index (`audio_sources`, `word_occurrences`, `audio_bases`, `audio_base_files`, `mix_jobs`).
  - JSON task queue persistence in `data/asr_task_queue.json`.
  - Audio base files in `audio_base/<base_name>/`.
  - Output mixes in `artifacts/mixes/`.
- Audio toolchain:
  - FFmpeg/FFprobe for decode/probe/transcode/segment/concat/render.
  - Silero VAD for speech detection.
  - faster-whisper for ASR.
  - whisperx optional forced alignment.

## Project Metadata and Dependency Model
### `pyproject.toml`
- Python `>=3.11`.
- Main dependencies:
  - `fastapi`, `uvicorn[standard]`, `python-multipart`
  - `faster-whisper`, `whisperx`, `tqdm`, `silero-vad`
- Optional extras:
  - `worker`: `dramatiq[redis]`, `redis`
  - `dev`: `pytest`, `httpx`, `ruff`
- Test pythonpath includes `apps/api` and `packages/core`.
- Ruff: line-length 100, target py311.

### `requirements.txt`
- Mirrors core+test dependencies (no worker extras).

## API Application Layer
### `apps/api/app/main.py`
- Creates FastAPI app titled from settings (`settings.app_name`).
- Includes router under prefix `/api/v1`.
- On startup:
  - Ensures directories exist.
  - Initializes SQLite schema/migrations.
  - Optionally preloads ASR model when configured (`AT_ASR_PRELOAD_MODEL`).

### `apps/api/app/core/config.py` (`Settings`)
- Central env-backed configuration with defaults.
- Key directory settings:
  - `AT_DATA_DIR` -> `data`
  - `AT_DB_PATH` -> `data/audio_typewriter.sqlite3`
  - `AT_MIX_OUTPUT_DIR` -> `artifacts/mixes`
  - `AT_AUDIO_BASE_DIR` -> `audio_base`
  - `AT_TEMP_DIR` -> `temp`
  - `AT_ASR_MODEL_CACHE_DIR` -> `./models`
- ASR settings:
  - preferred device `AT_ASR_DEVICE` default `cuda`
  - compute types: CUDA `float16`, CPU `int8`
  - default model `large-v3`
  - beam size default 5
  - preload model flag
- Multipart limits:
  - max files default 20000
  - max fields default 4000
- Mixing settings:
  - default word gap ms 120
  - max candidates per token default 8
- Model name resolution:
  - supports tier mapping (`tiny`..`large`)
  - supports local path, cache-dir direct name, and `faster-whisper-*` prefixed local dir fallback
  - non-English currently rejected.

### Data Models (`apps/api/app/models.py`)
- Dataclass records (`slots=True`) for DB/service payloads:
  - `AudioSourceRecord`, `WordOccurrenceRecord`, `PhraseOccurrenceRecord`
  - `MixJobRecord`, `MixPlanItem`, `MixResult`, `IngestResult`
  - `AudioBaseRecord`, `AudioBaseFileRecord`, `AudioBaseStats`

### Request/Response Schemas (`apps/api/app/schemas.py`)
- Pydantic models for endpoint contracts:
  - health, ingest, mix, stitch, model download
  - audio base import and stats/list
  - local folder import request
- Validation highlights:
  - mix speed `gt=0`, gap `ge=0`
  - base names in schemas min/max lengths (service further enforces regex)

## Database Layer (`apps/api/app/db.py`)
### Schema (`SCHEMA_SQL`)
- `audio_sources`
  - PK: `source_audio_id`
  - fields include base, path, language/model/device metadata, timestamps.
- `word_occurrences`
  - PK autoincrement `id`
  - FK to `audio_sources` with cascade delete
  - stores token, normalized token, start/end/confidence, segment and word index.
  - indexes on `normalized_token` and `source_audio_id`.
- `mix_jobs`
  - status and output tracking for each mix/stitch job.
- `audio_bases`
  - metadata per base.
- `audio_base_files`
  - per clip metadata keyed by `source_audio_id`, sequence ordering, size/duration.
- WAL mode + foreign key pragma enabled.

### Access Patterns
- Upsert source/base/job tables.
- Replace semantics for base files and occurrences.
- Scoped search:
  - single-token search optionally filtered by `base_name` and limit.
  - phrase search performs self-joins on contiguous `word_index` within same source/segment.
- Utility ops:
  - overwrite cleanup (`clear_audio_base_for_overwrite`)
  - reASR purge from sequence (`purge_asr_index_from_sequence`)
  - base index summary and top word frequency.

## Text Normalization (`apps/api/app/text.py`)
- Token regex: `[A-Za-z0-9']+`.
- Normalize by lowercasing and removing non `[a-z0-9']`, then trimming edge apostrophes.
- `tokenize_sentence` returns normalized non-empty tokens.

## ASR Service (`apps/api/app/services/asr_service.py`)
### Core Behavior
- Runtime device resolution:
  - honors configured preference (`cpu|cuda|auto`)
  - CUDA availability checked via `ctranslate2.get_supported_devices()` then torch fallback.
- Model caching:
  - keyed by `(model_name, device, compute_type)` with thread-safe lock.
- Optional whisperx align-model cache keyed by `(language, device)`.

### Transcription Pipeline
- Calls faster-whisper `transcribe(..., word_timestamps=True, beam_size=...)`.
- Builds `WordOccurrenceRecord` list from word timestamps.
- Collects segment payload for optional forced alignment.
- Forced alignment:
  - skipped when whisperx missing.
  - if aligned word count mismatches ASR count, keeps original timestamps.
  - otherwise replaces start/end and confidence when available.
- Failure strategy:
  - if CUDA fails, auto retry on CPU.
  - if both fail, raises `ASRTranscriptionError` with runtime event history.

### Additional Features
- `download_model()` warms model and returns resolved runtime metadata.
- `preload_default_model_if_configured()` startup preload hook.
- `ingest()` wraps transcribe and returns source record + occurrences + summary.

## Audio Base Service (`apps/api/app/services/audio_base_service.py`)
### Base and File Management
- Base name validation regex: `^[a-zA-Z0-9_-]{1,64}$`.
- Allowed import extensions: `.wav`, `.mp3`.
- Maintains per-base metadata JSON (`base_metadata.json`).

### FFmpeg/FFprobe Utilities
- Duration probing via ffprobe.
- Audio normalization for VAD to mono 16k PCM.
- Segment export (`-ss`/`-to`) and full clip transcode.
- PCM-safe concat implemented through Python `wave` streaming to avoid drift/duplication.

### VAD Logic
- Silero model lazy-loaded and cached.
- Detects speech timestamps with small head/tail margins.
- Supports split-only segmentation by speech boundaries while preserving gap intervals.
- If no speech segments, fallback to full-duration segment.

### Import/Stage Workflows
- `stage_vad_sources(...)` for multipart files:
  - filters and sorts uploads
  - copies into `temp/vad_jobs/<task_id>/sources`
  - writes `manifest.json`
  - emits preprocess progress callback events.
- `stage_vad_sources_from_folder_path(...)` for local folder scan (recursive).

### Clip Export Modes
- `export_sources_as_single_base_clip(...)`: concatenates all staged sources into one `base.wav` (sequence 1).
- `split_source_file_into_base_clips(...)` / `_split_upload_into_speech_clips(...)`: split by VAD boundaries to numbered clips.
- `append_*` helpers for extending existing clips.

## Index Service (`apps/api/app/services/index_service.py`)
- Upserts source and replaces occurrences atomically per source.
- Token search normalizes each query token and applies configured candidate cap.

## Mixing Service (`apps/api/app/services/mixing_service.py`)
### Planning
- Tokenizes sentence via normalized token stream.
- Phrase-first strategy:
  - attempts phrase lengths 4 -> 2 using contiguous phrase search in DB.
  - falls back to single-token search.
- Candidate selection policy:
  - first token/phrase: highest confidence (random tie-break)
  - subsequent: prefer same source with nearest timestamp to previous start, then confidence, then random tie-break.

### Rendering
- Builds ffmpeg command with one input per selected item.
- Per-segment filter: `atrim + asetpts + aresample + mono format`.
- Optional inter-word gaps via `anullsrc` nodes and concat chain.
- Speed control through generated `atempo` chain that decomposes out-of-range multipliers into 0.5..2.0 factors.
- Output codec PCM s16le WAV.

### Job Recording
- Persists `mix_jobs` queued/completed/failed states.
- On missing tokens, aborts with explicit preview message and marks job failed.
- Manual stitch path validates segment timestamps and preserves input order.

## Task Queue Service (`apps/api/app/services/task_queue_service.py`)
### Design
- In-process single worker thread + condition variable.
- Queue persisted to JSON file after each mutation.
- Task stages: `preprocess`, `vad`, `asr`.
- Statuses: `queued`, `running`, `paused`, `completed`, `failed`, `discarded`.

### QueueTask Fields
- Tracks:
  - ASR counters (`processed_files`, `next_sequence_number`, `asr_last_completed_sequence`, `token_count`)
  - VAD progress seconds and source indices
  - overwrite cleanup counters
  - elapsed timers for VAD/ASR and running-since stamps
  - human-readable `last_error` and `last_event`.

### Lifecycle Highlights
- On service load:
  - reads JSON queue
  - converts any persisted `running` tasks to `paused`
  - rewinds ASR checkpoint for resume consistency.
- Import enqueue path:
  - create preprocess task
  - update preprocess progress
  - activate VAD stage when staging complete.
- VAD run path (`_run_vad_task`):
  - loads manifest, exports `base.wav`, detects speech segments, persists base rows and metadata, activates ASR stage.
- ASR run path (`_run_task`):
  - iterates `audio_base_files` from checkpoint
  - ingests ASR + index
  - handles pause/discard boundaries
  - on ASR error: pauses task and records runtime event.
- Resume ASR:
  - rewinds to last completed sequence strategy
  - purges DB index from resume point before reprocessing.
- Delete task:
  - removes all tasks sharing same base
  - purges DB/base storage and temp VAD job dirs.
- Shutdown prep:
  - marks shutdown requested
  - pauses running tasks and flushes queue state.

## API Routing (`apps/api/app/api/routes.py`)
### Service Singletons
- Module-level instances:
  - `SQLiteDatabase`, `ASRService`, `AudioBaseService`, `IndexService`, `MixingService`, `TaskQueueService`.

### Helper Flows
- `_run_audio_base_import(...)` and local-folder variant:
  - validate base
  - handle overwrite cleanup and unfinished task discard
  - create preprocess task
  - stage VAD sources
  - activate VAD stage and return immediate queue response.
- Streaming import endpoints use NDJSON via `StreamingResponse` + worker thread + queue.

### Endpoints
- `GET /health`
  - returns configured and resolved ASR runtime + last-used runtime.
- `POST /audio-bases/import`
  - multipart import.
- `POST /audio-bases/import/local`
  - local folder path import.
- `POST /audio-bases/import/stream`
- `POST /audio-bases/import/local/stream`
  - NDJSON event stream variants.
- `GET /tasks`
- `POST /tasks/{task_id}/pause`
- `POST /tasks/{task_id}/resume`
- `DELETE /tasks/{task_id}`
  - queue control + cleanup.
- `POST /system/exit`
  - queue flush and delayed process hard-exit.
- `GET /audio-bases`
- `GET /audio-bases/{base_name}/stats`
- `POST /audio-bases/{base_name}/reasr`
  - purge existing index and enqueue ASR rebuild.
- `POST /models/download`
- `POST /ingest`
  - direct one-off ingest by path.
- `POST /mix`
- `POST /mix/stitch`

## Frontend (`apps/web`)
### Tooling
- React 18 + TypeScript + Vite (`apps/web/package.json`).
- Vite dev server port 5173; proxies `/api` to backend 8000 (`vite.config.ts`).
- Strict TS config with `noEmit`, `strict`, bundler resolution.

### API Client (`apps/web/src/api.ts`)
- Defines TS DTOs mirroring backend schemas.
- Implements REST wrappers for health, base list/stats, import, task operations, reASR, mix, stitch, and system exit.
- Import stream parsers consume NDJSON from `ReadableStream`, dispatch event callback, and require terminal `complete` event.
- `ApiError` supports status-aware handling for stats 404 pending case.

### UI (`apps/web/src/App.tsx`)
- Single-page app with two tabs:
  - Workbench: import base, select base, trigger reASR, submit mix.
  - Tasks: queue monitor and controls.
- Bilingual UI (`zh`/`en`) with runtime text toggle.
- Polling:
  - `refreshTasks()` every 1200 ms.
  - diff-aware log append using previous task snapshot ref.
- Import UX:
  - local folder path input
  - stream event driven progress/logging
  - cap logs to 500 lines.
- Task controls:
  - pause/resume/delete buttons by status.
  - separate VAD seconds progress bar vs ASR file progress bar.
- Exit button triggers backend `/system/exit`.

### Entry and HTML
- `src/main.tsx` mounts `<App />` in strict mode.
- `index.html` is standard Vite root document.

## Tests (`apps/api/tests`)
### `test_health.py`
- Basic unit test for `health()` response status.

### `test_pipeline.py`
- Large integration-style unittest suite (1000+ lines) covering:
  - ASR runtime fallback and model resolution behavior.
  - Model cache reuse semantics.
  - word timestamp preservation behavior.
  - VAD segmentation and split-only boundary behavior.
  - local-folder staging behavior.
  - queue pause/resume/rewind/purge/shutdown behavior.
  - mixing candidate strategy and ffmpeg filter graph assertions.
  - overwrite behavior and cleanup semantics.
  - metadata JSON writing and delete-task purge behavior.
- Uses temp dirs and method monkey-patching heavily for deterministic tests.

## Shared Core Package (`packages/core/audio_typewriter_core`)
- Protocol interfaces:
  - `ASREngine`, `IndexStore`, `AudioMixer`.
- Lightweight dataclasses:
  - `WordOccurrence`, `MixPlanItem`.
- Current API app does not directly import these interfaces in service logic; package is scaffolded for future decoupling.

## Worker Package (`packages/workers`)
- Placeholder worker entrypoint `process_mix_job(job_id, sentence)`.
- Returns queued `JobResult` without actual processing.

## Scripts and Startup
### `scripts/bootstrap.ps1`
- Creates venv, upgrades pip, installs `requirements.txt`, optional web deps install.

### `scripts/start_all.ps1`
- Validates Python and npm.
- Optional `-InstallIfMissing` bootstrap.
- Auto-runs `npm install` if missing.
- Starts API and Web in separate PowerShell windows.
- Sets `PYTHONPATH` to include `apps/api` and `packages/core`.
- Polls API health before launching web.

### Root launchers
- `start.bat`
  - Windows CMD launcher for API+Web with optional `--dry-run`.
  - Includes health wait loop and browser auto-open.
- `start.ps1`
  - thin wrapper calling `scripts/start_all.ps1`.
  - Note: current file begins with `1$ErrorActionPreference = "Stop"` (leading `1` appears unintended).

### Misc utility scripts
- `scripts/download_model.py`: CLI wrapper around `ASRService.download_model()`.
- `scripts/remove_vad_suffix_files.ps1`: recursive cleanup for `*_vad.wav` files.

## Infra
### `infra/docker/docker-compose.yml`
- Defines `api` service on `python:3.11-slim` with bind mount and inline install+uvicorn command.
- Defines `redis:7` service.
- Current API code uses in-process queue, so Redis is pre-provisioning for future worker architecture.

## Runtime Data and State Files
### `data/asr_task_queue.json`
- Persistent task snapshots with multiple historical completed tasks and one active preprocess task.
- Demonstrates real fields used by queue state machine:
  - overwrite flags
  - elapsed timings
  - VAD/ASR counters and events.

### SQLite file (`data/audio_typewriter.sqlite3`)
- Runtime DB file expected by settings; schema controlled by `SCHEMA_SQL` and lightweight migration in `initialize()`.

## Existing Docs
- `README.md`: user-facing setup and API flow.
- `docs/tech-stack.md`: stack rationale.
- `docs/repo-structure.md`: currently empty.
- `docs/roadmap.md`: currently empty.

## Notable Technical Characteristics
- Strong local-first design: local folder import, local DB, local FFmpeg pipeline.
- Queue is resilient across restart via JSON persistence and ASR checkpoint rewind.
- ASR robustness: CUDA preference with CPU fallback and event journaling.
- Phrase-aware mix planning reduces dependency on single-word timestamp precision.
- Metadata-rich lifecycle logging (`base_metadata.json`, queue `last_event`).
- Current implementation intentionally favors deterministic single-worker behavior over distributed throughput.

## Known Gaps / TODO Signals in Codebase
- Worker package is scaffold only; no real async distributed job execution yet.
- Core protocol package is present but not enforced in app wiring.
- API still includes direct one-off ingest path though primary UX is base import queue.
- README still mentions some scaffold-era statements (for example endpoint set descriptions) that are broader than finalized queue-first behavior.
- `start.ps1` appears to contain a typo (`1$ErrorActionPreference`).


# Audio_Typewriter

Audio_Typewriter is a sentence-mixing toolkit: it transcribes speech to word-level timestamps, stores token references, and rebuilds new target sentences by stitching clips.

## Current MVP Scope

- English-first word-level ASR ingestion.
- SQLite-based token index and mix-job tracking.
- Sentence token search and FFmpeg-based audio stitching.
- Default `cuda` ASR selection with automatic fallback to `cpu` when GPU is unavailable.
- Automatic model download to `./models` (faster-whisper cache root).

## What Is Already Scaffolded

- Monorepo layout for API, Web UI, shared core contracts, workers, docs, and infra.
- FastAPI backend with health/ingest/mix/model-download endpoints.
- React + Vite UI skeleton with a basic sentence submit flow.
- Core domain interfaces for ASR, index storage, and mixer adapters.
- Docs for stack rationale, roadmap, and repository boundaries.

## Proposed Tech Stack (Performance + Compatibility)

- Backend: `FastAPI` + `Uvicorn`
- ASR (English first): `faster-whisper` with `cuda -> cpu` fallback
- Audio processing: `FFmpeg` command line
- Database (MVP): built-in `SQLite3` with exact token indexes
- Queue (recommended later): `Dramatiq + Redis`
- Frontend: `React + TypeScript + Vite`

Why this stack:
- Runs on mainstream computers without mandatory GPU.
- Keeps local setup simple while allowing future scale-up.
- Uses mature open-source tooling with strong ecosystem support.

## Repository Layout

```text
Audio_Typewriter/
  apps/
    api/
    web/
  packages/
    core/
    workers/
  docs/
  infra/docker/
  scripts/
```

## Quick Start (Windows PowerShell)

1) Bootstrap environment:

```powershell
Set-Location "E:\Audio_Typewriter"
.\scripts\bootstrap.ps1 -InstallWeb
```

2) Start backend API:

```powershell
Set-Location "E:\Audio_Typewriter"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir apps/api
```

3) Start Web UI:

```powershell
Set-Location "E:\Audio_Typewriter\apps\web"
npm run dev
```

## Model Download (Large by Default)

The project stores ASR models in `./models` by default (`AT_ASR_MODEL_CACHE_DIR`).

Download `large` (`large-v3`) via script:

```powershell
Set-Location "E:\Audio_Typewriter"
.\.venv\Scripts\python.exe .\scripts\download_model.py --model-tier large
```

Or download an explicit model/repo:

```powershell
Set-Location "E:\Audio_Typewriter"
.\.venv\Scripts\python.exe .\scripts\download_model.py --model-name "Systran/faster-whisper-large-v3"
```

API alternative:

```powershell
Set-Location "E:\Audio_Typewriter"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/models/download" -Method POST -ContentType "application/json" -Body '{"model_tier":"large"}'
```

## API Flow

1. `POST /api/v1/models/download`
   - Trigger model download/warmup into `./models`.
2. `POST /api/v1/ingest`
   - Input audio file path, optional `model_tier` or explicit `model_name`.
   - Run ASR, extract word timestamps, and store token rows.
3. `POST /api/v1/mix`
   - Submit a target sentence.
   - Search the index and render a stitched audio file.

## Next Implementation Targets

- Add audio upload endpoints instead of path-only ingest.
- Add richer clip scoring when multiple sources contain the same word.
- Add preview playback and timeline visualization in Web UI.
- Add async worker execution and progress tracking.

# Tech Stack Proposal

## Goals
- Run on mainstream computers (Windows/macOS/Linux, CPU-first fallback, GPU-optional).
- Keep MVP simple while preserving scale-up paths.
- Favor open-source tooling with stable ecosystems.

## Recommended Stack
- Backend API: FastAPI + Uvicorn
- ASR and word timestamps: faster-whisper (CTranslate2 backend)
- Audio processing: FFmpeg CLI
- Storage (MVP): built-in SQLite3 with exact token indexes
- Storage (scale-up): PostgreSQL + object storage (S3/MinIO)
- Queue/background jobs: Dramatiq + Redis (MVP can start with FastAPI background tasks)
- Frontend: React + TypeScript + Vite

## Why This Works Well
- faster-whisper is fast on CPU and can use CUDA when available.
- FFmpeg is robust for normalization, slicing, and concat.
- SQLite keeps local setup friction very low for single-machine usage.
- FastAPI provides typed contracts and easy async extension.

## Performance Strategy
- Default to `cuda` when available, then fall back to `cpu` automatically.
- Use model tiers (`tiny`, `base`, `small`) selected by hardware profile.
- Cache normalized audio and ASR outputs to avoid repeated transcription.
- Store only word offsets and source references in DB.
- Offload heavy ASR/mixing tasks to worker processes.
- Batch clipping and concat through FFmpeg filter graphs for speed.


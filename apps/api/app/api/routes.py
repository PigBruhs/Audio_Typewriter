from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..db import SQLiteDatabase
from ..schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MixRequest,
    MixResponse,
    ModelDownloadRequest,
    ModelDownloadResponse,
)
from ..services.asr_service import ASRService
from ..services.index_service import IndexService
from ..services.mixing_service import MixingService

router = APIRouter()
_database = SQLiteDatabase()
_asr_service = ASRService()
_index_service = IndexService(_database)
_mixing_service = MixingService(_database, _index_service)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/models/download", response_model=ModelDownloadResponse)
def download_model(payload: ModelDownloadRequest) -> ModelDownloadResponse:
    try:
        result = _asr_service.download_model(
            model_tier=payload.model_tier,
            language=payload.language,
            model_name=payload.model_name,
        )
        return ModelDownloadResponse(
            model_name=result.model_name,
            status=result.status,
            device_used=result.device_used,
            compute_type=result.compute_type,
            cache_dir=result.cache_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest", response_model=IngestResponse)
def ingest_audio(payload: IngestRequest) -> IngestResponse:
    try:
        source_record, occurrences, result = _asr_service.ingest(
            source_path=payload.source_path,
            language=payload.language,
            model_tier=payload.model_tier,
            model_name=payload.model_name,
        )
        _index_service.ingest(source_record, occurrences)
        return IngestResponse(
            source_audio_id=result.source_audio_id,
            status=result.status,
            token_count=result.token_count,
            device_used=result.device_used,
            compute_type=result.compute_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/mix", response_model=MixResponse)
def create_mix(payload: MixRequest) -> MixResponse:
    try:
        result = _mixing_service.mix_sentence(payload.sentence, output_path=payload.output_path)
        return MixResponse(
            job_id=result.job_id,
            status=result.status,
            output_path=result.output_path,
            missing_tokens=result.missing_tokens,
            token_count=result.token_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

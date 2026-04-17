from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..db import SQLiteDatabase
from ..schemas import (
    HealthResponse,
    AudioBaseImportResponse,
    AudioBaseListItem,
    AudioBaseStatsResponse,
    IngestRequest,
    IngestResponse,
    MixRequest,
    MixResponse,
    ModelDownloadRequest,
    ModelDownloadResponse,
)
from ..services.asr_service import ASRService
from ..services.audio_base_service import AudioBaseService
from ..services.index_service import IndexService
from ..services.mixing_service import MixingService

router = APIRouter()
_database = SQLiteDatabase()
_asr_service = ASRService()
_audio_base_service = AudioBaseService()
_index_service = IndexService(_database)
_mixing_service = MixingService(_database, _index_service)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/audio-bases/import", response_model=AudioBaseImportResponse)
def import_audio_base(base_name: str = Form(...), files: list[UploadFile] = File(...)) -> AudioBaseImportResponse:
    try:
        base_record, file_records = _audio_base_service.import_audio_files(base_name=base_name, files=files)
        _database.create_audio_base(base_record)
        _database.replace_audio_base_files(base_record.base_name, file_records)

        total_tokens = 0
        for file_record in file_records:
            source_record, occurrences, result = _asr_service.ingest(
                source_path=file_record.file_path,
                source_audio_id=file_record.source_audio_id,
                base_name=base_record.base_name,
                language="en",
                model_tier="large",
            )
            _index_service.ingest(source_record, occurrences)
            total_tokens += result.token_count

        stats = _audio_base_service.summarize_records(base_record.base_name, file_records)
        return AudioBaseImportResponse(
            base_name=stats.base_name,
            audio_count=stats.audio_count,
            total_duration_sec=stats.total_duration_sec,
            total_file_size_bytes=stats.total_file_size_bytes,
            ingested_source_count=len(file_records),
            token_count=total_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/audio-bases", response_model=list[AudioBaseListItem])
def list_audio_bases() -> list[AudioBaseListItem]:
    items: list[AudioBaseListItem] = []
    for base in _database.list_audio_bases():
        stats = _database.get_audio_base_stats(base.base_name)
        if not stats:
            continue
        items.append(
            AudioBaseListItem(
                base_name=stats.base_name,
                audio_count=stats.audio_count,
                total_duration_sec=stats.total_duration_sec,
                total_file_size_bytes=stats.total_file_size_bytes,
            )
        )
    return items


@router.get("/audio-bases/{base_name}/stats", response_model=AudioBaseStatsResponse)
def get_audio_base_stats(base_name: str) -> AudioBaseStatsResponse:
    stats = _database.get_audio_base_stats(base_name)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Audio base '{base_name}' was not found")
    return AudioBaseStatsResponse(
        base_name=stats.base_name,
        audio_count=stats.audio_count,
        total_duration_sec=stats.total_duration_sec,
        total_file_size_bytes=stats.total_file_size_bytes,
    )


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
            source_audio_id=payload.source_audio_id,
            base_name=payload.base_name,
            language=payload.language,
            model_tier=payload.model_tier,
            model_name=payload.model_name,
        )
        _index_service.ingest(source_record, occurrences)
        return IngestResponse(
            source_audio_id=result.source_audio_id,
            base_name=result.base_name,
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
        result = _mixing_service.mix_sentence(
            payload.sentence,
            base_name=payload.base_name,
            output_path=payload.output_path,
            mix_mode=payload.mix_mode,
        )
        return MixResponse(
            job_id=result.job_id,
            base_name=result.base_name,
            status=result.status,
            output_path=result.output_path,
            missing_tokens=result.missing_tokens,
            token_count=result.token_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

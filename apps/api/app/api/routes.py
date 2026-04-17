from __future__ import annotations

import json
import os
import threading
import time
from queue import Empty, Queue
from typing import Callable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency at runtime
    tqdm = None

from ..db import SQLiteDatabase
from ..models import MixPlanItem
from ..schemas import (
    HealthResponse,
    AudioBaseImportResponse,
    AudioBaseListItem,
    AudioBaseStatsResponse,
    IngestRequest,
    IngestResponse,
    MixRequest,
    MixResponse,
    StitchRequest,
    ModelDownloadRequest,
    ModelDownloadResponse,
)
from ..services.asr_service import ASRService
from ..services.audio_base_service import AudioBaseService
from ..services.index_service import IndexService
from ..services.mixing_service import MixingService
from ..services.task_queue_service import TaskQueueService

router = APIRouter()
_database = SQLiteDatabase()
_asr_service = ASRService()
_audio_base_service = AudioBaseService()
_index_service = IndexService(_database)
_mixing_service = MixingService(_database, _index_service)
_task_queue_service = TaskQueueService(_database, _asr_service, _index_service)


def _iter_with_tqdm(file_records, base_name: str):
    if tqdm is None:
        return enumerate(file_records, start=1)
    return enumerate(
        tqdm(
            file_records,
            total=len(file_records),
            desc=f"PREP[{base_name}]",
            unit="file",
            leave=False,
        ),
        start=1,
    )


def _run_audio_base_import(
    *,
    base_name: str,
    files: list[UploadFile],
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> AudioBaseImportResponse:
    normalized_base_name = _audio_base_service.validate_base_name(base_name)
    overwritten = _audio_base_service.base_exists(normalized_base_name)
    cleared_audio_files = 0
    cleared_index_sources = 0
    discarded_info = {"discarded_tasks": 0, "had_running_task": 0}
    if overwritten:
        discarded_info = _task_queue_service.discard_unfinished_for_base(normalized_base_name)
        cleanup_info = _database.clear_audio_base_for_overwrite(normalized_base_name)
        cleared_index_sources = int(cleanup_info["cleared_index_sources"])
        cleared_audio_files = _audio_base_service.clear_base_storage(normalized_base_name)
        if progress_callback:
            progress_callback(
                {
                    "type": "overwrite",
                    "base_name": normalized_base_name,
                    "cleared_audio_files": cleared_audio_files,
                    "cleared_index_sources": cleared_index_sources,
                }
            )

    base_record, file_records = _audio_base_service.import_audio_files_with_progress(
        base_name=normalized_base_name,
        files=files,
        progress_callback=progress_callback,
    )
    _database.create_audio_base(base_record)
    _database.replace_audio_base_files(base_record.base_name, file_records)

    total_files = len(file_records)
    for _index, _file_record in _iter_with_tqdm(file_records, base_record.base_name):
        pass

    queued_task = _task_queue_service.enqueue_import_task(
        base_name=base_record.base_name,
        total_files=total_files,
        model_tier="large",
        overwritten=overwritten,
        cleared_audio_files=cleared_audio_files,
        cleared_index_sources=cleared_index_sources,
    )

    stats = _audio_base_service.summarize_records(base_record.base_name, file_records)
    response = AudioBaseImportResponse(
        base_name=stats.base_name,
        overwritten=overwritten,
        cleared_audio_files=cleared_audio_files,
        cleared_index_sources=cleared_index_sources,
        audio_count=stats.audio_count,
        total_duration_sec=stats.total_duration_sec,
        total_file_size_bytes=stats.total_file_size_bytes,
        ingested_source_count=len(file_records),
        token_count=0,
        task_id=str(queued_task["task_id"]),
        task_status=str(queued_task["status"]),
        discarded_task_count=int(discarded_info["discarded_tasks"]),
    )
    if progress_callback:
        progress_callback(
            {
                "type": "status",
                "message": "VAD and file import completed. ASR task is queued; pause/resume in Tasks tab.",
            }
        )
        progress_callback(
            {"type": "start", "base_name": base_record.base_name, "total": total_files}
        )
        progress_callback({"type": "complete", "result": response.model_dump()})
    return response


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    resolved_device, compute_type = _asr_service.resolve_runtime()
    return HealthResponse(
        status="ok",
        asr_preferred_device=_asr_service.settings.asr_device,
        asr_resolved_device=resolved_device,
        asr_compute_type=compute_type,
        asr_last_device_used=_asr_service.last_device_used,
        asr_last_compute_type=_asr_service.last_compute_type,
    )


@router.post("/audio-bases/import", response_model=AudioBaseImportResponse)
def import_audio_base(base_name: str = Form(...), files: list[UploadFile] = File(...)) -> AudioBaseImportResponse:
    try:
        return _run_audio_base_import(base_name=base_name, files=files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/audio-bases/import/stream")
def import_audio_base_stream(base_name: str = Form(...), files: list[UploadFile] = File(...)) -> StreamingResponse:
    def _iter_events():
        event_queue: Queue[dict[str, object]] = Queue()

        def _emit(payload: dict[str, object]) -> None:
            event_queue.put(payload)

        def _worker() -> None:
            try:
                _run_audio_base_import(
                    base_name=base_name,
                    files=files,
                    progress_callback=_emit,
                )
            except ValueError as exc:
                _emit({"type": "error", "detail": str(exc)})
            except RuntimeError as exc:
                _emit({"type": "error", "detail": str(exc)})

        worker = threading.Thread(target=_worker, daemon=True, name="import-stream-worker")
        worker.start()

        while True:
            try:
                payload = event_queue.get(timeout=0.2)
                yield json.dumps(payload, ensure_ascii=False) + "\n"
                if payload.get("type") in {"complete", "error"} and not worker.is_alive():
                    break
            except Empty:
                if not worker.is_alive() and event_queue.empty():
                    break

    return StreamingResponse(_iter_events(), media_type="application/x-ndjson")


@router.get("/tasks")
def list_tasks() -> list[dict[str, object]]:
    return _task_queue_service.list_tasks()


@router.post("/tasks/{task_id}/pause")
def pause_task(task_id: str) -> dict[str, object]:
    try:
        return _task_queue_service.pause_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/resume")
def resume_task(task_id: str) -> dict[str, object]:
    try:
        return _task_queue_service.resume_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/system/exit")
def system_exit() -> dict[str, str]:
    _task_queue_service.flush()

    def _shutdown() -> None:
        time.sleep(0.2)
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()
    return {"status": "shutting_down"}


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


@router.post("/mix/stitch", response_model=MixResponse)
def stitch_mix(payload: StitchRequest) -> MixResponse:
    try:
        segments = [
            MixPlanItem(
                token=(segment.label or segment.source_audio_id),
                source_audio_id=segment.source_audio_id,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
            )
            for segment in payload.segments
        ]
        result = _mixing_service.stitch_segments(
            base_name=payload.base_name,
            segments=segments,
            output_path=payload.output_path,
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



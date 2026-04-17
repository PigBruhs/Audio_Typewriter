from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import Settings, settings
from ..db import SQLiteDatabase
from ..models import AudioBaseRecord
from .asr_service import ASRService
from .audio_base_service import AudioBaseService
from .index_service import IndexService

_TASK_RUNNING = "running"
_TASK_QUEUED = "queued"
_TASK_PAUSED = "paused"
_TASK_COMPLETED = "completed"
_TASK_FAILED = "failed"
_TASK_DISCARDED = "discarded"

_STAGE_VAD = "vad"
_STAGE_ASR = "asr"


@dataclass(slots=True)
class QueueTask:
    task_id: str
    base_name: str
    status: str
    total_files: int
    processed_files: int
    next_sequence_number: int
    token_count: int
    stage: str
    ready_for_asr: bool
    vad_total_audio_sec: float
    vad_processed_audio_sec: float
    vad_source_dir: str | None
    vad_total_sources: int
    vad_next_source_index: int
    vad_next_segment_index: int
    vad_next_sequence_number: int
    vad_created_at: str | None
    asr_last_completed_sequence: int
    model_tier: str
    created_at: str
    updated_at: str
    last_error: str | None = None
    overwritten: bool = False
    cleared_audio_files: int = 0
    cleared_index_sources: int = 0
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "base_name": self.base_name,
            "status": self.status,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "next_sequence_number": self.next_sequence_number,
            "token_count": self.token_count,
            "stage": self.stage,
            "ready_for_asr": self.ready_for_asr,
            "vad_total_audio_sec": self.vad_total_audio_sec,
            "vad_processed_audio_sec": self.vad_processed_audio_sec,
            "vad_source_dir": self.vad_source_dir,
            "vad_total_sources": self.vad_total_sources,
            "vad_next_source_index": self.vad_next_source_index,
            "vad_next_segment_index": self.vad_next_segment_index,
            "vad_next_sequence_number": self.vad_next_sequence_number,
            "vad_created_at": self.vad_created_at,
            "asr_last_completed_sequence": self.asr_last_completed_sequence,
            "model_tier": self.model_tier,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
            "overwritten": self.overwritten,
            "cleared_audio_files": self.cleared_audio_files,
            "cleared_index_sources": self.cleared_index_sources,
        }

    @staticmethod
    def from_dict(payload: dict[str, object]) -> "QueueTask":
        return QueueTask(
            task_id=str(payload["task_id"]),
            base_name=str(payload["base_name"]),
            status=str(payload["status"]),
            total_files=int(payload["total_files"]),
            processed_files=int(payload.get("processed_files", 0)),
            next_sequence_number=int(payload.get("next_sequence_number", 1)),
            token_count=int(payload.get("token_count", 0)),
            stage=str(payload.get("stage", _STAGE_ASR)),
            ready_for_asr=bool(payload.get("ready_for_asr", True)),
            vad_total_audio_sec=float(payload.get("vad_total_audio_sec", 0.0)),
            vad_processed_audio_sec=float(payload.get("vad_processed_audio_sec", 0.0)),
            vad_source_dir=(str(payload["vad_source_dir"]) if payload.get("vad_source_dir") else None),
            vad_total_sources=int(payload.get("vad_total_sources", 0)),
            vad_next_source_index=int(payload.get("vad_next_source_index", 1)),
            vad_next_segment_index=int(payload.get("vad_next_segment_index", 0)),
            vad_next_sequence_number=int(payload.get("vad_next_sequence_number", 1)),
            vad_created_at=(str(payload["vad_created_at"]) if payload.get("vad_created_at") else None),
            asr_last_completed_sequence=int(payload.get("asr_last_completed_sequence", 0)),
            model_tier=str(payload.get("model_tier", "large")),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            last_error=(str(payload["last_error"]) if payload.get("last_error") is not None else None),
            overwritten=bool(payload.get("overwritten", False)),
            cleared_audio_files=int(payload.get("cleared_audio_files", 0)),
            cleared_index_sources=int(payload.get("cleared_index_sources", 0)),
            cancel_requested=False,
        )


class TaskQueueService:
    def __init__(
        self,
        database: SQLiteDatabase,
        asr_service: ASRService,
        index_service: IndexService,
        audio_base_service: AudioBaseService,
        runtime_settings: Settings | None = None,
    ) -> None:
        self.settings = runtime_settings or settings
        self.database = database
        self.asr_service = asr_service
        self.index_service = index_service
        self.audio_base_service = audio_base_service
        self.queue_path = self.settings.data_dir / "asr_task_queue.json"
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._tasks: list[QueueTask] = []
        self._shutdown_requested = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="asr-task-worker")
        self._load_tasks()
        self._worker.start()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _save_tasks(self) -> None:
        self.settings.ensure_directories()
        payload = [task.to_dict() for task in self._tasks]
        self.queue_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_tasks(self) -> None:
        self.settings.ensure_directories()
        if not self.queue_path.exists():
            self._tasks = []
            return
        try:
            payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except Exception:
            self._tasks = []
            return

        loaded: list[QueueTask] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            task = QueueTask.from_dict(row)
            if task.status == _TASK_RUNNING:
                task.status = _TASK_PAUSED
            self._rewind_asr_checkpoint_for_resume(task)
            loaded.append(task)
        self._tasks = loaded
        self._save_tasks()

    def _rewind_asr_checkpoint_for_resume(self, task: QueueTask) -> None:
        if task.stage != _STAGE_ASR:
            return
        if task.asr_last_completed_sequence > 0:
            task.next_sequence_number = max(1, min(task.next_sequence_number, task.asr_last_completed_sequence))

    def list_tasks(self) -> list[dict[str, object]]:
        with self._lock:
            return [task.to_dict() for task in self._tasks]

    def discard_unfinished_for_base(self, base_name: str) -> dict[str, int]:
        with self._condition:
            discarded = 0
            waiting_running = 0
            for task in self._tasks:
                if task.base_name != base_name:
                    continue
                if task.status in {_TASK_COMPLETED, _TASK_DISCARDED}:
                    continue
                if task.status == _TASK_RUNNING:
                    task.cancel_requested = True
                    waiting_running += 1
                    continue
                task.status = _TASK_DISCARDED
                task.updated_at = self._now_iso()
                discarded += 1
            self._save_tasks()
            self._condition.notify_all()

        # Wait for running tasks of the same base to stop at file boundary.
        if waiting_running:
            deadline = time.time() + 120
            while time.time() < deadline:
                with self._lock:
                    still_running = any(
                        task.base_name == base_name and task.status == _TASK_RUNNING for task in self._tasks
                    )
                if not still_running:
                    break
                time.sleep(0.2)

            with self._lock:
                for task in self._tasks:
                    if task.base_name == base_name and task.status == _TASK_RUNNING and task.cancel_requested:
                        task.status = _TASK_DISCARDED
                        task.updated_at = self._now_iso()
                        discarded += 1
                self._save_tasks()

        return {"discarded_tasks": discarded, "had_running_task": waiting_running}

    def enqueue_import_task(
        self,
        *,
        base_name: str,
        total_files: int,
        vad_source_dir: str,
        vad_total_sources: int,
        vad_total_audio_sec: float,
        task_id: str | None = None,
        model_tier: str = "large",
        overwritten: bool = False,
        cleared_audio_files: int = 0,
        cleared_index_sources: int = 0,
    ) -> dict[str, object]:
        now = self._now_iso()
        task = QueueTask(
            task_id=str(task_id or uuid.uuid4()),
            base_name=base_name,
            status=_TASK_RUNNING,
            total_files=total_files,
            processed_files=0,
            next_sequence_number=1,
            token_count=0,
            stage=_STAGE_VAD,
            ready_for_asr=True,
            vad_total_audio_sec=vad_total_audio_sec,
            vad_processed_audio_sec=0.0,
            vad_source_dir=vad_source_dir,
            vad_total_sources=vad_total_sources,
            vad_next_source_index=1,
            vad_next_segment_index=0,
            vad_next_sequence_number=1,
            vad_created_at=now,
            asr_last_completed_sequence=0,
            model_tier=model_tier,
            created_at=now,
            updated_at=now,
            overwritten=overwritten,
            cleared_audio_files=cleared_audio_files,
            cleared_index_sources=cleared_index_sources,
        )
        with self._condition:
            self._tasks.append(task)
            self._save_tasks()
            self._condition.notify_all()
        return task.to_dict()

    def pause_task(self, task_id: str) -> dict[str, object]:
        with self._condition:
            for task in self._tasks:
                if task.task_id != task_id:
                    continue
                if task.status == _TASK_COMPLETED:
                    return task.to_dict()
                if task.status == _TASK_QUEUED:
                    task.status = _TASK_PAUSED
                elif task.status == _TASK_RUNNING:
                    task.status = _TASK_PAUSED
                task.updated_at = self._now_iso()
                self._save_tasks()
                self._condition.notify_all()
                return task.to_dict()
        raise ValueError(f"Task not found: {task_id}")

    def resume_task(self, task_id: str) -> dict[str, object]:
        with self._condition:
            for task in self._tasks:
                if task.task_id != task_id:
                    continue
                if task.status in {_TASK_COMPLETED, _TASK_DISCARDED}:
                    return task.to_dict()
                if task.stage == _STAGE_VAD:
                    task.status = _TASK_RUNNING
                else:
                    task.status = _TASK_QUEUED
                task.last_error = None
                task.updated_at = self._now_iso()
                self._save_tasks()
                self._condition.notify_all()
                return task.to_dict()
        raise ValueError(f"Task not found: {task_id}")

    def checkpoint_vad(self, task_id: str) -> None:
        with self._condition:
            while True:
                task = next((item for item in self._tasks if item.task_id == task_id), None)
                if task is None:
                    raise RuntimeError(f"Task not found: {task_id}")
                if task.cancel_requested or task.status == _TASK_DISCARDED:
                    task.status = _TASK_DISCARDED
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                    raise RuntimeError(f"Task discarded: {task_id}")
                if task.status == _TASK_PAUSED:
                    self._condition.wait(timeout=0.5)
                    continue
                return

    def mark_task_failed(self, task_id: str, detail: str) -> None:
        with self._condition:
            for task in self._tasks:
                if task.task_id != task_id:
                    continue
                task.status = _TASK_FAILED
                task.last_error = detail
                task.updated_at = self._now_iso()
                self._save_tasks()
                self._condition.notify_all()
                return

    def activate_asr_stage(self, task_id: str, total_files: int) -> dict[str, object]:
        with self._condition:
            for task in self._tasks:
                if task.task_id != task_id:
                    continue
                task.stage = _STAGE_ASR
                task.ready_for_asr = True
                task.total_files = total_files
                task.processed_files = 0
                task.next_sequence_number = 1
                task.token_count = 0
                task.status = _TASK_QUEUED
                task.asr_last_completed_sequence = 0
                if task.vad_total_audio_sec > 0:
                    task.vad_processed_audio_sec = task.vad_total_audio_sec
                task.vad_source_dir = None
                task.updated_at = self._now_iso()
                self._save_tasks()
                self._condition.notify_all()
                return task.to_dict()
        raise ValueError(f"Task not found: {task_id}")

    def update_vad_progress(self, task_id: str, *, processed_audio_sec: float, total_audio_sec: float) -> dict[str, object]:
        with self._condition:
            for task in self._tasks:
                if task.task_id != task_id:
                    continue
                task.vad_total_audio_sec = max(float(total_audio_sec), 0.0)
                task.vad_processed_audio_sec = max(float(processed_audio_sec), 0.0)
                task.updated_at = self._now_iso()
                self._save_tasks()
                return task.to_dict()
        raise ValueError(f"Task not found: {task_id}")

    def flush(self) -> None:
        with self._lock:
            self._save_tasks()

    def delete_task(self, task_id: str, wait_timeout_sec: float = 5.0) -> dict[str, int | str]:
        with self._condition:
            target = next((task for task in self._tasks if task.task_id == task_id), None)
            if target is None:
                raise ValueError(f"Task not found: {task_id}")

            target_base = target.base_name
            for task in self._tasks:
                if task.base_name == target_base and task.status == _TASK_RUNNING:
                    task.cancel_requested = True

            self._save_tasks()
            self._condition.notify_all()

        deadline = time.time() + wait_timeout_sec
        while time.time() < deadline:
            with self._lock:
                still_running = any(task.base_name == target_base and task.status == _TASK_RUNNING for task in self._tasks)
            if not still_running:
                break
            time.sleep(0.2)

        with self._condition:
            removed_task_ids = [task.task_id for task in self._tasks if task.base_name == target_base]
            self._tasks = [task for task in self._tasks if task.base_name != target_base]
            self._save_tasks()
            self._condition.notify_all()

        cleanup_info = self.database.clear_audio_base_for_overwrite(target_base)
        removed_base_rows = self.database.delete_audio_base(target_base)
        removed_audio_files = self.audio_base_service.clear_base_storage(target_base)
        for removed_task_id in removed_task_ids:
            self.audio_base_service.clear_vad_job_storage(removed_task_id)

        return {
            "base_name": target_base,
            "removed_tasks": len(removed_task_ids),
            "removed_audio_files": removed_audio_files,
            "removed_index_sources": int(cleanup_info.get("cleared_index_sources", 0)),
            "removed_base_rows": removed_base_rows,
        }

    def prepare_for_shutdown(self, wait_timeout_sec: float = 5.0) -> dict[str, int]:
        with self._condition:
            self._shutdown_requested = True
            paused_running = 0
            for task in self._tasks:
                if task.status == _TASK_RUNNING:
                    task.status = _TASK_PAUSED
                    if task.stage == _STAGE_ASR:
                        self._rewind_asr_checkpoint_for_resume(task)
                    task.updated_at = self._now_iso()
                    paused_running += 1
            self._save_tasks()
            self._condition.notify_all()

        deadline = time.time() + wait_timeout_sec
        while time.time() < deadline:
            with self._lock:
                active_running = sum(1 for task in self._tasks if task.status == _TASK_RUNNING)
            if active_running == 0:
                break
            time.sleep(0.2)

        with self._lock:
            remaining_running = sum(1 for task in self._tasks if task.status == _TASK_RUNNING)
            self._save_tasks()

        return {
            "paused_running": paused_running,
            "remaining_running": remaining_running,
        }

    def _next_runnable_task(self) -> QueueTask | None:
        for task in self._tasks:
            if task.status in {_TASK_QUEUED, _TASK_RUNNING} and task.stage == _STAGE_VAD and task.ready_for_asr:
                return task
            if task.status == _TASK_QUEUED and task.stage == _STAGE_ASR and task.ready_for_asr:
                return task
        return None

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                if self._shutdown_requested:
                    self._condition.wait(timeout=0.5)
                    continue
                task = self._next_runnable_task()
                if task is None:
                    self._condition.wait(timeout=1.0)
                    continue
                if task.status != _TASK_RUNNING:
                    task.status = _TASK_RUNNING
                task.updated_at = self._now_iso()
                self._save_tasks()

            try:
                if task.stage == _STAGE_VAD:
                    self._run_vad_task(task)
                else:
                    self._run_task(task)
            except Exception as exc:  # pragma: no cover - safety net
                with self._condition:
                    task.status = _TASK_FAILED
                    task.last_error = str(exc)
                    task.updated_at = self._now_iso()
                    self._save_tasks()

    def _run_vad_task(self, task: QueueTask) -> None:
        if not task.vad_source_dir:
            raise RuntimeError("VAD source directory is missing.")

        source_dir = Path(task.vad_source_dir)
        if not source_dir.exists():
            raise RuntimeError("VAD source directory does not exist on disk.")

        manifest = self.audio_base_service.load_vad_manifest(task.task_id)
        if not manifest:
            raise RuntimeError("VAD manifest is missing or empty.")

        for item in manifest:
            index = int(item.get("index", 0))
            if index < task.vad_next_source_index:
                continue

            with self._condition:
                if task.cancel_requested:
                    task.status = _TASK_DISCARDED
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                    return
                if task.status == _TASK_PAUSED:
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                    return

            file_name = str(item.get("file_name", ""))
            source_path = source_dir / file_name
            if not source_path.exists():
                raise RuntimeError(f"VAD source file is missing: {source_path}")

            segments = self.audio_base_service.detect_source_speech_segments(source_path)
            if not segments:
                with self._condition:
                    task.vad_next_source_index = index + 1
                    task.vad_next_segment_index = 0
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                continue

            start_segment_index = task.vad_next_segment_index if index == task.vad_next_source_index else 0
            for segment_index in range(start_segment_index, len(segments)):
                self.checkpoint_vad(task.task_id)
                start_sec, end_sec = segments[segment_index]
                record = self.audio_base_service.export_segment_record(
                    source_path=source_path,
                    base_name=task.base_name,
                    sequence_number=task.vad_next_sequence_number,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    created_at=task.vad_created_at or task.created_at,
                )

                with self._condition:
                    task.vad_processed_audio_sec = min(
                        task.vad_total_audio_sec,
                        round(task.vad_processed_audio_sec + record.duration_sec, 3),
                    )
                    task.vad_next_sequence_number += 1
                    task.vad_next_segment_index = segment_index + 1
                    task.updated_at = self._now_iso()
                    self._save_tasks()

            with self._condition:
                task.vad_next_source_index = index + 1
                task.vad_next_segment_index = 0
                task.updated_at = self._now_iso()
                self._save_tasks()

        records = self.audio_base_service.collect_base_records(task.base_name, task.vad_created_at or task.created_at)
        base_record = AudioBaseRecord(
            base_name=task.base_name,
            base_path=str(self.audio_base_service.base_path(task.base_name)),
            created_at=task.vad_created_at or task.created_at,
            updated_at=self._now_iso(),
        )
        self.database.create_audio_base(base_record)
        self.database.replace_audio_base_files(task.base_name, records)
        self.activate_asr_stage(task.task_id, total_files=len(records))

    def _run_task(self, task: QueueTask) -> None:
        file_records = self.database.list_audio_base_files(task.base_name, task.next_sequence_number)
        for record in file_records:
            with self._condition:
                if task.cancel_requested:
                    task.status = _TASK_DISCARDED
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                    return
                if task.status == _TASK_PAUSED:
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                    return

            source_record, occurrences, result = self.asr_service.ingest(
                source_path=record.file_path,
                source_audio_id=record.source_audio_id,
                base_name=record.base_name,
                language="en",
                model_tier=task.model_tier,
            )
            self.index_service.ingest(source_record, occurrences)

            with self._condition:
                task.processed_files += 1
                task.token_count += int(result.token_count)
                task.asr_last_completed_sequence = int(record.sequence_number)
                task.next_sequence_number = record.sequence_number + 1
                task.updated_at = self._now_iso()
                self._save_tasks()

        with self._condition:
            if task.status == _TASK_RUNNING:
                task.status = _TASK_COMPLETED
                task.updated_at = self._now_iso()
                self._save_tasks()


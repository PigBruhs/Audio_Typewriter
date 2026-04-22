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
from .asr_service import ASRService, ASRTranscriptionError
from .audio_base_service import AudioBaseService
from .index_service import IndexService

_TASK_RUNNING = "running"
_TASK_QUEUED = "queued"
_TASK_PAUSED = "paused"
_TASK_COMPLETED = "completed"
_TASK_FAILED = "failed"
_TASK_DISCARDED = "discarded"

_STAGE_PREPROCESS = "preprocess"
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
    vad_source_paths: list[str]
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
    last_event: str | None = None
    vad_elapsed_sec: float = 0.0
    asr_elapsed_sec: float = 0.0
    vad_running_since: str | None = None
    asr_running_since: str | None = None
    asr_total_audio_sec: float = 0.0
    asr_processed_audio_sec: float = 0.0
    asr_language: str = "en"

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
            "vad_source_paths": list(self.vad_source_paths),
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
            "last_event": self.last_event,
            "vad_elapsed_sec": self.vad_elapsed_sec,
            "asr_elapsed_sec": self.asr_elapsed_sec,
            "vad_running_since": self.vad_running_since,
            "asr_running_since": self.asr_running_since,
            "asr_total_audio_sec": self.asr_total_audio_sec,
            "asr_processed_audio_sec": self.asr_processed_audio_sec,
            "asr_language": self.asr_language,
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
            vad_source_paths=[str(item) for item in (payload.get("vad_source_paths") or []) if str(item)],
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
            last_event=(str(payload["last_event"]) if payload.get("last_event") is not None else None),
            vad_elapsed_sec=float(payload.get("vad_elapsed_sec", 0.0)),
            asr_elapsed_sec=float(payload.get("asr_elapsed_sec", 0.0)),
            vad_running_since=(str(payload["vad_running_since"]) if payload.get("vad_running_since") else None),
            asr_running_since=(str(payload["asr_running_since"]) if payload.get("asr_running_since") else None),
            asr_total_audio_sec=float(payload.get("asr_total_audio_sec", 0.0)),
            asr_processed_audio_sec=float(payload.get("asr_processed_audio_sec", 0.0)),
            asr_language=str(payload.get("asr_language", settings.asr_default_language) or settings.asr_default_language),
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
                task.vad_running_since = None
                task.asr_running_since = None
            self._rewind_asr_checkpoint_for_resume(task)
            loaded.append(task)
        self._tasks = loaded
        self._save_tasks()

    def _parse_iso_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _start_stage_timer(self, task: QueueTask, stage: str, now_iso: str | None = None) -> None:
        stamp = now_iso or self._now_iso()
        if stage == _STAGE_VAD:
            if not task.vad_running_since:
                task.vad_running_since = stamp
            task.asr_running_since = None
            return
        if stage == _STAGE_ASR:
            if not task.asr_running_since:
                task.asr_running_since = stamp
            task.vad_running_since = None

    def _accumulate_stage_timer(self, task: QueueTask, stage: str, now_iso: str | None = None) -> float:
        stamp = now_iso or self._now_iso()
        now_dt = self._parse_iso_datetime(stamp)
        if now_dt is None:
            return 0.0

        if stage == _STAGE_VAD:
            since_dt = self._parse_iso_datetime(task.vad_running_since)
            task.vad_running_since = None
            if since_dt is None:
                return task.vad_elapsed_sec
            task.vad_elapsed_sec = round(max(0.0, task.vad_elapsed_sec + max(0.0, (now_dt - since_dt).total_seconds())), 3)
            return task.vad_elapsed_sec

        since_dt = self._parse_iso_datetime(task.asr_running_since)
        task.asr_running_since = None
        if since_dt is None:
            return task.asr_elapsed_sec
        task.asr_elapsed_sec = round(max(0.0, task.asr_elapsed_sec + max(0.0, (now_dt - since_dt).total_seconds())), 3)
        return task.asr_elapsed_sec

    def _rewind_asr_checkpoint_for_resume(self, task: QueueTask) -> None:
        if task.stage != _STAGE_ASR:
            return
        if task.asr_last_completed_sequence > 0:
            task.next_sequence_number = max(1, min(task.next_sequence_number, task.asr_last_completed_sequence))
            # Keep progress counters consistent with rewind-to-last-completed strategy.
            task.processed_files = min(task.processed_files, max(0, task.next_sequence_number - 1))

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
        vad_source_paths: list[str] | None = None,
        vad_total_sources: int,
        vad_total_audio_sec: float,
        task_id: str | None = None,
        model_tier: str = "large",
        asr_language: str | None = None,
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
            vad_source_paths=list(vad_source_paths or []),
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
            vad_elapsed_sec=0.0,
            asr_elapsed_sec=0.0,
            vad_running_since=now,
            asr_running_since=None,
            asr_total_audio_sec=0.0,
            asr_processed_audio_sec=0.0,
            asr_language=(str(asr_language or self.settings.asr_default_language).strip().lower() or self.settings.asr_default_language),
        )
        with self._condition:
            self._tasks.append(task)
            self._save_tasks()
            self._condition.notify_all()
        return task.to_dict()

    def create_preprocess_task(
        self,
        *,
        base_name: str,
        task_id: str,
        model_tier: str = "large",
        asr_language: str | None = None,
        overwritten: bool = False,
        cleared_audio_files: int = 0,
        cleared_index_sources: int = 0,
    ) -> dict[str, object]:
        now = self._now_iso()
        task = QueueTask(
            task_id=task_id,
            base_name=base_name,
            status=_TASK_RUNNING,
            total_files=0,
            processed_files=0,
            next_sequence_number=1,
            token_count=0,
            stage=_STAGE_PREPROCESS,
            ready_for_asr=False,
            vad_total_audio_sec=0.0,
            vad_processed_audio_sec=0.0,
            vad_source_dir=None,
            vad_source_paths=[],
            vad_total_sources=0,
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
            last_event="Preprocess started.",
            vad_elapsed_sec=0.0,
            asr_elapsed_sec=0.0,
            vad_running_since=None,
            asr_running_since=None,
            asr_total_audio_sec=0.0,
            asr_processed_audio_sec=0.0,
            asr_language=(str(asr_language or self.settings.asr_default_language).strip().lower() or self.settings.asr_default_language),
        )
        with self._condition:
            self._tasks.append(task)
            self._save_tasks()
            self._condition.notify_all()
        return task.to_dict()

    def update_preprocess_progress(
        self,
        task_id: str,
        *,
        migrated_files: int,
        total_files: int,
        message: str,
    ) -> dict[str, object]:
        with self._condition:
            for task in self._tasks:
                if task.task_id != task_id:
                    continue
                if task.stage != _STAGE_PREPROCESS:
                    return task.to_dict()
                task.processed_files = max(0, int(migrated_files))
                task.total_files = max(task.processed_files, int(total_files))
                task.vad_total_sources = task.total_files
                task.last_event = message
                task.updated_at = self._now_iso()
                self._save_tasks()
                return task.to_dict()
        raise ValueError(f"Task not found: {task_id}")

    def activate_vad_stage(
        self,
        task_id: str,
        *,
        vad_source_paths: list[str],
        vad_total_sources: int,
        vad_total_audio_sec: float,
    ) -> dict[str, object]:
        with self._condition:
            for task in self._tasks:
                if task.task_id != task_id:
                    continue
                task.stage = _STAGE_VAD
                task.status = _TASK_QUEUED
                task.ready_for_asr = True
                task.vad_source_dir = None
                task.vad_source_paths = [str(path) for path in vad_source_paths]
                task.vad_total_sources = max(0, int(vad_total_sources))
                task.total_files = task.vad_total_sources
                task.processed_files = 0
                task.vad_total_audio_sec = max(0.0, float(vad_total_audio_sec))
                task.vad_processed_audio_sec = 0.0
                task.vad_next_source_index = 1
                task.vad_next_segment_index = 0
                task.vad_next_sequence_number = 1
                task.last_error = None
                task.last_event = "Preprocess completed. VAD queued."
                task.updated_at = self._now_iso()
                self._save_tasks()
                self._condition.notify_all()
                return task.to_dict()
        raise ValueError(f"Task not found: {task_id}")

    def fail_preprocess_task(self, task_id: str, detail: str) -> dict[str, object]:
        with self._condition:
            for task in self._tasks:
                if task.task_id != task_id:
                    continue
                task.status = _TASK_FAILED
                task.last_error = detail
                task.last_event = f"Preprocess failed: {detail}"
                task.updated_at = self._now_iso()
                self._save_tasks()
                self._condition.notify_all()
                return task.to_dict()
        raise ValueError(f"Task not found: {task_id}")

    def enqueue_reasr_task(
        self,
        *,
        base_name: str,
        total_files: int,
        model_tier: str = "large",
        asr_language: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, object]:
        now = self._now_iso()
        task = QueueTask(
            task_id=str(task_id or uuid.uuid4()),
            base_name=base_name,
            status=_TASK_QUEUED,
            total_files=max(0, int(total_files)),
            processed_files=0,
            next_sequence_number=1,
            token_count=0,
            stage=_STAGE_ASR,
            ready_for_asr=True,
            vad_total_audio_sec=0.0,
            vad_processed_audio_sec=0.0,
            vad_source_dir=None,
            vad_source_paths=[],
            vad_total_sources=0,
            vad_next_source_index=1,
            vad_next_segment_index=0,
            vad_next_sequence_number=1,
            vad_created_at=None,
            asr_last_completed_sequence=0,
            model_tier=model_tier,
            created_at=now,
            updated_at=now,
            last_event=f"reASR queued for base '{base_name}' ({total_files} files).",
            vad_elapsed_sec=0.0,
            asr_elapsed_sec=0.0,
            vad_running_since=None,
            asr_running_since=None,
            asr_total_audio_sec=0.0,
            asr_processed_audio_sec=0.0,
            asr_language=(str(asr_language or self.settings.asr_default_language).strip().lower() or self.settings.asr_default_language),
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
                    self._accumulate_stage_timer(task, task.stage)
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
                if task.status == _TASK_RUNNING:
                    return task.to_dict()
                if task.stage in {_STAGE_PREPROCESS, _STAGE_VAD}:
                    task.status = _TASK_RUNNING
                    if task.stage == _STAGE_VAD:
                        self._start_stage_timer(task, _STAGE_VAD)
                else:
                    self._rewind_asr_checkpoint_for_resume(task)
                    self.database.purge_asr_index_from_sequence(task.base_name, task.next_sequence_number)
                    task.status = _TASK_QUEUED
                    task.asr_running_since = None
                task.last_error = None
                task.last_event = None
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
                self._accumulate_stage_timer(task, _STAGE_VAD)
                task.asr_elapsed_sec = 0.0
                task.asr_running_since = None
                all_records = self.database.list_audio_base_files(task.base_name, start_sequence_number=1)
                task.asr_total_audio_sec = round(sum(max(0.0, row.duration_sec) for row in all_records), 3)
                task.asr_processed_audio_sec = 0.0
                if task.vad_total_audio_sec > 0:
                    task.vad_processed_audio_sec = task.vad_total_audio_sec
                task.last_event = f"VAD total elapsed: {task.vad_elapsed_sec:.1f}s"
                task.vad_source_dir = None
                task.vad_source_paths = []
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
                    self._accumulate_stage_timer(task, task.stage)
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
                    self._start_stage_timer(task, task.stage)
                task.updated_at = self._now_iso()
                self._save_tasks()

            try:
                if task.stage == _STAGE_VAD:
                    self._run_vad_task(task)
                else:
                    self._run_task(task)
            except Exception as exc:  # pragma: no cover - safety net
                with self._condition:
                    self._accumulate_stage_timer(task, task.stage)
                    task.status = _TASK_FAILED
                    task.last_error = str(exc)
                    task.updated_at = self._now_iso()
                    self._save_tasks()

    def _run_vad_task(self, task: QueueTask) -> None:
        source_paths = [Path(path) for path in task.vad_source_paths if str(path).strip()]
        source_paths = [path for path in source_paths if path.exists()]

        if not source_paths:
            raise RuntimeError("VAD has no valid source files.")

        self.checkpoint_vad(task.task_id)
        record = self.audio_base_service.export_sources_as_single_base_clip(
            base_name=task.base_name,
            source_paths=source_paths,
            created_at=task.vad_created_at or task.created_at,
        )

        speech_segments = self.audio_base_service.detect_source_speech_segments(Path(record.file_path))
        speech_total_sec = round(sum(max(0.0, end - start) for start, end in speech_segments), 3)

        with self._condition:
            task.vad_processed_audio_sec = max(task.vad_total_audio_sec, record.duration_sec)
            task.vad_next_source_index = max(task.vad_total_sources, len(source_paths)) + 1
            task.vad_next_segment_index = len(speech_segments)
            task.vad_next_sequence_number = 2
            task.updated_at = self._now_iso()
            self._save_tasks()

        records = [record]
        base_record = AudioBaseRecord(
            base_name=task.base_name,
            base_path=str(self.audio_base_service.base_path(task.base_name)),
            created_at=task.vad_created_at or task.created_at,
            updated_at=self._now_iso(),
        )
        self.database.create_audio_base(base_record)
        self.database.replace_audio_base_files(task.base_name, records)
        metadata_path = self.audio_base_service.update_base_metadata(
            task.base_name,
            {
                "base_path": str(self.audio_base_service.base_path(task.base_name)),
                "audio_count": len(records),
                "vad_total_elapsed_sec": round(task.vad_elapsed_sec, 3),
                "vad_speech_segment_count": len(speech_segments),
                "vad_speech_total_sec": speech_total_sec,
                "vad_speech_segments": [
                    {
                        "start_sec": round(float(start), 3),
                        "end_sec": round(float(end), 3),
                    }
                    for start, end in speech_segments[:2000]
                ],
                "vad_completed_at": self._now_iso(),
            },
        )
        with self._condition:
            task.last_event = f"VAD total elapsed: {task.vad_elapsed_sec:.1f}s (saved {metadata_path.name})"
            task.updated_at = self._now_iso()
            self._save_tasks()
        self.activate_asr_stage(task.task_id, total_files=len(records))
        self.audio_base_service.clear_staged_sources(task.base_name)
        self.audio_base_service.clear_vad_job_storage(task.task_id)

    def _run_task(self, task: QueueTask) -> None:
        file_records = self.database.list_audio_base_files(task.base_name, task.next_sequence_number)
        with self._condition:
            if task.asr_total_audio_sec <= 0.0:
                all_records = self.database.list_audio_base_files(task.base_name, start_sequence_number=1)
                task.asr_total_audio_sec = round(sum(max(0.0, row.duration_sec) for row in all_records), 3)
                self._save_tasks()

        for record in file_records:
            with self._condition:
                if task.cancel_requested:
                    self._accumulate_stage_timer(task, _STAGE_ASR)
                    task.status = _TASK_DISCARDED
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                    return
                if task.status == _TASK_PAUSED:
                    self._accumulate_stage_timer(task, _STAGE_ASR)
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                    return

            with self._condition:
                completed_before_record = max(0.0, float(task.asr_processed_audio_sec))
                record_duration = max(0.0, float(record.duration_sec))

            def _on_asr_progress(processed_audio_sec: float) -> None:
                with self._condition:
                    total_audio_sec = max(0.0, float(task.asr_total_audio_sec))
                    current = completed_before_record + max(0.0, float(processed_audio_sec))
                    if total_audio_sec > 0.0:
                        current = min(current, total_audio_sec)
                    task.asr_processed_audio_sec = max(float(task.asr_processed_audio_sec), round(current, 3))
                    task.last_event = (
                        f"ASR progress: {task.asr_processed_audio_sec:.1f}s/"
                        f"{task.asr_total_audio_sec:.1f}s"
                    )
                    task.updated_at = self._now_iso()
                    self._save_tasks()

            try:
                source_record, occurrences, result = self.asr_service.ingest(
                    source_path=record.file_path,
                    source_audio_id=record.source_audio_id,
                    base_name=record.base_name,
                    language=(str(task.asr_language).strip().lower() or self.settings.asr_default_language),
                    model_tier=task.model_tier,
                    progress_callback=_on_asr_progress,
                )
            except ASRTranscriptionError as exc:
                with self._condition:
                    self._accumulate_stage_timer(task, _STAGE_ASR)
                    task.status = _TASK_PAUSED
                    task.last_error = (
                        f"ASR paused at seq={record.sequence_number} ({record.source_audio_id}): {exc}"
                    )
                    task.last_event = exc.runtime_events[-1] if exc.runtime_events else "ASR failed and task was paused."
                    task.updated_at = self._now_iso()
                    self._save_tasks()
                return

            runtime_events = self.asr_service.consume_runtime_events()
            if runtime_events:
                with self._condition:
                    task.last_event = runtime_events[-1]
                    task.updated_at = self._now_iso()
                    self._save_tasks()
            self.index_service.ingest(source_record, occurrences)

            with self._condition:
                task.processed_files += 1
                task.token_count += int(result.token_count)
                task.asr_last_completed_sequence = int(record.sequence_number)
                task.next_sequence_number = record.sequence_number + 1
                task.asr_processed_audio_sec = max(
                    float(task.asr_processed_audio_sec),
                    round(completed_before_record + record_duration, 3),
                )
                if task.asr_total_audio_sec > 0.0:
                    task.asr_processed_audio_sec = min(task.asr_processed_audio_sec, task.asr_total_audio_sec)
                task.updated_at = self._now_iso()
                self._save_tasks()

        with self._condition:
            if task.status == _TASK_RUNNING:
                self._accumulate_stage_timer(task, _STAGE_ASR)
                task.status = _TASK_COMPLETED
                index_summary = self.database.get_base_index_summary(task.base_name)
                top_words = self.database.list_top_words_for_base(task.base_name, limit=50)
                metadata_path = self.audio_base_service.update_base_metadata(
                    task.base_name,
                    {
                        "asr_total_elapsed_sec": round(task.asr_elapsed_sec, 3),
                        "asr_completed_at": self._now_iso(),
                        "indexed_sources": index_summary["indexed_sources"],
                        "indexed_occurrences": index_summary["indexed_occurrences"],
                        "distinct_tokens": index_summary["distinct_tokens"],
                        "top_words": top_words,
                    },
                )
                removed_vad_files = self.audio_base_service.clear_vad_suffix_files(task.base_name)
                task.last_event = f"ASR total elapsed: {task.asr_elapsed_sec:.1f}s (saved {metadata_path.name})"
                if removed_vad_files > 0:
                    task.last_event = f"{task.last_event}; removed {removed_vad_files} _vad files"
                if task.asr_total_audio_sec > 0.0:
                    task.asr_processed_audio_sec = task.asr_total_audio_sec
                task.updated_at = self._now_iso()
                self._save_tasks()


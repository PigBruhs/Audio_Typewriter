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
from .asr_service import ASRService
from .index_service import IndexService

_TASK_RUNNING = "running"
_TASK_QUEUED = "queued"
_TASK_PAUSED = "paused"
_TASK_COMPLETED = "completed"
_TASK_FAILED = "failed"
_TASK_DISCARDED = "discarded"


@dataclass(slots=True)
class QueueTask:
    task_id: str
    base_name: str
    status: str
    total_files: int
    processed_files: int
    next_sequence_number: int
    token_count: int
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
        runtime_settings: Settings | None = None,
    ) -> None:
        self.settings = runtime_settings or settings
        self.database = database
        self.asr_service = asr_service
        self.index_service = index_service
        self.queue_path = self.settings.data_dir / "asr_task_queue.json"
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._tasks: list[QueueTask] = []
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
            loaded.append(task)
        self._tasks = loaded
        self._save_tasks()

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
        model_tier: str = "large",
        overwritten: bool = False,
        cleared_audio_files: int = 0,
        cleared_index_sources: int = 0,
    ) -> dict[str, object]:
        now = self._now_iso()
        task = QueueTask(
            task_id=str(uuid.uuid4()),
            base_name=base_name,
            status=_TASK_QUEUED,
            total_files=total_files,
            processed_files=0,
            next_sequence_number=1,
            token_count=0,
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
                task.status = _TASK_QUEUED
                task.last_error = None
                task.updated_at = self._now_iso()
                self._save_tasks()
                self._condition.notify_all()
                return task.to_dict()
        raise ValueError(f"Task not found: {task_id}")

    def flush(self) -> None:
        with self._lock:
            self._save_tasks()

    def _next_runnable_task(self) -> QueueTask | None:
        for task in self._tasks:
            if task.status == _TASK_QUEUED:
                return task
        return None

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                task = self._next_runnable_task()
                if task is None:
                    self._condition.wait(timeout=1.0)
                    continue
                task.status = _TASK_RUNNING
                task.updated_at = self._now_iso()
                self._save_tasks()

            try:
                self._run_task(task)
            except Exception as exc:  # pragma: no cover - safety net
                with self._condition:
                    task.status = _TASK_FAILED
                    task.last_error = str(exc)
                    task.updated_at = self._now_iso()
                    self._save_tasks()

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
                task.next_sequence_number = record.sequence_number + 1
                task.updated_at = self._now_iso()
                self._save_tasks()

        with self._condition:
            if task.status == _TASK_RUNNING:
                task.status = _TASK_COMPLETED
                task.updated_at = self._now_iso()
                self._save_tasks()


from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from ..core.config import Settings, settings
from ..models import AudioBaseFileRecord, AudioBaseRecord, AudioBaseStats

_ALLOWED_EXTENSIONS = {".wav", ".mp3"}
_BASE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class AudioBaseService:
    def __init__(self, runtime_settings: Settings | None = None) -> None:
        self.settings = runtime_settings or settings

    def validate_base_name(self, base_name: str) -> str:
        normalized = (base_name or "").strip()
        if not _BASE_NAME_PATTERN.fullmatch(normalized):
            raise ValueError("Base name must match [a-zA-Z0-9_-] and be 1-64 chars.")
        return normalized

    def base_path(self, base_name: str) -> Path:
        return self.settings.audio_base_dir / base_name

    def _probe_duration(self, file_path: Path) -> float:
        command = [
            self.settings.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            return 0.0
        try:
            return float((completed.stdout or "0").strip())
        except ValueError:
            return 0.0

    def import_audio_files(self, base_name: str, files: list[UploadFile]) -> tuple[AudioBaseRecord, list[AudioBaseFileRecord]]:
        self.settings.ensure_directories()
        base_name = self.validate_base_name(base_name)
        target_dir = self.base_path(base_name)
        if target_dir.exists():
            raise ValueError(f"Audio base '{base_name}' already exists.")

        filtered = [file for file in files if Path(file.filename or "").suffix.lower() in _ALLOWED_EXTENSIONS]
        if not filtered:
            raise ValueError("No .wav or .mp3 files were provided.")

        filtered.sort(key=lambda item: (item.filename or "").lower())
        target_dir.mkdir(parents=True, exist_ok=False)

        now = datetime.now(timezone.utc).isoformat()
        records: list[AudioBaseFileRecord] = []
        try:
            for index, upload in enumerate(filtered, start=1):
                extension = Path(upload.filename or "").suffix.lower()
                source_audio_id = f"{base_name}:{index:06d}"
                target_name = f"{index:06d}{extension}"
                target_path = target_dir / target_name

                with target_path.open("wb") as destination:
                    shutil.copyfileobj(upload.file, destination)

                file_size_bytes = target_path.stat().st_size
                duration_sec = self._probe_duration(target_path)
                records.append(
                    AudioBaseFileRecord(
                        source_audio_id=source_audio_id,
                        base_name=base_name,
                        sequence_number=index,
                        file_name=target_name,
                        file_path=str(target_path),
                        duration_sec=duration_sec,
                        file_size_bytes=file_size_bytes,
                        created_at=now,
                    )
                )
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

        base_record = AudioBaseRecord(
            base_name=base_name,
            base_path=str(target_dir),
            created_at=now,
            updated_at=now,
        )
        return base_record, records

    def summarize_records(self, base_name: str, records: list[AudioBaseFileRecord]) -> AudioBaseStats:
        return AudioBaseStats(
            base_name=base_name,
            audio_count=len(records),
            total_duration_sec=round(sum(record.duration_sec for record in records), 3),
            total_file_size_bytes=sum(record.file_size_bytes for record in records),
        )


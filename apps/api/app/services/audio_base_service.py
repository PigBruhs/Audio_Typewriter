from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import UploadFile
import numpy as np
import torch

from ..core.config import Settings, settings
from ..models import AudioBaseFileRecord, AudioBaseRecord, AudioBaseStats

try:
    from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    get_speech_timestamps = None  # type: ignore[assignment]
    load_silero_vad = None  # type: ignore[assignment]

_ALLOWED_EXTENSIONS = {".wav", ".mp3"}
_BASE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class AudioBaseService:
    def __init__(self, runtime_settings: Settings | None = None) -> None:
        self.settings = runtime_settings or settings
        self._silero_model: object | None = None

    def validate_base_name(self, base_name: str) -> str:
        normalized = (base_name or "").strip()
        if not _BASE_NAME_PATTERN.fullmatch(normalized):
            raise ValueError("Base name must match [a-zA-Z0-9_-] and be 1-64 chars.")
        return normalized

    def base_path(self, base_name: str) -> Path:
        return self.settings.audio_base_dir / base_name

    def base_exists(self, base_name: str) -> bool:
        return self.base_path(base_name).exists()

    def clear_base_storage(self, base_name: str) -> int:
        target_dir = self.base_path(base_name)
        if not target_dir.exists():
            return 0
        removed_file_count = sum(1 for path in target_dir.rglob("*") if path.is_file())
        shutil.rmtree(target_dir, ignore_errors=True)
        return removed_file_count

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

    def _get_silero_model(self) -> object:
        if load_silero_vad is None or get_speech_timestamps is None:
            raise RuntimeError(
                "silero-vad is required for base import segmentation. Install dependencies and restart the API."
            )
        if self._silero_model is None:
            self._silero_model = load_silero_vad()
        return self._silero_model

    def _read_audio_for_vad(self, source_path: Path) -> torch.Tensor:
        command = [
            self.settings.ffmpeg_binary,
            "-v",
            "error",
            "-i",
            str(source_path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ]
        completed = subprocess.run(command, capture_output=True)
        if completed.returncode != 0:
            stderr_text = completed.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg audio decode for VAD failed: {stderr_text}")

        pcm = np.frombuffer(completed.stdout, dtype=np.int16)
        if pcm.size == 0:
            return torch.zeros(0, dtype=torch.float32)
        return torch.from_numpy((pcm.astype(np.float32) / 32768.0).copy())

    def _detect_speech_segments(self, source_path: Path) -> list[tuple[float, float]]:
        model = self._get_silero_model()
        waveform = self._read_audio_for_vad(source_path.resolve())
        timestamps = get_speech_timestamps(
            waveform,
            model,
            sampling_rate=16000,
            return_seconds=True,
        )

        # Keep a tiny margin around each speech range to avoid chopping phonemes.
        segments: list[tuple[float, float]] = []
        for segment in timestamps:
            start_sec = max(0.0, float(segment["start"]) - 0.05)
            end_sec = max(start_sec, float(segment["end"]) + 0.05)
            if end_sec - start_sec < 0.08:
                continue
            segments.append((start_sec, end_sec))
        return segments

    def _export_segment_clip(self, source_path: Path, target_path: Path, start_sec: float, end_sec: float) -> None:
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-i",
            str(source_path),
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(target_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg segment export failed:\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

    def _normalize_audio_for_vad(self, source_path: Path, work_dir: Path) -> Path:
        normalized_path = work_dir / f"{source_path.stem}_vad.wav"
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(normalized_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg normalization for Silero failed:\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return normalized_path

    def _split_upload_into_speech_clips(
        self,
        source_path: Path,
        *,
        base_name: str,
        target_dir: Path,
        sequence_start: int,
        created_at: str,
    ) -> tuple[list[AudioBaseFileRecord], int]:
        work_dir = source_path.parent
        vad_source = self._normalize_audio_for_vad(source_path, work_dir)

        records: list[AudioBaseFileRecord] = []
        sequence = sequence_start
        for start_sec, end_sec in self._detect_speech_segments(vad_source):
            target_name = f"{sequence:06d}.wav"
            target_path = target_dir / target_name
            self._export_segment_clip(source_path, target_path, start_sec, end_sec)
            records.append(
                AudioBaseFileRecord(
                    source_audio_id=f"{base_name}:{sequence:06d}",
                    base_name=base_name,
                    sequence_number=sequence,
                    file_name=target_name,
                    file_path=str(target_path),
                    duration_sec=self._probe_duration(target_path),
                    file_size_bytes=target_path.stat().st_size,
                    created_at=created_at,
                )
            )
            sequence += 1
        return records, sequence

    def import_audio_files(self, base_name: str, files: list[UploadFile]) -> tuple[AudioBaseRecord, list[AudioBaseFileRecord]]:
        return self.import_audio_files_with_progress(base_name, files, progress_callback=None)

    def import_audio_files_with_progress(
        self,
        base_name: str,
        files: list[UploadFile],
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> tuple[AudioBaseRecord, list[AudioBaseFileRecord]]:
        self.settings.ensure_directories()
        base_name = self.validate_base_name(base_name)
        target_dir = self.base_path(base_name)
        if target_dir.exists():
            raise ValueError(f"Audio base '{base_name}' already exists. Clear it before importing.")

        filtered = [file for file in files if Path(file.filename or "").suffix.lower() in _ALLOWED_EXTENSIONS]
        if not filtered:
            raise ValueError("No .wav or .mp3 files were provided.")

        filtered.sort(key=lambda item: (item.filename or "").lower())
        target_dir.mkdir(parents=True, exist_ok=False)

        now = datetime.now(timezone.utc).isoformat()
        records: list[AudioBaseFileRecord] = []
        sequence = 1
        try:
            with tempfile.TemporaryDirectory(
                prefix="audio_typewriter_raw_",
                dir=str(self.settings.temp_dir),
            ) as temp_dir:
                work_dir = Path(temp_dir)
                prepared_sources: list[tuple[Path, float, str]] = []
                for index, upload in enumerate(filtered, start=1):
                    raw_name = Path(upload.filename or f"upload_{index:06d}.wav")
                    source_path = work_dir / raw_name.name
                    upload.file.seek(0)
                    with source_path.open("wb") as destination:
                        shutil.copyfileobj(upload.file, destination)
                    source_duration = self._probe_duration(source_path)
                    prepared_sources.append((source_path, source_duration, raw_name.name))

                total_audio_sec = round(sum(duration for _path, duration, _name in prepared_sources), 3)
                processed_audio_sec = 0.0
                if progress_callback:
                    progress_callback(
                        {
                            "type": "vad_start",
                            "base_name": base_name,
                            "total_audio_sec": total_audio_sec,
                            "processed_audio_sec": processed_audio_sec,
                        }
                    )
                for source_path, source_duration, source_name in prepared_sources:
                    split_records, sequence = self._split_upload_into_speech_clips(
                        source_path,
                        base_name=base_name,
                        target_dir=target_dir,
                        sequence_start=sequence,
                        created_at=now,
                    )
                    records.extend(split_records)
                    processed_audio_sec = round(processed_audio_sec + source_duration, 3)
                    if progress_callback:
                        progress_callback(
                            {
                                "type": "vad_progress",
                                "base_name": base_name,
                                "file_name": source_name,
                                "processed_audio_sec": processed_audio_sec,
                                "total_audio_sec": total_audio_sec,
                            }
                        )

                if progress_callback:
                    progress_callback(
                        {
                            "type": "vad_complete",
                            "base_name": base_name,
                            "processed_audio_sec": processed_audio_sec,
                            "total_audio_sec": total_audio_sec,
                        }
                    )

            if not records:
                raise ValueError("No speech segments detected by Silero VAD. Nothing was imported.")
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


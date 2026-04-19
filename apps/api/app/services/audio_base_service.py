from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
import time
import json
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

    def base_metadata_path(self, base_name: str) -> Path:
        return self.base_path(base_name) / "base_metadata.json"

    def vad_job_dir(self, task_id: str) -> Path:
        return self.settings.temp_dir / "vad_jobs" / task_id

    def base_exists(self, base_name: str) -> bool:
        return self.base_path(base_name).exists()

    def clear_base_storage(self, base_name: str) -> int:
        target_dir = self.base_path(base_name)
        if not target_dir.exists():
            return 0
        removed_file_count = sum(1 for path in target_dir.rglob("*") if path.is_file())
        shutil.rmtree(target_dir, ignore_errors=True)
        return removed_file_count

    def clear_vad_job_storage(self, task_id: str) -> None:
        shutil.rmtree(self.vad_job_dir(task_id), ignore_errors=True)

    def update_base_metadata(self, base_name: str, patch: dict[str, object]) -> Path:
        metadata_path = self.base_metadata_path(base_name)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        current: dict[str, object] = {}
        if metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded
            except Exception:
                current = {}

        current.update(patch)
        current["base_name"] = base_name
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata_path

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

        min_clip_sec = max(0.0, float(getattr(self.settings, "vad_min_clip_sec", 20.0)))
        return self._merge_segments_by_min_duration(segments, min_clip_sec)

    def _merge_segments_by_min_duration(
        self,
        segments: list[tuple[float, float]],
        min_clip_sec: float,
    ) -> list[tuple[float, float]]:
        if not segments or min_clip_sec <= 0.0:
            return list(segments)

        merged: list[tuple[float, float]] = []
        chunk_start, chunk_end = segments[0]
        for start_sec, end_sec in segments[1:]:
            if (chunk_end - chunk_start) < min_clip_sec:
                # Extend short chunks with following sentence regions, including natural pauses.
                chunk_end = max(chunk_end, end_sec)
                continue
            merged.append((chunk_start, chunk_end))
            chunk_start, chunk_end = start_sec, end_sec

        merged.append((chunk_start, chunk_end))

        # If the tail is shorter than min duration, append it to the previous chunk.
        if len(merged) > 1:
            tail_start, tail_end = merged[-1]
            if (tail_end - tail_start) < min_clip_sec:
                prev_start, _prev_end = merged[-2]
                merged[-2] = (prev_start, tail_end)
                merged.pop()

        return merged

    def _build_split_only_segments(
        self,
        source_duration_sec: float,
        speech_segments: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        duration = max(0.0, float(source_duration_sec))
        if duration <= 0.0:
            return []

        if not speech_segments:
            return [(0.0, duration)]

        cut_points: list[float] = [0.0, duration]
        for start_sec, end_sec in speech_segments:
            clamped_start = min(max(0.0, float(start_sec)), duration)
            clamped_end = min(max(clamped_start, float(end_sec)), duration)
            if clamped_end - clamped_start < 0.001:
                continue
            cut_points.append(clamped_start)
            cut_points.append(clamped_end)

        ordered_points = sorted(cut_points)
        deduplicated_points: list[float] = []
        for value in ordered_points:
            if not deduplicated_points or abs(value - deduplicated_points[-1]) >= 1e-4:
                deduplicated_points.append(value)

        segments: list[tuple[float, float]] = []
        for index in range(len(deduplicated_points) - 1):
            start_sec = deduplicated_points[index]
            end_sec = deduplicated_points[index + 1]
            if end_sec - start_sec < 0.001:
                continue
            segments.append((start_sec, end_sec))

        if not segments:
            return [(0.0, duration)]
        return segments

    def detect_source_speech_segments(self, source_path: Path) -> list[tuple[float, float]]:
        normalized = self._normalize_audio_for_vad(source_path, source_path.parent)
        speech_segments = self._detect_speech_segments(normalized)
        source_duration = self._probe_duration(source_path)
        if source_duration <= 0.0:
            source_duration = self._probe_duration(normalized)
        return self._build_split_only_segments(source_duration, speech_segments)

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

    def export_segment_record(
        self,
        *,
        source_path: Path,
        base_name: str,
        sequence_number: int,
        start_sec: float,
        end_sec: float,
        created_at: str,
    ) -> AudioBaseFileRecord:
        target_dir = self.base_path(base_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"{sequence_number:06d}.wav"
        target_path = target_dir / target_name
        self._export_segment_clip(source_path, target_path, start_sec, end_sec)
        return AudioBaseFileRecord(
            source_audio_id=f"{base_name}:{sequence_number:06d}",
            base_name=base_name,
            sequence_number=sequence_number,
            file_name=target_name,
            file_path=str(target_path),
            duration_sec=self._probe_duration(target_path),
            file_size_bytes=target_path.stat().st_size,
            created_at=created_at,
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

    def stage_vad_sources(self, task_id: str, files: list[UploadFile]) -> tuple[str, list[dict[str, object]], float]:
        filtered = [file for file in files if Path(file.filename or "").suffix.lower() in _ALLOWED_EXTENSIONS]
        if not filtered:
            raise ValueError("No .wav or .mp3 files were provided.")

        filtered.sort(key=lambda item: (item.filename or "").lower())
        job_dir = self.vad_job_dir(task_id)
        source_dir = job_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict[str, object]] = []
        total_audio_sec = 0.0
        for index, upload in enumerate(filtered, start=1):
            suffix = Path(upload.filename or "").suffix.lower() or ".wav"
            saved_name = f"{index:06d}{suffix}"
            source_path = source_dir / saved_name
            upload.file.seek(0)
            with source_path.open("wb") as destination:
                shutil.copyfileobj(upload.file, destination)
            duration_sec = self._probe_duration(source_path)
            total_audio_sec += duration_sec
            manifest.append(
                {
                    "index": index,
                    "file_name": saved_name,
                    "duration_sec": duration_sec,
                }
            )

        (job_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(source_dir), manifest, round(total_audio_sec, 3)

    def load_vad_manifest(self, task_id: str) -> list[dict[str, object]]:
        manifest_path = self.vad_job_dir(task_id) / "manifest.json"
        if not manifest_path.exists():
            return []
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def split_source_file_into_base_clips(
        self,
        source_path: Path,
        *,
        base_name: str,
        sequence_start: int,
        created_at: str,
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> tuple[list[AudioBaseFileRecord], int]:
        target_dir = self.base_path(base_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        return self._split_upload_into_speech_clips(
            source_path,
            base_name=base_name,
            target_dir=target_dir,
            sequence_start=sequence_start,
            created_at=created_at,
            checkpoint_callback=checkpoint_callback,
        )

    def collect_base_records(self, base_name: str, created_at: str) -> list[AudioBaseFileRecord]:
        base_dir = self.base_path(base_name)
        records: list[AudioBaseFileRecord] = []
        for path in sorted(base_dir.glob("*.wav"), key=lambda item: int(item.stem) if item.stem.isdigit() else 0):
            if not path.stem.isdigit():
                continue
            seq = int(path.stem)
            records.append(
                AudioBaseFileRecord(
                    source_audio_id=f"{base_name}:{seq:06d}",
                    base_name=base_name,
                    sequence_number=seq,
                    file_name=path.name,
                    file_path=str(path),
                    duration_sec=self._probe_duration(path),
                    file_size_bytes=path.stat().st_size,
                    created_at=created_at,
                )
            )
        return records

    def _split_upload_into_speech_clips(
        self,
        source_path: Path,
        *,
        base_name: str,
        target_dir: Path,
        sequence_start: int,
        created_at: str,
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> tuple[list[AudioBaseFileRecord], int]:
        work_dir = source_path.parent
        vad_source = self._normalize_audio_for_vad(source_path, work_dir)

        records: list[AudioBaseFileRecord] = []
        sequence = sequence_start
        speech_segments = self._detect_speech_segments(vad_source)
        source_duration = self._probe_duration(source_path)
        if source_duration <= 0.0:
            source_duration = self._probe_duration(vad_source)
        for start_sec, end_sec in self._build_split_only_segments(source_duration, speech_segments):
            if checkpoint_callback:
                checkpoint_callback()
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
        checkpoint_callback: Callable[[], None] | None = None,
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
        vad_processed_sec = 0.0
        vad_last_sequence = 0
        vad_lock = threading.Lock()
        vad_stop_event = threading.Event()
        vad_progress_interval_sec = 10.0

        def _accumulate_new_vad_duration() -> bool:
            nonlocal vad_processed_sec, vad_last_sequence
            latest_sequence = 0
            for path in target_dir.glob("*.wav"):
                stem = path.stem
                if stem.isdigit():
                    latest_sequence = max(latest_sequence, int(stem))
            if latest_sequence <= vad_last_sequence:
                return False

            increment = 0.0
            for current_sequence in range(vad_last_sequence + 1, latest_sequence + 1):
                clip_path = target_dir / f"{current_sequence:06d}.wav"
                if not clip_path.exists():
                    continue
                increment += self._probe_duration(clip_path)

            vad_processed_sec = round(vad_processed_sec + increment, 3)
            vad_last_sequence = latest_sequence
            return True

        def _emit_vad_progress(total_audio_sec: float) -> None:
            if not progress_callback:
                return
            safe_total = max(total_audio_sec, vad_processed_sec, 0.001)
            progress_callback(
                {
                    "type": "vad_progress",
                    "base_name": base_name,
                    "file_name": f"{vad_last_sequence:06d}.wav",
                    "processed_audio_sec": vad_processed_sec,
                    "total_audio_sec": round(safe_total, 3),
                }
            )

        def _vad_progress_worker(total_audio_sec: float) -> None:
            while not vad_stop_event.wait(vad_progress_interval_sec):
                with vad_lock:
                    changed = _accumulate_new_vad_duration()
                    if not changed:
                        continue
                    _emit_vad_progress(total_audio_sec)

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
                vad_progress_thread: threading.Thread | None = None
                if progress_callback:
                    progress_callback(
                        {
                            "type": "vad_start",
                            "base_name": base_name,
                            "total_audio_sec": total_audio_sec,
                            "processed_audio_sec": 0.0,
                        }
                    )
                    vad_progress_thread = threading.Thread(
                        target=_vad_progress_worker,
                        args=(total_audio_sec,),
                        daemon=True,
                        name=f"vad-progress-{base_name}",
                    )
                    vad_progress_thread.start()
                for source_path, _source_duration, _source_name in prepared_sources:
                    if checkpoint_callback:
                        checkpoint_callback()
                    split_records, sequence = self._split_upload_into_speech_clips(
                        source_path,
                        base_name=base_name,
                        target_dir=target_dir,
                        sequence_start=sequence,
                        created_at=now,
                        checkpoint_callback=checkpoint_callback,
                    )
                    records.extend(split_records)
                    with vad_lock:
                        if _accumulate_new_vad_duration():
                            _emit_vad_progress(total_audio_sec)

                vad_stop_event.set()
                if vad_progress_thread is not None:
                    vad_progress_thread.join(timeout=1.0)
                with vad_lock:
                    if _accumulate_new_vad_duration():
                        _emit_vad_progress(total_audio_sec)

                if progress_callback:
                    safe_total = max(total_audio_sec, vad_processed_sec, 0.001)
                    progress_callback(
                        {
                            "type": "vad_complete",
                            "base_name": base_name,
                            "processed_audio_sec": vad_processed_sec,
                            "total_audio_sec": round(safe_total, 3),
                        }
                    )

            if not records:
                raise ValueError("No valid segments were produced from input audio.")
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


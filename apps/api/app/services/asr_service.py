from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Callable
import wave

from ..core.config import Settings, settings
from ..models import AudioSourceRecord, IngestResult, WordOccurrenceRecord
from ..text import normalize_word

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None  # type: ignore[assignment]

try:
    import ctranslate2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ctranslate2 = None  # type: ignore[assignment]


@dataclass(slots=True)
class TranscriptionOutput:
    source_audio_id: str
    device_used: str
    compute_type: str
    occurrences: list[WordOccurrenceRecord]


@dataclass(slots=True)
class ModelDownloadResult:
    model_name: str
    status: str
    device_used: str
    compute_type: str
    cache_dir: str


class ASRTranscriptionError(RuntimeError):
    def __init__(self, detail: str, runtime_events: list[str] | None = None) -> None:
        super().__init__(detail)
        self.runtime_events = list(runtime_events or [])


class ASRService:
    _MIN_WORD_DURATION_SEC = 0.001

    def __init__(
        self,
        runtime_settings: Settings | None = None,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        self.settings = runtime_settings or settings
        self.model_factory = model_factory or WhisperModel
        self.last_device_used = "cpu"
        self.last_compute_type = self.settings.asr_cpu_compute_type
        self._model_cache: dict[tuple[str, str, str], object] = {}
        self._model_cache_lock = threading.RLock()
        self.last_runtime_events: list[str] = []

    def consume_runtime_events(self) -> list[str]:
        events = list(self.last_runtime_events)
        self.last_runtime_events = []
        return events

    def _cuda_available(self) -> bool:
        if ctranslate2 is not None:
            try:
                supported_devices = set(ctranslate2.get_supported_devices())
                if "cuda" in supported_devices:
                    return True
            except Exception:
                pass

        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def resolve_runtime(self) -> tuple[str, str]:
        preferred_device = (self.settings.asr_device or "cuda").lower()
        if preferred_device == "cpu":
            return "cpu", self.settings.asr_cpu_compute_type
        if preferred_device == "cuda":
            if self._cuda_available():
                return "cuda", self.settings.asr_compute_type
            return "cpu", self.settings.asr_cpu_compute_type
        if preferred_device == "auto":
            if self._cuda_available():
                return "cuda", self.settings.asr_compute_type
            return "cpu", self.settings.asr_cpu_compute_type
        return "cpu", self.settings.asr_cpu_compute_type

    def _create_model(self, model_name: str, device: str, compute_type: str) -> object:
        if self.model_factory is None:
            raise RuntimeError(
                "faster-whisper is not installed. Install the project dependencies before running ASR."
            )
        return self.model_factory(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(self.settings.asr_model_cache_dir),
        )

    def _get_or_create_model(self, model_name: str, device: str, compute_type: str) -> object:
        key = (model_name, device, compute_type)
        with self._model_cache_lock:
            model = self._model_cache.get(key)
            if model is not None:
                return model
            model = self._create_model(model_name, device, compute_type)
            self._model_cache[key] = model
            return model

    def _probe_wav_duration_sec(self, source_path: str) -> float | None:
        path = Path(source_path)
        if path.suffix.lower() != ".wav":
            return None
        try:
            with wave.open(str(path), "rb") as wav_file:
                framerate = wav_file.getframerate()
                if framerate <= 0:
                    return None
                return float(wav_file.getnframes()) / float(framerate)
        except Exception:
            return None

    def _resolve_clip_runtime(self, source_path: str, device: str, compute_type: str) -> tuple[str, str]:
        if device != "cuda":
            return device, compute_type
        clip_duration = self._probe_wav_duration_sec(source_path)
        if clip_duration is None:
            return device, compute_type
        threshold = max(0.0, float(getattr(self.settings, "asr_cuda_short_audio_cpu_threshold_sec", 0.8)))
        if 0.0 < clip_duration <= threshold:
            return "cpu", self.settings.asr_cpu_compute_type
        return device, compute_type

    def _should_use_internal_vad(self, source_path: str) -> bool:
        try:
            source = Path(source_path).resolve()
            audio_base_root = self.settings.audio_base_dir.resolve()
            return audio_base_root not in source.parents
        except Exception:
            return True

    def download_model(
        self,
        *,
        model_tier: str = "large",
        language: str = "en",
        model_name: str | None = None,
    ) -> ModelDownloadResult:
        self.settings.ensure_directories()
        resolved_name = self.settings.resolve_model_name(model_tier, language, model_name=model_name)
        device, compute_type = self.resolve_runtime()
        try:
            self._get_or_create_model(resolved_name, device, compute_type)
        except Exception:
            if device == "cuda":
                device = "cpu"
                compute_type = self.settings.asr_cpu_compute_type
                self._get_or_create_model(resolved_name, device, compute_type)
            else:
                raise

        self.last_device_used = device
        self.last_compute_type = compute_type
        return ModelDownloadResult(
            model_name=resolved_name,
            status="downloaded",
            device_used=device,
            compute_type=compute_type,
            cache_dir=str(self.settings.asr_model_cache_dir),
        )

    def preload_default_model_if_configured(self) -> ModelDownloadResult | None:
        if not self.settings.asr_preload_model.strip():
            return None
        return self.download_model(model_name=self.settings.asr_preload_model)

    def _transcribe_with_model(
        self,
        model: object,
        source_path: str,
        language: str,
        *,
        vad_filter: bool,
    ) -> list[WordOccurrenceRecord]:
        segments, _info = model.transcribe(  # type: ignore[attr-defined]
            source_path,
            language=language,
            beam_size=self.settings.asr_beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
        )
        occurrences: list[WordOccurrenceRecord] = []

        def _to_float(value: object, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        for segment_index, segment in enumerate(segments):
            words = getattr(segment, "words", None)
            if not words:
                continue

            raw_words: list[tuple[str, str, float, float, float, int]] = []
            for word_index, word in enumerate(words):
                token_text = getattr(word, "word", "").strip()
                normalized_token = normalize_word(token_text)
                if not normalized_token:
                    continue
                start_sec = max(0.0, _to_float(getattr(word, "start", 0.0), 0.0))
                end_sec = max(start_sec + self._MIN_WORD_DURATION_SEC, _to_float(getattr(word, "end", start_sec), start_sec))
                confidence = _to_float(getattr(word, "probability", 0.0), 0.0)
                raw_words.append((token_text, normalized_token, start_sec, end_sec, confidence, word_index))

            if not raw_words:
                continue

            for token_text, normalized_token, start_sec, end_sec, confidence, original_word_index in raw_words:
                occurrences.append(
                    WordOccurrenceRecord(
                        id=None,
                        source_audio_id=source_path,
                        token=token_text,
                        normalized_token=normalized_token,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        confidence=confidence,
                        segment_index=segment_index,
                        word_index=original_word_index,
                    )
                )
        return occurrences

    def transcribe(
        self,
        source_path: str,
        language: str = "en",
        model_tier: str = "large",
        model_name: str | None = None,
    ) -> list[WordOccurrenceRecord]:
        resolved_name = self.settings.resolve_model_name(model_tier, language, model_name=model_name)
        preferred_device, preferred_compute_type = self.resolve_runtime()
        device, compute_type = self._resolve_clip_runtime(source_path, preferred_device, preferred_compute_type)
        vad_filter = self._should_use_internal_vad(source_path)
        runtime_events: list[str] = []
        if preferred_device == "cuda" and device == "cpu":
            runtime_events.append(
                f"Short clip detected for {source_path}; use CPU ({compute_type}) to avoid CUDA instability."
            )

        occurrences: list[WordOccurrenceRecord] = []
        try:
            model = self._get_or_create_model(resolved_name, device, compute_type)
            occurrences = self._transcribe_with_model(model, source_path, language, vad_filter=vad_filter)
        except Exception as primary_exc:
            if device == "cuda":
                runtime_events.append(
                    f"CUDA failed for {source_path}: {type(primary_exc).__name__}: {primary_exc}. Retrying on CPU."
                )
                device = "cpu"
                compute_type = self.settings.asr_cpu_compute_type
                try:
                    model = self._get_or_create_model(resolved_name, device, compute_type)
                    occurrences = self._transcribe_with_model(model, source_path, language, vad_filter=vad_filter)
                    runtime_events.append(f"CPU fallback succeeded for {source_path}.")
                except Exception as cpu_exc:
                    detail = (
                        f"ASR failed on both CUDA and CPU for {source_path}. "
                        f"CUDA error: {type(primary_exc).__name__}: {primary_exc}; "
                        f"CPU error: {type(cpu_exc).__name__}: {cpu_exc}"
                    )
                    runtime_events.append(
                        f"CPU fallback failed for {source_path}: {type(cpu_exc).__name__}: {cpu_exc}."
                    )
                    self.last_runtime_events = runtime_events
                    raise ASRTranscriptionError(detail, runtime_events=runtime_events) from cpu_exc
            else:
                detail = f"ASR failed for {source_path} on {device}: {type(primary_exc).__name__}: {primary_exc}"
                self.last_runtime_events = runtime_events
                raise ASRTranscriptionError(detail, runtime_events=runtime_events) from primary_exc

        self.last_device_used = device
        self.last_compute_type = compute_type
        self.last_runtime_events = runtime_events
        return occurrences

    def ingest(
        self,
        source_path: str,
        source_audio_id: str | None = None,
        base_name: str | None = None,
        language: str = "en",
        model_tier: str = "large",
        model_name: str | None = None,
    ) -> tuple[AudioSourceRecord, list[WordOccurrenceRecord], IngestResult]:
        occurrences = self.transcribe(
            source_path,
            language=language,
            model_tier=model_tier,
            model_name=model_name,
        )
        now = datetime.now(timezone.utc).isoformat()
        resolved_source_audio_id = source_audio_id or source_path
        resolved_base_name = (base_name or "").strip()
        source_record = AudioSourceRecord(
            source_audio_id=resolved_source_audio_id,
            base_name=resolved_base_name,
            source_path=source_path,
            language=language,
            model_tier=model_tier,
            device=self.last_device_used,
            compute_type=self.last_compute_type,
            created_at=now,
            updated_at=now,
        )
        for row in occurrences:
            row.source_audio_id = resolved_source_audio_id
        result = IngestResult(
            source_audio_id=resolved_source_audio_id,
            base_name=resolved_base_name,
            status="completed",
            token_count=len(occurrences),
            device_used=self.last_device_used,
            compute_type=self.last_compute_type,
        )
        return source_record, occurrences, result


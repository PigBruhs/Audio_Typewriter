from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

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


class ASRService:
    def __init__(
        self,
        runtime_settings: Settings | None = None,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        self.settings = runtime_settings or settings
        self.model_factory = model_factory or WhisperModel
        self.last_device_used = "cpu"
        self.last_compute_type = self.settings.asr_cpu_compute_type

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
            self._create_model(resolved_name, device, compute_type)
        except Exception:
            if device == "cuda":
                device = "cpu"
                compute_type = self.settings.asr_cpu_compute_type
                self._create_model(resolved_name, device, compute_type)
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

    def _transcribe_with_model(self, model: object, source_path: str, language: str) -> list[WordOccurrenceRecord]:
        segments, _info = model.transcribe(  # type: ignore[attr-defined]
            source_path,
            language=language,
            beam_size=self.settings.asr_beam_size,
            word_timestamps=True,
            vad_filter=True,
        )
        occurrences: list[WordOccurrenceRecord] = []
        for segment_index, segment in enumerate(segments):
            words = getattr(segment, "words", None)
            if not words:
                continue
            for word_index, word in enumerate(words):
                token_text = getattr(word, "word", "").strip()
                normalized_token = normalize_word(token_text)
                if not normalized_token:
                    continue
                occurrences.append(
                    WordOccurrenceRecord(
                        id=None,
                        source_audio_id=source_path,
                        token=token_text,
                        normalized_token=normalized_token,
                        start_sec=float(getattr(word, "start", 0.0)),
                        end_sec=float(getattr(word, "end", 0.0)),
                        confidence=float(getattr(word, "probability", 0.0)),
                        segment_index=segment_index,
                        word_index=word_index,
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
        device, compute_type = self.resolve_runtime()

        occurrences: list[WordOccurrenceRecord] = []
        try:
            model = self._create_model(resolved_name, device, compute_type)
            occurrences = self._transcribe_with_model(model, source_path, language)
        except Exception:
            if device == "cuda":
                device = "cpu"
                compute_type = self.settings.asr_cpu_compute_type
                model = self._create_model(resolved_name, device, compute_type)
                occurrences = self._transcribe_with_model(model, source_path, language)
            else:
                raise

        self.last_device_used = device
        self.last_compute_type = compute_type
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


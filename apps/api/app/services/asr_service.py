from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import shutil
import subprocess
import threading
import tempfile
import time
from pathlib import Path
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

try:
    import whisperx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    whisperx = None  # type: ignore[assignment]


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
        self._align_model_cache: dict[tuple[str, str], tuple[object, object]] = {}
        self._align_model_cache_lock = threading.RLock()
        self.last_runtime_events: list[str] = []

    def consume_runtime_events(self) -> list[str]:
        events = list(self.last_runtime_events)
        self.last_runtime_events = []
        return events

    def _cuda_available(self) -> bool:
        if ctranslate2 is not None:
            # Newer ctranslate2 exposes CUDA availability via get_cuda_device_count().
            try:
                get_cuda_device_count = getattr(ctranslate2, "get_cuda_device_count", None)
                if callable(get_cuda_device_count) and int(get_cuda_device_count()) > 0:
                    return True
            except Exception:
                pass

            # Backward compatibility for older ctranslate2 builds.
            try:
                get_supported_devices = getattr(ctranslate2, "get_supported_devices", None)
                if callable(get_supported_devices):
                    supported_devices = set(get_supported_devices())
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

    def _get_or_create_align_model(self, language: str, device: str) -> tuple[object, object]:
        if whisperx is None:
            raise RuntimeError("whisperx is not installed")
        language_code = (language or "en").lower()
        key = (language_code, device)
        with self._align_model_cache_lock:
            cached = self._align_model_cache.get(key)
            if cached is not None:
                return cached
            align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            self._align_model_cache[key] = (align_model, metadata)
            return align_model, metadata

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
        progress_callback: Callable[[float], None] | None = None,
        progress_interval_sec: float = 20.0,
    ) -> tuple[list[WordOccurrenceRecord], list[dict[str, object]]]:
        segments, _info = model.transcribe(  # type: ignore[attr-defined]
            source_path,
            language=language,
            beam_size=self.settings.asr_beam_size,
            word_timestamps=True,
        )
        occurrences: list[WordOccurrenceRecord] = []
        align_segments: list[dict[str, object]] = []
        last_progress_emit = time.monotonic()
        latest_segment_end = 0.0

        def _to_float(value: object, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _is_cjk(char: str) -> bool:
            if not char:
                return False
            code = ord(char)
            return 0x3400 <= code <= 0x9FFF

        is_chinese = str(language or "").lower().startswith("zh")

        for segment_index, segment in enumerate(segments):
            words = getattr(segment, "words", None)
            if not words:
                continue

            segment_start = _to_float(getattr(segment, "start", 0.0), 0.0)
            segment_end = max(segment_start, _to_float(getattr(segment, "end", segment_start), segment_start))
            latest_segment_end = max(latest_segment_end, segment_end)
            segment_text = str(getattr(segment, "text", "") or "").strip()

            raw_words: list[tuple[str, str, float, float, float, int]] = []
            next_word_index = 0
            for word_index, word in enumerate(words):
                token_text = getattr(word, "word", "").strip()
                normalized_token = normalize_word(token_text)
                if not normalized_token:
                    continue
                start_sec = max(0.0, _to_float(getattr(word, "start", 0.0), 0.0))
                end_sec = max(start_sec, _to_float(getattr(word, "end", start_sec), start_sec))
                confidence = _to_float(getattr(word, "probability", 0.0), 0.0)
                if is_chinese and len(normalized_token) > 1 and any(_is_cjk(ch) for ch in normalized_token):
                    pieces = [ch for ch in normalized_token if _is_cjk(ch)]
                    if pieces:
                        piece_total = len(pieces)
                        span = max(0.0, end_sec - start_sec)
                        for piece_idx, piece in enumerate(pieces):
                            piece_start = start_sec + (span * piece_idx / piece_total)
                            piece_end = start_sec + (span * (piece_idx + 1) / piece_total)
                            raw_words.append((piece, piece, piece_start, max(piece_start, piece_end), confidence, next_word_index))
                            next_word_index += 1
                        continue
                raw_words.append((token_text, normalized_token, start_sec, end_sec, confidence, next_word_index))
                next_word_index += 1

            if not raw_words:
                continue

            if not segment_text:
                segment_text = " ".join(token_text for token_text, *_rest in raw_words)
            align_segments.append(
                {
                    "start": segment_start,
                    "end": segment_end,
                    "text": segment_text,
                }
            )

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

            if progress_callback and (time.monotonic() - last_progress_emit) >= max(1.0, float(progress_interval_sec)):
                progress_callback(latest_segment_end)
                last_progress_emit = time.monotonic()

        if progress_callback and latest_segment_end > 0.0:
            progress_callback(latest_segment_end)
        return occurrences, align_segments

    def _extract_aligned_words(self, aligned_payload: dict[str, object]) -> list[tuple[str, float, float, float]]:
        aligned_words: list[tuple[str, float, float, float]] = []

        def _to_float(value: object, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        word_segments = aligned_payload.get("word_segments")
        if isinstance(word_segments, list):
            for item in word_segments:
                if not isinstance(item, dict):
                    continue
                token_text = str(item.get("word", "") or "").strip()
                if not normalize_word(token_text):
                    continue
                start_sec = max(0.0, _to_float(item.get("start", 0.0), 0.0))
                end_sec = max(start_sec, _to_float(item.get("end", start_sec), start_sec))
                score = _to_float(item.get("score", 0.0), 0.0)
                aligned_words.append((token_text, start_sec, end_sec, score))
            if aligned_words:
                return aligned_words

        segments = aligned_payload.get("segments")
        if not isinstance(segments, list):
            return aligned_words
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            words = segment.get("words")
            if not isinstance(words, list):
                continue
            for word in words:
                if not isinstance(word, dict):
                    continue
                token_text = str(word.get("word", "") or "").strip()
                if not normalize_word(token_text):
                    continue
                start_sec = max(0.0, _to_float(word.get("start", 0.0), 0.0))
                end_sec = max(start_sec, _to_float(word.get("end", start_sec), start_sec))
                score = _to_float(word.get("score", 0.0), 0.0)
                aligned_words.append((token_text, start_sec, end_sec, score))
        return aligned_words

    def _align_occurrences_with_intervals(
        self,
        *,
        occurrences: list[WordOccurrenceRecord],
        intervals: list[tuple[float, float, str]],
        runtime_events: list[str],
        backend_label: str,
    ) -> list[WordOccurrenceRecord]:
        if not occurrences or not intervals:
            return occurrences

        updated: list[WordOccurrenceRecord] = []
        interval_index = 0
        matched = 0
        for original in occurrences:
            original_norm = normalize_word(original.token)
            while interval_index < len(intervals):
                start_sec, end_sec, interval_text = intervals[interval_index]
                interval_index += 1
                interval_norm = normalize_word(interval_text)
                if not interval_norm:
                    continue
                if interval_norm != original_norm:
                    continue
                updated.append(
                    WordOccurrenceRecord(
                        id=None,
                        source_audio_id=original.source_audio_id,
                        token=original.token,
                        normalized_token=original.normalized_token,
                        start_sec=max(0.0, float(start_sec)),
                        end_sec=max(float(start_sec), float(end_sec)),
                        confidence=original.confidence,
                        segment_index=original.segment_index,
                        word_index=original.word_index,
                    )
                )
                matched += 1
                break
            else:
                updated.append(original)

        min_required = max(1, int(len(occurrences) * 0.6))
        if matched < min_required:
            runtime_events.append(
                f"{backend_label} alignment matched only {matched}/{len(occurrences)} words; keep ASR timestamps."
            )
            return occurrences

        runtime_events.append(f"{backend_label} alignment applied to {matched}/{len(occurrences)} words.")
        return updated

    def _parse_textgrid_word_intervals(self, payload: str) -> list[tuple[float, float, str]]:
        intervals: list[tuple[float, float, str]] = []
        in_word_tier = False
        current_start: float | None = None
        current_end: float | None = None
        current_text: str | None = None

        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if line.startswith("name ="):
                tier_name = line.split("=", 1)[1].strip().strip('"').lower()
                in_word_tier = "word" in tier_name
                continue
            if not in_word_tier:
                continue
            if line.startswith("xmin ="):
                try:
                    current_start = float(line.split("=", 1)[1].strip())
                except ValueError:
                    current_start = None
                continue
            if line.startswith("xmax ="):
                try:
                    current_end = float(line.split("=", 1)[1].strip())
                except ValueError:
                    current_end = None
                continue
            if line.startswith("text ="):
                current_text = line.split("=", 1)[1].strip().strip('"')
                if current_start is None or current_end is None:
                    continue
                if current_end <= current_start:
                    continue
                intervals.append((current_start, current_end, current_text or ""))
                current_start = None
                current_end = None
                current_text = None

        return intervals

    def _run_mfa_alignment(
        self,
        *,
        source_path: str,
        transcript: str,
        runtime_events: list[str],
    ) -> list[tuple[float, float, str]] | None:
        dictionary_path = self.settings.asr_mfa_dictionary_path
        acoustic_model_path = self.settings.asr_mfa_acoustic_model_path
        if dictionary_path is None or acoustic_model_path is None:
            runtime_events.append("MFA alignment skipped: dictionary/model paths are not configured.")
            return None
        if not dictionary_path.exists() or not acoustic_model_path.exists():
            runtime_events.append("MFA alignment skipped: configured dictionary/model path does not exist.")
            return None
        if not transcript.strip():
            runtime_events.append("MFA alignment skipped: transcript is empty.")
            return None

        mfa_binary = str(self.settings.asr_mfa_binary or "mfa").strip() or "mfa"
        if shutil.which(mfa_binary) is None:
            runtime_events.append(f"MFA alignment skipped: binary not found ({mfa_binary}).")
            return None

        self.settings.ensure_directories()
        mfa_temp_root = self.settings.temp_dir / "mfa"
        mfa_temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mfa_align_", dir=str(mfa_temp_root)) as temp_dir:
            work_dir = Path(temp_dir)
            corpus_dir = work_dir / "corpus"
            output_dir = work_dir / "output"
            corpus_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            source = Path(source_path)
            corpus_audio_path = corpus_dir / "utt.wav"
            if source.suffix.lower() == ".wav":
                shutil.copy2(source, corpus_audio_path)
            else:
                runtime_events.append("MFA alignment skipped: source audio is not wav.")
                return None

            (corpus_dir / "utt.lab").write_text(transcript, encoding="utf-8")

            command = [
                mfa_binary,
                "align",
                str(corpus_dir),
                str(dictionary_path),
                str(acoustic_model_path),
                str(output_dir),
                "--clean",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                runtime_events.append(
                    f"MFA alignment failed: {completed.stderr.strip() or completed.stdout.strip() or 'unknown error'}"
                )
                return None

            grid_path = next(output_dir.rglob("*.TextGrid"), None)
            if grid_path is None:
                runtime_events.append("MFA alignment failed: TextGrid output was not generated.")
                return None
            intervals = self._parse_textgrid_word_intervals(grid_path.read_text(encoding="utf-8", errors="ignore"))
            if not intervals:
                runtime_events.append("MFA alignment failed: TextGrid has no parsable word intervals.")
                return None
            return intervals

    def _apply_mfa_alignment(
        self,
        *,
        source_path: str,
        occurrences: list[WordOccurrenceRecord],
        runtime_events: list[str],
    ) -> list[WordOccurrenceRecord]:
        transcript = " ".join(item.token for item in occurrences if normalize_word(item.token))
        intervals = self._run_mfa_alignment(source_path=source_path, transcript=transcript, runtime_events=runtime_events)
        if not intervals:
            return occurrences
        return self._align_occurrences_with_intervals(
            occurrences=occurrences,
            intervals=intervals,
            runtime_events=runtime_events,
            backend_label="MFA",
        )

    def _apply_selected_alignment(
        self,
        *,
        source_path: str,
        language: str,
        occurrences: list[WordOccurrenceRecord],
        segments: list[dict[str, object]],
        runtime_events: list[str],
    ) -> list[WordOccurrenceRecord]:
        if str(language or "").lower().startswith("zh"):
            runtime_events.append("Chinese pipeline uses direct ASR timestamps (forced alignment disabled).")
            return occurrences

        backend = str(getattr(self.settings, "asr_alignment_backend", "whisperx") or "whisperx").lower()
        if backend == "none":
            return occurrences
        if backend == "mfa":
            return self._apply_mfa_alignment(
                source_path=source_path,
                occurrences=occurrences,
                runtime_events=runtime_events,
            )
        if backend == "auto":
            mfa_aligned = self._apply_mfa_alignment(
                source_path=source_path,
                occurrences=occurrences,
                runtime_events=runtime_events,
            )
            if mfa_aligned is not occurrences:
                return mfa_aligned
            return self._apply_forced_alignment(
                source_path=source_path,
                language=language,
                occurrences=occurrences,
                segments=segments,
                runtime_events=runtime_events,
            )
        return self._apply_forced_alignment(
            source_path=source_path,
            language=language,
            occurrences=occurrences,
            segments=segments,
            runtime_events=runtime_events,
        )

    def _apply_forced_alignment(
        self,
        *,
        source_path: str,
        language: str,
        occurrences: list[WordOccurrenceRecord],
        segments: list[dict[str, object]],
        runtime_events: list[str],
    ) -> list[WordOccurrenceRecord]:
        if not occurrences or not segments:
            return occurrences
        if whisperx is None:
            runtime_events.append("Forced alignment skipped: whisperx is not installed.")
            return occurrences

        device, _compute = self.resolve_runtime()
        align_device = "cuda" if device == "cuda" else "cpu"

        try:
            align_model, align_metadata = self._get_or_create_align_model(language, align_device)
            audio = whisperx.load_audio(source_path)
            aligned_payload = whisperx.align(
                segments,
                align_model,
                align_metadata,
                audio,
                align_device,
                return_char_alignments=False,
            )
            aligned_words = self._extract_aligned_words(aligned_payload)
            if len(aligned_words) != len(occurrences):
                runtime_events.append(
                    f"Forced alignment returned {len(aligned_words)} words but ASR has {len(occurrences)} words; keep ASR timestamps."
                )
                return occurrences

            refined: list[WordOccurrenceRecord] = []
            for original, aligned in zip(occurrences, aligned_words):
                token_text, start_sec, end_sec, score = aligned
                refined.append(
                    WordOccurrenceRecord(
                        id=None,
                        source_audio_id=original.source_audio_id,
                        token=original.token or token_text,
                        normalized_token=original.normalized_token,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        confidence=max(0.0, score) if score > 0 else original.confidence,
                        segment_index=original.segment_index,
                        word_index=original.word_index,
                    )
                )
            runtime_events.append(f"Forced alignment applied for {source_path}.")
            return refined
        except Exception as exc:
            runtime_events.append(
                f"Forced alignment failed for {source_path}: {type(exc).__name__}: {exc}. Keep ASR timestamps."
            )
            return occurrences

    def transcribe(
        self,
        source_path: str,
        language: str = "en",
        model_tier: str = "large",
        model_name: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[WordOccurrenceRecord]:
        resolved_name = self.settings.resolve_model_name(model_tier, language, model_name=model_name)
        device, compute_type = self.resolve_runtime()
        runtime_events: list[str] = []

        occurrences: list[WordOccurrenceRecord] = []
        try:
            model = self._get_or_create_model(resolved_name, device, compute_type)
            occurrences, align_segments = self._transcribe_with_model(
                model,
                source_path,
                language,
                progress_callback=progress_callback,
            )
            occurrences = self._apply_selected_alignment(
                source_path=source_path,
                language=language,
                occurrences=occurrences,
                segments=align_segments,
                runtime_events=runtime_events,
            )
        except Exception as primary_exc:
            if device == "cuda":
                runtime_events.append(
                    f"CUDA failed for {source_path}: {type(primary_exc).__name__}: {primary_exc}. Retrying on CPU."
                )
                device = "cpu"
                compute_type = self.settings.asr_cpu_compute_type
                try:
                    model = self._get_or_create_model(resolved_name, device, compute_type)
                    occurrences, align_segments = self._transcribe_with_model(
                        model,
                        source_path,
                        language,
                        progress_callback=progress_callback,
                    )
                    occurrences = self._apply_selected_alignment(
                        source_path=source_path,
                        language=language,
                        occurrences=occurrences,
                        segments=align_segments,
                        runtime_events=runtime_events,
                    )
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
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[AudioSourceRecord, list[WordOccurrenceRecord], IngestResult]:
        occurrences = self.transcribe(
            source_path,
            language=language,
            model_tier=model_tier,
            model_name=model_name,
            progress_callback=progress_callback,
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


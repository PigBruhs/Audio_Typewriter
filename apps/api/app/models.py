from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AudioSourceRecord:
    source_audio_id: str
    base_name: str
    source_path: str
    language: str
    model_tier: str
    device: str
    compute_type: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class WordOccurrenceRecord:
    id: int | None
    source_audio_id: str
    token: str
    normalized_token: str
    start_sec: float
    end_sec: float
    confidence: float
    segment_index: int
    word_index: int


@dataclass(slots=True)
class PhraseOccurrenceRecord:
    source_audio_id: str
    phrase_text: str
    normalized_phrase: str
    start_sec: float
    end_sec: float
    confidence: float
    segment_index: int
    start_word_index: int
    end_word_index: int


@dataclass(slots=True)
class MixJobRecord:
    job_id: str
    sentence: str
    status: str
    output_path: str | None
    missing_tokens: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class MixPlanItem:
    token: str
    source_audio_id: str
    start_sec: float
    end_sec: float


@dataclass(slots=True)
class MixResult:
    job_id: str
    status: str
    output_path: str | None
    missing_tokens: list[str]
    base_name: str | None = None
    device_used: str | None = None
    token_count: int = 0
    output_files: list[str] | None = None


@dataclass(slots=True)
class IngestResult:
    source_audio_id: str
    base_name: str
    status: str
    token_count: int
    device_used: str
    compute_type: str


@dataclass(slots=True)
class AudioBaseRecord:
    base_name: str
    base_path: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class AudioBaseFileRecord:
    source_audio_id: str
    base_name: str
    sequence_number: int
    file_name: str
    file_path: str
    duration_sec: float
    file_size_bytes: int
    created_at: str


@dataclass(slots=True)
class AudioBaseStats:
    base_name: str
    audio_count: int
    total_duration_sec: float
    total_file_size_bytes: int



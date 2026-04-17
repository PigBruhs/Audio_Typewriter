from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class AudioSourceRecord:
    source_audio_id: str
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
class MixJobRecord:
    job_id: str
    sentence: str
    status: str
    output_path: str | None
    missing_tokens: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class MixResult:
    job_id: str
    status: str
    output_path: str | None
    missing_tokens: list[str]
    device_used: str | None = None
    token_count: int = 0


@dataclass(slots=True)
class IngestResult:
    source_audio_id: str
    status: str
    token_count: int
    device_used: str
    compute_type: str


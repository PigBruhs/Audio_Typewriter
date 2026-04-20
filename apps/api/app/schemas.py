from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    asr_preferred_device: str
    asr_resolved_device: str
    asr_compute_type: str
    asr_last_device_used: str
    asr_last_compute_type: str


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Audio file path")
    base_name: str | None = Field(default=None, description="Optional audio base name for scoped indexing")
    source_audio_id: str | None = Field(default=None, description="Optional internal source id")
    language: str = Field(default="en", description="Only English is supported in the first release")
    model_tier: str = Field(default="large", description="tiny, base, small, medium, large")
    model_name: str | None = Field(default=None, description="Explicit faster-whisper model name or HF repo ID")


class IngestResponse(BaseModel):
    source_audio_id: str
    base_name: str
    status: str
    token_count: int
    device_used: str
    compute_type: str


class MixRequest(BaseModel):
    base_name: str = Field(..., min_length=1)
    sentence: str = Field(..., min_length=1)
    speed_multiplier: float = Field(default=1.0, gt=0.0, description="Playback speed multiplier")
    gap_ms: int = Field(default=100, ge=0, description="Silence gap in milliseconds between words")
    mix_mode: Literal["word", "word_phrase", "word_phrase_sentence"] = Field(
        default="word_phrase_sentence",
        description="Planning mode: word, word+phrase, or word+phrase+sentence",
    )
    tail_extension_ms: int = Field(
        default=20,
        ge=0,
        le=500,
        description="Random tail extension range (0..N ms) per clip, capped by source end.",
    )
    output_path: str | None = None


class MixResponse(BaseModel):
    job_id: str
    base_name: str | None = None
    status: str
    output_path: str | None = None
    missing_tokens: list[str] = Field(default_factory=list)
    token_count: int = 0


class ClipSegmentRequest(BaseModel):
    source_audio_id: str = Field(..., min_length=1)
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., gt=0)
    label: str | None = None


class StitchRequest(BaseModel):
    base_name: str = Field(..., min_length=1)
    segments: list[ClipSegmentRequest] = Field(..., min_length=1)
    output_path: str | None = None


class ModelDownloadRequest(BaseModel):
    language: str = Field(default="en")
    model_tier: str = Field(default="large")
    model_name: str | None = Field(default=None, description="Optional explicit model name")


class ModelDownloadResponse(BaseModel):
    model_name: str
    status: str
    device_used: str
    compute_type: str
    cache_dir: str


class AudioBaseImportResponse(BaseModel):
    base_name: str
    overwritten: bool = False
    cleared_audio_files: int = 0
    cleared_index_sources: int = 0
    audio_count: int
    total_duration_sec: float
    total_file_size_bytes: int
    ingested_source_count: int
    token_count: int
    task_id: str | None = None
    task_status: str | None = None
    discarded_task_count: int = 0


class LocalAudioBaseImportRequest(BaseModel):
    base_name: str = Field(..., min_length=1, max_length=64)
    folder_path: str = Field(..., min_length=1, description="Local folder path containing .wav/.mp3 files")


class AudioBaseListItem(BaseModel):
    base_name: str
    audio_count: int
    total_duration_sec: float
    total_file_size_bytes: int


class AudioBaseStatsResponse(BaseModel):
    base_name: str
    audio_count: int
    total_duration_sec: float
    total_file_size_bytes: int



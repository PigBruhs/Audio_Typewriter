from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Audio file path")
    language: str = Field(default="en", description="Only English is supported in the first release")
    model_tier: str = Field(default="large", description="tiny, base, small, medium, large")
    model_name: str | None = Field(default=None, description="Explicit faster-whisper model name or HF repo ID")


class IngestResponse(BaseModel):
    source_audio_id: str
    status: str
    token_count: int
    device_used: str
    compute_type: str


class MixRequest(BaseModel):
    sentence: str = Field(..., min_length=1)
    output_path: str | None = None


class MixResponse(BaseModel):
    job_id: str
    status: str
    output_path: str | None = None
    missing_tokens: list[str] = Field(default_factory=list)
    token_count: int = 0


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


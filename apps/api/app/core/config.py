from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(
        self,
        *,
        app_name: str | None = None,
        app_env: str | None = None,
        data_dir: str | Path | None = None,
        database_path: str | Path | None = None,
        asr_device: str | None = None,
        asr_compute_type: str | None = None,
        asr_cpu_compute_type: str | None = None,
        asr_default_model: str | None = None,
        asr_model_cache_dir: str | Path | None = None,
        asr_beam_size: int | None = None,
        asr_preload_model: str | None = None,
        ffmpeg_binary: str | None = None,
        mix_output_dir: str | Path | None = None,
        max_candidates_per_token: int | None = None,
    ) -> None:
        self.app_name = app_name or os.getenv("AT_APP_NAME", "Audio Typewriter API")
        self.app_env = app_env or os.getenv("AT_APP_ENV", "dev")
        self.data_dir = Path(data_dir or os.getenv("AT_DATA_DIR", "data"))
        self.database_path = Path(
            database_path or os.getenv("AT_DB_PATH", str(self.data_dir / "audio_typewriter.sqlite3"))
        )
        self.asr_device = asr_device or os.getenv("AT_ASR_DEVICE", "cuda")
        self.asr_compute_type = asr_compute_type or os.getenv("AT_ASR_COMPUTE_TYPE", "float16")
        self.asr_cpu_compute_type = asr_cpu_compute_type or os.getenv("AT_ASR_CPU_COMPUTE_TYPE", "int8")
        self.asr_default_model = asr_default_model or os.getenv("AT_ASR_DEFAULT_MODEL", "large-v3")
        self.asr_model_cache_dir = Path(asr_model_cache_dir or os.getenv("AT_ASR_MODEL_CACHE_DIR", "./models"))
        self.asr_beam_size = int(asr_beam_size or os.getenv("AT_ASR_BEAM_SIZE", "5"))
        self.asr_preload_model = asr_preload_model or os.getenv("AT_ASR_PRELOAD_MODEL", "")
        self.ffmpeg_binary = ffmpeg_binary or os.getenv("AT_FFMPEG_BINARY", "ffmpeg")
        self.mix_output_dir = Path(mix_output_dir or os.getenv("AT_MIX_OUTPUT_DIR", "artifacts/mixes"))
        self.max_candidates_per_token = int(
            max_candidates_per_token or os.getenv("AT_MAX_CANDIDATES_PER_TOKEN", "8")
        )

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.database_path.as_posix()}"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.asr_model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.mix_output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_model_name(self, model_tier: str, language: str, model_name: str | None = None) -> str:
        language = (language or "en").lower()
        if language != "en":
            raise ValueError("Current scaffold only supports English ASR.")

        if model_name and model_name.strip():
            return model_name.strip()

        model_tier = (model_tier or self.asr_default_model).strip().lower()
        model_map = {
            "tiny": "tiny.en",
            "base": "base.en",
            "small": "small.en",
            "medium": "medium.en",
            "large": "large-v3",
            "xlarge": "large-v3",
            "xxlarge": "large-v3",
        }
        if model_tier in model_map:
            return model_map[model_tier]

        # Allow explicit faster-whisper model names or HuggingFace repo IDs.
        if "/" in model_tier or "." in model_tier or "-" in model_tier:
            return model_tier
        return self.asr_default_model


settings = Settings()

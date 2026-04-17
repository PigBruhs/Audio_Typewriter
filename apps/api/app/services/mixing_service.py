from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..core.config import Settings, settings
from ..db import SQLiteDatabase
from ..models import MixJobRecord, MixResult
from ..text import tokenize_sentence
from .index_service import IndexService
from audio_typewriter_core.models import MixPlanItem, WordOccurrence


@dataclass(slots=True)
class MixPlan:
    job_id: str
    tokens: list[str]
    items: list[MixPlanItem]
    missing_tokens: list[str]


class MixingService:
    def __init__(
        self,
        database: SQLiteDatabase | None = None,
        index_service: IndexService | None = None,
        runtime_settings: Settings | None = None,
    ) -> None:
        self.settings = runtime_settings or settings
        self.database = database or SQLiteDatabase()
        self.index_service = index_service or IndexService(self.database, self.settings)

    def build_mix_plan(self, sentence: str, job_id: str | None = None) -> MixPlan:
        tokens = tokenize_sentence(sentence)
        if not tokens:
            raise ValueError("Sentence is empty after tokenization.")

        token_search_results = self.index_service.search_tokens(tokens)
        items: list[MixPlanItem] = []
        missing_tokens: list[str] = []

        for search_result in token_search_results:
            if not search_result.candidates:
                missing_tokens.append(search_result.token)
                continue
            selected = search_result.candidates[0]
            items.append(
                MixPlanItem(
                    token=search_result.token,
                    source_audio_id=selected.source_audio_id,
                    start_sec=selected.start_sec,
                    end_sec=selected.end_sec,
                )
            )

        return MixPlan(
            job_id=job_id or str(uuid.uuid4()),
            tokens=tokens,
            items=items,
            missing_tokens=missing_tokens,
        )

    def _segment_filter(self, index: int, item: MixPlanItem) -> str:
        return (
            f"[{index}:a]atrim=start={item.start_sec:.3f}:end={item.end_sec:.3f},"
            f"asetpts=PTS-STARTPTS,aresample=44100,aformat=channel_layouts=mono[a{index}]"
        )

    def render_plan(self, plan: MixPlan, output_path: str | Path | None = None) -> str:
        if not plan.items:
            raise ValueError("No mixable tokens were found.")

        self.settings.ensure_directories()
        target_path = Path(output_path or self.settings.mix_output_dir / f"{plan.job_id}.wav")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        command = [self.settings.ffmpeg_binary, "-y"]
        for item in plan.items:
            command.extend(["-i", item.source_audio_id])

        filters = [self._segment_filter(index, item) for index, item in enumerate(plan.items)]
        if len(plan.items) == 1:
            filters.append("[a0]anull[outa]")
        else:
            concat_inputs = "".join(f"[a{index}]" for index in range(len(plan.items)))
            filters.append(f"{concat_inputs}concat=n={len(plan.items)}:v=0:a=1[outa]")

        command.extend([
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outa]",
            "-acodec",
            "pcm_s16le",
            str(target_path),
        ])

        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg render failed:\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return str(target_path)

    def mix_sentence(self, sentence: str, output_path: str | Path | None = None) -> MixResult:
        plan = self.build_mix_plan(sentence)
        now = datetime.now(timezone.utc).isoformat()
        missing_json = json.dumps(plan.missing_tokens, ensure_ascii=False)
        job_record = MixJobRecord(
            job_id=plan.job_id,
            sentence=sentence,
            status="queued",
            output_path=None,
            missing_tokens=missing_json,
            created_at=now,
            updated_at=now,
        )
        self.database.create_mix_job(job_record)

        if plan.missing_tokens:
            job_record.status = "failed"
            job_record.updated_at = datetime.now(timezone.utc).isoformat()
            self.database.create_mix_job(job_record)
            return MixResult(
                job_id=plan.job_id,
                status="failed",
                output_path=None,
                missing_tokens=plan.missing_tokens,
                token_count=len(plan.items),
            )

        rendered_path = self.render_plan(plan, output_path=output_path)
        job_record.status = "completed"
        job_record.output_path = rendered_path
        job_record.missing_tokens = missing_json
        job_record.updated_at = datetime.now(timezone.utc).isoformat()
        self.database.create_mix_job(job_record)
        return MixResult(
            job_id=plan.job_id,
            status="completed",
            output_path=rendered_path,
            missing_tokens=[],
            token_count=len(plan.items),
        )

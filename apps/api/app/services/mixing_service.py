from __future__ import annotations

import json
import random
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import Settings, settings
from ..db import SQLiteDatabase
from ..models import MixJobRecord, MixPlanItem, MixResult
from ..text import tokenize_sentence
from .index_service import IndexService


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

    def _select_candidate(
        self,
        candidates,
        previous_source_audio_id: str | None,
        previous_segment_index: int | None,
        mix_mode: str,
        rng: random.Random,
    ):
        if mix_mode == "all_random":
            return rng.choice(candidates)

        if mix_mode != "context_priority":
            raise ValueError("mix_mode must be one of: context_priority, all_random")

        if previous_source_audio_id is not None and previous_segment_index is not None:
            same_segment = [
                candidate
                for candidate in candidates
                if candidate.source_audio_id == previous_source_audio_id
                and candidate.segment_index >= 0
                and candidate.segment_index == previous_segment_index
            ]
            if same_segment:
                return rng.choice(same_segment)

            same_source = [candidate for candidate in candidates if candidate.source_audio_id == previous_source_audio_id]
            if same_source:
                return rng.choice(same_source)

        return rng.choice(candidates)

    def build_mix_plan(
        self,
        sentence: str,
        base_name: str,
        job_id: str | None = None,
        mix_mode: str = "context_priority",
    ) -> MixPlan:
        if mix_mode not in {"context_priority", "all_random"}:
            raise ValueError("mix_mode must be one of: context_priority, all_random")

        tokens = tokenize_sentence(sentence)
        if not tokens:
            raise ValueError("Sentence is empty after tokenization.")

        token_search_results = self.index_service.search_tokens(tokens, base_name=base_name)
        items: list[MixPlanItem] = []
        missing_tokens: list[str] = []
        rng = random.Random(job_id or sentence)
        previous_source_audio_id: str | None = None
        previous_segment_index: int | None = None

        for search_result in token_search_results:
            if not search_result.candidates:
                missing_tokens.append(search_result.token)
                continue
            selected = self._select_candidate(
                search_result.candidates,
                previous_source_audio_id,
                previous_segment_index,
                mix_mode,
                rng,
            )
            item = MixPlanItem(
                token=search_result.token,
                source_audio_id=selected.source_audio_id,
                start_sec=selected.start_sec,
                end_sec=selected.end_sec,
            )
            items.append(item)
            previous_source_audio_id = selected.source_audio_id
            previous_segment_index = selected.segment_index

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

    def _validate_manual_segment(self, item: MixPlanItem) -> None:
        if item.start_sec < 0:
            raise ValueError(f"Invalid segment for {item.source_audio_id}: start_sec must be >= 0")
        if item.end_sec <= item.start_sec:
            raise ValueError(f"Invalid segment for {item.source_audio_id}: end_sec must be > start_sec")

    def render_plan(
        self,
        plan: MixPlan,
        output_path: str | Path | None = None,
        base_name: str | None = None,
    ) -> str:
        if not plan.items:
            raise ValueError("No mixable tokens were found.")

        self.settings.ensure_directories()
        target_path = Path(output_path or self.settings.mix_output_dir / f"{plan.job_id}.wav")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        command = [self.settings.ffmpeg_binary, "-y"]
        for item in plan.items:
            if base_name:
                source_path = self.database.get_audio_source_path_for_base(item.source_audio_id, base_name)
            else:
                source_path = self.database.get_audio_source_path(item.source_audio_id)
            if not source_path:
                raise ValueError(f"Audio source not found for id: {item.source_audio_id}")
            command.extend(["-i", source_path])

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

    def stitch_segments(
        self,
        base_name: str,
        segments: list[MixPlanItem],
        output_path: str | Path | None = None,
    ) -> MixResult:
        if not segments:
            raise ValueError("At least one segment is required.")

        for segment in segments:
            self._validate_manual_segment(segment)

        job_id = str(uuid.uuid4())
        plan = MixPlan(
            job_id=job_id,
            tokens=[segment.token for segment in segments],
            items=segments,
            missing_tokens=[],
        )

        now = datetime.now(timezone.utc).isoformat()
        job_record = MixJobRecord(
            job_id=job_id,
            sentence="[manual-stitch]",
            status="queued",
            output_path=None,
            missing_tokens="[]",
            created_at=now,
            updated_at=now,
        )
        self.database.create_mix_job(job_record)

        rendered_path = self.render_plan(plan, output_path=output_path, base_name=base_name)
        job_record.status = "completed"
        job_record.output_path = rendered_path
        job_record.updated_at = datetime.now(timezone.utc).isoformat()
        self.database.create_mix_job(job_record)
        return MixResult(
            job_id=job_id,
            status="completed",
            output_path=rendered_path,
            missing_tokens=[],
            base_name=base_name,
            token_count=len(segments),
        )

    def mix_sentence(
        self,
        sentence: str,
        base_name: str,
        output_path: str | Path | None = None,
        mix_mode: str = "context_priority",
    ) -> MixResult:
        plan = self.build_mix_plan(sentence, base_name=base_name, mix_mode=mix_mode)
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
                base_name=base_name,
                token_count=len(plan.items),
            )

        rendered_path = self.render_plan(plan, output_path=output_path, base_name=base_name)
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
            base_name=base_name,
            token_count=len(plan.items),
        )

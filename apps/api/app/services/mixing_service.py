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
from ..text import normalize_word, tokenize_sentence
from .index_service import IndexService


@dataclass(slots=True)
class MixPlan:
    job_id: str
    tokens: list[str]
    items: list[MixPlanItem]
    missing_tokens: list[str]


class MixingService:
    _FLOAT_EPS = 1e-9
    _MAX_PHRASE_LEN = 4
    _SUPPORTED_MIX_MODES = {"word", "word_phrase", "word_phrase_sentence"}

    def __init__(
        self,
        database: SQLiteDatabase | None = None,
        index_service: IndexService | None = None,
        runtime_settings: Settings | None = None,
    ) -> None:
        self.settings = runtime_settings or settings
        self.database = database or SQLiteDatabase()
        self.index_service = index_service or IndexService(self.database, self.settings)

    def _pick_random(self, candidates, rng: random.Random):
        return rng.choice(candidates)

    def _build_sentence_seed_map(self, tokens: list[str], base_name: str) -> dict[int, object]:
        normalized_tokens = [normalize_word(token) for token in tokens]
        seed_segment = self.database.find_best_sentence_segment(normalized_tokens, base_name=base_name)
        if not seed_segment:
            return {}

        seed_source_audio_id, seed_segment_index = seed_segment
        segment_words = self.database.list_segment_words(seed_source_audio_id, seed_segment_index)
        if not segment_words:
            return {}

        mapped: dict[int, object] = {}
        pointer = 0
        for index, token in enumerate(normalized_tokens):
            if not token:
                continue
            while pointer < len(segment_words) and segment_words[pointer].normalized_token != token:
                pointer += 1
            if pointer >= len(segment_words):
                continue
            mapped[index] = segment_words[pointer]
            pointer += 1
        return mapped

    def _try_whole_sentence_direct_pass(
        self,
        *,
        tokens: list[str],
        base_name: str,
        rng: random.Random,
    ) -> MixPlanItem | None:
        # Try exact full-sentence phrase match first so rendering can use a single continuous trim.
        normalized_tokens = [normalize_word(token) for token in tokens]
        if any(not token for token in normalized_tokens):
            return None
        if len(normalized_tokens) < 2:
            return None

        full_phrase_candidates = self.database.search_phrase_tokens(
            normalized_tokens,
            base_name=base_name,
            limit=self.settings.max_candidates_per_token,
        )
        if not full_phrase_candidates:
            return None

        selected = self._pick_random(full_phrase_candidates, rng)
        return MixPlanItem(
            token=" ".join(tokens),
            source_audio_id=selected.source_audio_id,
            start_sec=selected.start_sec,
            end_sec=selected.end_sec,
        )

    def _select_candidate(
        self,
        candidates,
        previous_source_audio_id: str | None,
        previous_start_sec: float | None,
        rng: random.Random,
    ):
        # First token: random pick.
        if previous_source_audio_id is None or previous_start_sec is None:
            return self._pick_random(candidates, rng)

        # Later tokens: prefer same-source candidates closest to the previous token timestamp.
        same_source = [candidate for candidate in candidates if candidate.source_audio_id == previous_source_audio_id]
        if same_source:
            min_distance = min(abs(float(candidate.start_sec) - previous_start_sec) for candidate in same_source)
            closest = [
                candidate
                for candidate in same_source
                if abs(abs(float(candidate.start_sec) - previous_start_sec) - min_distance) <= self._FLOAT_EPS
            ]
            return rng.choice(closest)

        # No same-source hit: fallback to random from all candidates.
        return self._pick_random(candidates, rng)

    def build_mix_plan(
        self,
        sentence: str,
        base_name: str,
        job_id: str | None = None,
        mix_mode: str = "word_phrase_sentence",
    ) -> MixPlan:
        tokens = tokenize_sentence(sentence)
        if not tokens:
            raise ValueError("Sentence is empty after tokenization.")

        items: list[MixPlanItem] = []
        missing_tokens: list[str] = []
        rng = random.Random(job_id) if job_id else random.Random()
        previous_source_audio_id: str | None = None
        previous_start_sec: float | None = None

        normalized_mode = str(mix_mode or "word_phrase_sentence").strip().lower()
        if normalized_mode not in self._SUPPORTED_MIX_MODES:
            raise ValueError(f"Unsupported mix_mode: {mix_mode}")

        use_phrase = normalized_mode in {"word_phrase", "word_phrase_sentence"}
        use_sentence = normalized_mode == "word_phrase_sentence"

        if use_sentence:
            direct_sentence_item = self._try_whole_sentence_direct_pass(tokens=tokens, base_name=base_name, rng=rng)
            if direct_sentence_item is not None:
                return MixPlan(
                    job_id=job_id or str(uuid.uuid4()),
                    tokens=tokens,
                    items=[direct_sentence_item],
                    missing_tokens=[],
                )

        sentence_seed_map = self._build_sentence_seed_map(tokens, base_name) if use_sentence else {}

        index = 0
        while index < len(tokens):
            seed_word = sentence_seed_map.get(index)
            if seed_word is not None:
                # Prefer contiguous words from the best-matching sentence segment as an atomic chunk.
                end_index = index
                seed_end_word = seed_word
                while (end_index + 1) < len(tokens) and (end_index + 1) in sentence_seed_map:
                    next_seed = sentence_seed_map[end_index + 1]
                    if next_seed.source_audio_id != seed_word.source_audio_id:
                        break
                    if int(next_seed.segment_index) != int(seed_word.segment_index):
                        break
                    if int(next_seed.word_index) != int(seed_end_word.word_index) + 1:
                        break
                    if (end_index - index + 1) >= self._MAX_PHRASE_LEN:
                        break
                    end_index += 1
                    seed_end_word = next_seed

                token_text = " ".join(tokens[index : end_index + 1]) if end_index > index else tokens[index]
                items.append(
                    MixPlanItem(
                        token=token_text,
                        source_audio_id=seed_word.source_audio_id,
                        start_sec=seed_word.start_sec,
                        end_sec=seed_end_word.end_sec,
                    )
                )
                previous_source_audio_id = seed_word.source_audio_id
                previous_start_sec = seed_word.start_sec
                index = end_index + 1
                continue

            phrase_selected = False
            remaining = len(tokens) - index
            max_phrase_len = min(self._MAX_PHRASE_LEN, remaining)

            if use_phrase:
                # Try longer phrases first to reduce dependence on per-word timestamp precision.
                for phrase_len in range(max_phrase_len, 1, -1):
                    phrase_tokens = tokens[index : index + phrase_len]
                    normalized_phrase_tokens = [normalize_word(token) for token in phrase_tokens]
                    if any(not token for token in normalized_phrase_tokens):
                        continue

                    phrase_candidates = self.database.search_phrase_tokens(
                        normalized_phrase_tokens,
                        base_name=base_name,
                        limit=self.settings.max_candidates_per_token,
                    )
                    if not phrase_candidates:
                        continue

                    selected_phrase = self._select_candidate(
                        phrase_candidates,
                        previous_source_audio_id,
                        previous_start_sec,
                        rng,
                    )
                    phrase_text = " ".join(phrase_tokens)
                    items.append(
                        MixPlanItem(
                            token=phrase_text,
                            source_audio_id=selected_phrase.source_audio_id,
                            start_sec=selected_phrase.start_sec,
                            end_sec=selected_phrase.end_sec,
                        )
                    )
                    previous_source_audio_id = selected_phrase.source_audio_id
                    previous_start_sec = selected_phrase.start_sec
                    index += phrase_len
                    phrase_selected = True
                    break

            if phrase_selected:
                continue

            token = tokens[index]
            normalized = normalize_word(token)
            if not normalized:
                missing_tokens.append(token)
                index += 1
                continue

            candidates = self.database.search_token(normalized, limit=None, base_name=base_name)
            if not candidates:
                missing_tokens.append(token)
                index += 1
                continue

            selected = self._select_candidate(
                candidates,
                previous_source_audio_id,
                previous_start_sec,
                rng,
            )
            items.append(
                MixPlanItem(
                    token=token,
                    source_audio_id=selected.source_audio_id,
                    start_sec=selected.start_sec,
                    end_sec=selected.end_sec,
                )
            )
            previous_source_audio_id = selected.source_audio_id
            previous_start_sec = selected.start_sec
            index += 1

        return MixPlan(
            job_id=job_id or str(uuid.uuid4()),
            tokens=tokens,
            items=items,
            missing_tokens=missing_tokens,
        )

    def _segment_filter(
        self,
        index: int,
        item: MixPlanItem,
    ) -> str:
        end_sec = max(item.start_sec + 0.001, item.end_sec)
        return (
            f"[{index}:a]atrim=start={item.start_sec:.3f}:end={end_sec:.3f},"
            f"asetpts=PTS-STARTPTS,aresample=44100,aformat=channel_layouts=mono[a{index}]"
        )

    def _gap_filter(self, index: int, gap_sec: float) -> str:
        return f"anullsrc=r=44100:cl=mono:d={gap_sec:.3f}[g{index}]"

    def _atempo_chain(self, speed_multiplier: float) -> str:
        remaining = max(0.01, float(speed_multiplier))
        factors: list[float] = []
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        while remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        factors.append(remaining)
        return ",".join(f"atempo={factor:.6f}" for factor in factors)

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
        insert_word_gap: bool = False,
        word_gap_ms: int | None = None,
        speed_multiplier: float = 1.0,
        tail_extension_ms: int = 0,
    ) -> str:
        if not plan.items:
            raise ValueError("No mixable tokens were found.")

        self.settings.ensure_directories()
        target_path = Path(output_path or self.settings.mix_output_dir / f"{plan.job_id}.wav")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        command = [self.settings.ffmpeg_binary, "-y"]
        source_durations: list[float | None] = []
        for item in plan.items:
            if base_name:
                source_path = self.database.get_audio_source_path_for_base(item.source_audio_id, base_name)
                source_duration = self.database.get_audio_source_duration_for_base(item.source_audio_id, base_name)
            else:
                source_path = self.database.get_audio_source_path(item.source_audio_id)
                source_duration = None
            if not source_path:
                raise ValueError(f"Audio source not found for id: {item.source_audio_id}")
            command.extend(["-i", source_path])
            source_durations.append(source_duration)

        tail_ms = max(0, int(tail_extension_ms))
        tail_rng = random.Random(plan.job_id)
        effective_items: list[MixPlanItem] = []
        for idx, item in enumerate(plan.items):
            extended_end = float(item.end_sec)
            source_duration = source_durations[idx]
            if tail_ms > 0 and source_duration is not None and (source_duration - item.end_sec) > self._FLOAT_EPS:
                random_extra_sec = tail_rng.uniform(float(tail_ms) / 2000.0, float(tail_ms) / 1000.0)
                available = max(0.0, float(source_duration) - float(item.end_sec))
                extended_end = min(float(source_duration), float(item.end_sec) + min(random_extra_sec, available))
            effective_items.append(
                MixPlanItem(
                    token=item.token,
                    source_audio_id=item.source_audio_id,
                    start_sec=float(item.start_sec),
                    end_sec=max(float(item.end_sec), extended_end),
                )
            )

        filters: list[str] = []
        for index, item in enumerate(effective_items):
            filters.append(self._segment_filter(index, item))
        if len(effective_items) == 1:
            filters.append("[a0]anull[rawout]")
        else:
            if insert_word_gap:
                configured_gap_ms = word_gap_ms if word_gap_ms is not None else int(getattr(self.settings, "mix_word_gap_ms", 120))
                gap_sec = max(0.0, float(configured_gap_ms) / 1000.0)
                concat_chain: list[str] = ["[a0]"]
                for index in range(1, len(effective_items)):
                    filters.append(self._gap_filter(index - 1, gap_sec))
                    concat_chain.append(f"[g{index - 1}]")
                    concat_chain.append(f"[a{index}]")
                concat_inputs = "".join(concat_chain)
                filters.append(f"{concat_inputs}concat=n={len(concat_chain)}:v=0:a=1[rawout]")
            else:
                concat_inputs = "".join(f"[a{index}]" for index in range(len(effective_items)))
                filters.append(f"{concat_inputs}concat=n={len(effective_items)}:v=0:a=1[rawout]")

        normalized_speed = max(0.01, float(speed_multiplier))
        if abs(normalized_speed - 1.0) > 1e-6:
            filters.append(f"[rawout]{self._atempo_chain(normalized_speed)}[outa]")
        else:
            filters.append("[rawout]anull[outa]")

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
        speed_multiplier: float = 1.0,
        gap_ms: int | None = None,
        mix_mode: str = "word_phrase_sentence",
        tail_extension_ms: int = 20,
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
            missing_preview = ", ".join(plan.missing_tokens[:12])
            if len(plan.missing_tokens) > 12:
                missing_preview = f"{missing_preview}, ..."
            raise ValueError(
                f"Mix aborted: missing tokens in base '{base_name}': {missing_preview}"
            )

        rendered_path = self.render_plan(
            plan,
            output_path=output_path,
            base_name=base_name,
            insert_word_gap=True,
            word_gap_ms=gap_ms,
            speed_multiplier=speed_multiplier,
            tail_extension_ms=tail_extension_ms,
        )
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

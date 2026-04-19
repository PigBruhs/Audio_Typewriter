from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

from app.core.config import Settings
from app.db import SQLiteDatabase
from app.models import AudioBaseFileRecord, AudioBaseRecord, AudioSourceRecord, MixPlanItem, WordOccurrenceRecord
from app.services.asr_service import ASRService, ASRTranscriptionError
from app.services.audio_base_service import AudioBaseService
from app.services.index_service import IndexService
from app.services.mixing_service import MixingService
from app.services.task_queue_service import QueueTask, TaskQueueService


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)
        self.settings = Settings(
            data_dir=base / "data",
            database_path=base / "audio_typewriter.sqlite3",
            mix_output_dir=base / "mixes",
            audio_base_dir=base / "audio_base",
            temp_dir=base / "temp",
            asr_model_cache_dir=base / "models",
            asr_device="cuda",
        )
        self.database = SQLiteDatabase(self.settings.database_path)
        self.database.initialize()
        self.index_service = IndexService(self.database, self.settings)
        self.mixing_service = MixingService(self.database, self.index_service, self.settings)
        self.audio_base_service = AudioBaseService(self.settings)
        self.audio_base_service._probe_duration = lambda _path: 0.0  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_cuda_falls_back_to_cpu_when_unavailable(self) -> None:
        service = ASRService(self.settings, model_factory=None)
        service._cuda_available = lambda: False  # type: ignore[method-assign]
        device, compute_type = service.resolve_runtime()
        self.assertEqual(device, "cpu")
        self.assertEqual(compute_type, "int8")

    def test_resolve_large_and_custom_model(self) -> None:
        self.assertEqual(self.settings.resolve_model_name("large", "en"), "large-v3")
        self.assertEqual(self.settings.resolve_model_name("xlarge", "en"), "large-v3")
        self.assertEqual(
            self.settings.resolve_model_name("base", "en", model_name="Systran/faster-whisper-large-v3"),
            "Systran/faster-whisper-large-v3",
        )

    def test_resolve_local_prefixed_model_directory(self) -> None:
        local_model_dir = self.settings.asr_model_cache_dir / "faster-whisper-large-v3"
        local_model_dir.mkdir(parents=True, exist_ok=True)
        resolved = self.settings.resolve_model_name("large", "en")
        self.assertEqual(Path(resolved), local_model_dir)

    def test_model_download_uses_models_cache_dir(self) -> None:
        calls: list[dict[str, str]] = []

        def fake_factory(model_name: str, *, device: str, compute_type: str, download_root: str) -> object:
            calls.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "download_root": download_root,
                }
            )

            class DummyModel:
                pass

            return DummyModel()

        service = ASRService(self.settings, model_factory=fake_factory)
        service._cuda_available = lambda: False  # type: ignore[method-assign]
        result = service.download_model(model_tier="large", language="en")

        self.assertEqual(result.model_name, "large-v3")
        self.assertEqual(result.device_used, "cpu")
        self.assertEqual(Path(result.cache_dir), self.settings.asr_model_cache_dir)
        self.assertTrue(calls)
        self.assertEqual(Path(calls[0]["download_root"]), self.settings.asr_model_cache_dir)

    def test_transcribe_reuses_cached_model_for_same_runtime(self) -> None:
        calls: list[dict[str, str]] = []

        class DummyModel:
            def transcribe(self, *_args, **_kwargs):
                return [], None

        def fake_factory(model_name: str, *, device: str, compute_type: str, download_root: str) -> object:
            calls.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "download_root": download_root,
                }
            )
            return DummyModel()

        service = ASRService(self.settings, model_factory=fake_factory)
        service._cuda_available = lambda: False  # type: ignore[method-assign]

        service.transcribe("demo.wav", language="en", model_tier="large")
        service.transcribe("demo.wav", language="en", model_tier="large")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["device"], "cpu")

    def test_transcribe_routes_very_short_cuda_clip_to_cpu(self) -> None:
        settings = Settings(
            data_dir=self.settings.data_dir,
            database_path=self.settings.database_path,
            mix_output_dir=self.settings.mix_output_dir,
            audio_base_dir=self.settings.audio_base_dir,
            temp_dir=self.settings.temp_dir,
            asr_model_cache_dir=self.settings.asr_model_cache_dir,
            asr_device="cuda",
            asr_compute_type="float16",
            asr_cpu_compute_type="int8",
            asr_cuda_short_audio_cpu_threshold_sec=0.8,
        )
        calls: list[dict[str, str]] = []

        class DummyModel:
            def transcribe(self, *_args, **_kwargs):
                return [], None

        def fake_factory(model_name: str, *, device: str, compute_type: str, download_root: str) -> object:
            calls.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "download_root": download_root,
                }
            )
            return DummyModel()

        service = ASRService(settings, model_factory=fake_factory)
        service._cuda_available = lambda: True  # type: ignore[method-assign]
        service._probe_wav_duration_sec = lambda _path: 0.6  # type: ignore[method-assign]

        service.transcribe("audio_base/demo_base/000001.wav", language="en", model_tier="large")

        self.assertTrue(calls)
        self.assertEqual(calls[0]["device"], "cpu")
        self.assertEqual(service.last_device_used, "cpu")

    def test_transcribe_keeps_whisper_word_boundaries(self) -> None:
        class DummyWord:
            def __init__(self, text: str, start: float, end: float, prob: float) -> None:
                self.word = text
                self.start = start
                self.end = end
                self.probability = prob

        class DummySegment:
            def __init__(self, words, end: float) -> None:
                self.words = words
                self.end = end

        class DummyModel:
            def transcribe(self, *_args, **_kwargs):
                return [DummySegment([DummyWord("hello", 0.0, 0.2, 0.9), DummyWord("world", 0.3, 0.5, 0.9)], 0.5)], None

        def fake_factory(*_args, **_kwargs):
            return DummyModel()

        service = ASRService(self.settings, model_factory=fake_factory)
        service._cuda_available = lambda: False  # type: ignore[method-assign]
        occurrences = service.transcribe("demo.wav", language="en", model_tier="large")

        self.assertEqual(len(occurrences), 2)
        self.assertAlmostEqual(occurrences[0].end_sec, 0.2, places=3)
        self.assertAlmostEqual(occurrences[1].end_sec, 0.5, places=3)

    def test_vad_merges_segments_to_min_duration_and_appends_short_tail(self) -> None:
        segments = [
            (0.0, 6.0),
            (7.0, 12.0),
            (13.0, 19.0),
            (20.0, 28.0),
            (29.0, 34.0),
        ]
        merged = self.audio_base_service._merge_segments_by_min_duration(segments, 20.0)
        self.assertEqual(merged, [(0.0, 34.0)])

    def test_split_only_vad_uses_speech_boundaries_without_removing_gaps(self) -> None:
        source_path = Path(self.tempdir.name) / "source.wav"
        source_path.write_bytes(b"wav")
        target_dir = self.settings.audio_base_dir / "demo_base"
        target_dir.mkdir(parents=True, exist_ok=True)

        captured_ranges: list[tuple[float, float]] = []

        def fake_export(_source: Path, target: Path, start_sec: float, end_sec: float) -> None:
            captured_ranges.append((round(start_sec, 3), round(end_sec, 3)))
            target.write_bytes(b"clip")

        self.audio_base_service._normalize_audio_for_vad = lambda source, _work: source  # type: ignore[method-assign]
        self.audio_base_service._detect_speech_segments = lambda _path: [(2.0, 3.0), (6.0, 7.0)]  # type: ignore[method-assign]
        self.audio_base_service._probe_duration = lambda path: 10.0 if Path(path) == source_path else 1.0  # type: ignore[method-assign]
        self.audio_base_service._export_segment_clip = fake_export  # type: ignore[method-assign]

        records, next_sequence = self.audio_base_service._split_upload_into_speech_clips(
            source_path,
            base_name="demo_base",
            target_dir=target_dir,
            sequence_start=1,
            created_at="2026-01-01T00:00:00+00:00",
        )

        self.assertEqual(
            captured_ranges,
            [(0.0, 2.0), (2.0, 3.0), (3.0, 6.0), (6.0, 7.0), (7.0, 10.0)],
        )
        self.assertEqual(len(records), 5)
        self.assertEqual(next_sequence, 6)

    def test_task_pauses_when_asr_cuda_and_cpu_both_fail(self) -> None:
        queue_service = TaskQueueService(self.database, ASRService(self.settings), self.index_service, self.audio_base_service, self.settings)
        with queue_service._condition:
            queue_service._shutdown_requested = True
            queue_service._condition.notify_all()

        now = "2026-01-01T00:00:00+00:00"
        self.database.create_audio_base(
            AudioBaseRecord(
                base_name="demo_base",
                base_path=str(self.settings.audio_base_dir / "demo_base"),
                created_at=now,
                updated_at=now,
            )
        )
        self.database.replace_audio_base_files(
            "demo_base",
            [
                AudioBaseFileRecord(
                    source_audio_id="demo_base:000001",
                    base_name="demo_base",
                    sequence_number=1,
                    file_name="000001.wav",
                    file_path="audio_base/demo_base/000001.wav",
                    duration_sec=1.0,
                    file_size_bytes=1,
                    created_at=now,
                )
            ],
        )

        task = QueueTask(
            task_id="task-asr-fallback-fail",
            base_name="demo_base",
            status="running",
            total_files=1,
            processed_files=0,
            next_sequence_number=1,
            token_count=0,
            stage="asr",
            ready_for_asr=True,
            vad_total_audio_sec=1.0,
            vad_processed_audio_sec=1.0,
            vad_source_dir=None,
            vad_total_sources=0,
            vad_next_source_index=1,
            vad_next_segment_index=0,
            vad_next_sequence_number=1,
            vad_created_at=now,
            asr_last_completed_sequence=0,
            model_tier="large",
            created_at=now,
            updated_at=now,
        )

        def fake_ingest(*_args, **_kwargs):
            raise ASRTranscriptionError(
                "ASR failed on both CUDA and CPU",
                runtime_events=["CPU fallback failed for audio_base/demo_base/000001.wav: RuntimeError: bad clip"],
            )

        queue_service.asr_service.ingest = fake_ingest  # type: ignore[method-assign]

        with queue_service._condition:
            queue_service._tasks = [task]
            queue_service._save_tasks()

        queue_service._run_task(task)
        tasks = queue_service.list_tasks()
        self.assertEqual(tasks[0]["status"], "paused")
        self.assertEqual(tasks[0]["next_sequence_number"], 1)
        self.assertIn("ASR paused at seq=1", str(tasks[0]["last_error"]))
        self.assertIn("CPU fallback failed", str(tasks[0]["last_event"]))

    def test_index_search_and_mix_plan(self) -> None:
        audio_source = AudioSourceRecord(
            source_audio_id="clip-a.wav",
            base_name="demo_base",
            source_path="clip-a.wav",
            language="en",
            model_tier="base",
            device="cuda",
            compute_type="float16",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        occurrences = [
            WordOccurrenceRecord(None, "clip-a.wav", "Hello", "hello", 0.10, 0.40, 0.98, 0, 0),
            WordOccurrenceRecord(None, "clip-a.wav", "world", "world", 0.50, 0.80, 0.97, 0, 1),
        ]
        stored = self.index_service.ingest(audio_source, occurrences)
        self.assertEqual(stored, 2)

        matches = self.index_service.search_tokens(["hello", "world"])
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].candidates[0].normalized_token, "hello")
        self.assertEqual(matches[1].candidates[0].normalized_token, "world")

        plan = self.mixing_service.build_mix_plan("hello world", base_name="demo_base")
        self.assertEqual(plan.missing_tokens, [])
        self.assertEqual(len(plan.items), 2)

    def test_mix_sentence_reports_missing_tokens(self) -> None:
        audio_source = AudioSourceRecord(
            source_audio_id="clip-b.wav",
            base_name="demo_base",
            source_path="clip-b.wav",
            language="en",
            model_tier="base",
            device="cuda",
            compute_type="float16",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        occurrences = [WordOccurrenceRecord(None, "clip-b.wav", "hello", "hello", 0.0, 0.2, 0.9, 0, 0)]
        self.index_service.ingest(audio_source, occurrences)

        with self.assertRaises(ValueError) as exc:
            self.mixing_service.mix_sentence("hello missing", base_name="demo_base")
        self.assertIn("missing tokens", str(exc.exception))
        self.assertIn("missing", str(exc.exception))

    def test_mix_sentence_inserts_gap_between_words(self) -> None:
        source = AudioSourceRecord(
            source_audio_id="demo_base:000001",
            base_name="demo_base",
            source_path="audio_base/demo_base/000001.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self.index_service.ingest(
            source,
            [
                WordOccurrenceRecord(None, "demo_base:000001", "hello", "hello", 0.0, 0.2, 0.9, 0, 0),
                WordOccurrenceRecord(None, "demo_base:000001", "world", "world", 0.3, 0.5, 0.9, 0, 1),
            ],
        )

        with patch("app.services.mixing_service.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = ""
            run_mock.return_value.stderr = ""
            self.mixing_service.mix_sentence("hello world", base_name="demo_base")

            command = run_mock.call_args.args[0]
            filter_graph = command[command.index("-filter_complex") + 1]
            self.assertIn("atrim=start=0.000:end=0.200", filter_graph)
            self.assertIn("atrim=start=0.300:end=0.500", filter_graph)
            self.assertIn("anullsrc", filter_graph)
            self.assertIn("concat=n=3:v=0:a=1[rawout]", filter_graph)

    def test_mix_sentence_experimental_mode_clips_to_next_physical_word_start(self) -> None:
        source = AudioSourceRecord(
            source_audio_id="demo_base:000011",
            base_name="demo_base",
            source_path="audio_base/demo_base/000011.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self.index_service.ingest(
            source,
            [
                WordOccurrenceRecord(None, "demo_base:000011", "i", "i", 0.0, 0.1, 0.9, 0, 0),
                WordOccurrenceRecord(None, "demo_base:000011", "eat", "eat", 0.3, 0.6, 0.9, 0, 1),
            ],
        )

        with patch("app.services.mixing_service.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = ""
            run_mock.return_value.stderr = ""
            self.mixing_service.mix_sentence(
                "i",
                base_name="demo_base",
                clip_timing_mode="experimental_next_word_start",
            )

            command = run_mock.call_args.args[0]
            filter_graph = command[command.index("-filter_complex") + 1]
            self.assertIn("atrim=start=0.000:end=0.300", filter_graph)

    def test_mix_sentence_applies_speed_multiplier(self) -> None:
        source = AudioSourceRecord(
            source_audio_id="demo_base:000002",
            base_name="demo_base",
            source_path="audio_base/demo_base/000002.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self.index_service.ingest(
            source,
            [
                WordOccurrenceRecord(None, "demo_base:000002", "hello", "hello", 0.0, 0.2, 0.9, 0, 0),
                WordOccurrenceRecord(None, "demo_base:000002", "world", "world", 0.3, 0.5, 0.9, 0, 1),
            ],
        )

        with patch("app.services.mixing_service.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = ""
            run_mock.return_value.stderr = ""
            self.mixing_service.mix_sentence("hello world", base_name="demo_base", speed_multiplier=1.5)

            command = run_mock.call_args.args[0]
            filter_graph = command[command.index("-filter_complex") + 1]
            self.assertIn("atempo=1.500000", filter_graph)

    def test_context_priority_prefers_same_source_segment(self) -> None:
        source_a = AudioSourceRecord(
            source_audio_id="demo_base:000001",
            base_name="demo_base",
            source_path="audio_base/demo_base/000001.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        source_b = AudioSourceRecord(
            source_audio_id="demo_base:000002",
            base_name="demo_base",
            source_path="audio_base/demo_base/000002.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self.index_service.ingest(source_a, [WordOccurrenceRecord(None, "demo_base:000001", "hello", "hello", 0.0, 0.2, 0.95, 0, 0)])
        self.index_service.ingest(
            source_b,
            [
                WordOccurrenceRecord(None, "demo_base:000002", "world", "world", 0.2, 0.4, 0.99, 1, 0),
                WordOccurrenceRecord(None, "demo_base:000001", "world", "world", 0.3, 0.5, 0.20, 0, 1),
            ],
        )

        plan = self.mixing_service.build_mix_plan(
            "hello world",
            base_name="demo_base",
            mix_mode="context_priority",
        )
        self.assertEqual(len(plan.items), 2)
        self.assertEqual(plan.items[1].source_audio_id, "demo_base:000001")

    def test_context_priority_treats_adjacent_audio_as_same_sentence(self) -> None:
        source_center = AudioSourceRecord(
            source_audio_id="demo_base:000005",
            base_name="demo_base",
            source_path="audio_base/demo_base/000005.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        source_adjacent = AudioSourceRecord(
            source_audio_id="demo_base:000006",
            base_name="demo_base",
            source_path="audio_base/demo_base/000006.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        source_far = AudioSourceRecord(
            source_audio_id="demo_base:000010",
            base_name="demo_base",
            source_path="audio_base/demo_base/000010.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self.index_service.ingest(source_center, [WordOccurrenceRecord(None, "demo_base:000005", "hello", "hello", 0.1, 0.2, 0.99, 0, 0)])
        self.index_service.ingest(
            source_adjacent,
            [WordOccurrenceRecord(None, "demo_base:000006", "world", "world", 0.2, 0.3, 0.98, 0, 0)],
        )
        self.index_service.ingest(
            source_far,
            [WordOccurrenceRecord(None, "demo_base:000010", "world", "world", 0.2, 0.3, 0.97, 0, 0)],
        )

        plan = self.mixing_service.build_mix_plan("hello world", base_name="demo_base", mix_mode="context_priority")
        self.assertEqual(plan.items[1].source_audio_id, "demo_base:000006")

    def test_gap_modes_rank_by_audio_gap_then_time_components(self) -> None:
        source_005 = AudioSourceRecord(
            source_audio_id="demo_base:000005",
            base_name="demo_base",
            source_path="audio_base/demo_base/000005.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        source_004 = AudioSourceRecord(
            source_audio_id="demo_base:000004",
            base_name="demo_base",
            source_path="audio_base/demo_base/000004.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        source_001 = AudioSourceRecord(
            source_audio_id="demo_base:000001",
            base_name="demo_base",
            source_path="audio_base/demo_base/000001.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        source_007 = AudioSourceRecord(
            source_audio_id="demo_base:000007",
            base_name="demo_base",
            source_path="audio_base/demo_base/000007.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

        self.index_service.ingest(source_005, [WordOccurrenceRecord(None, "demo_base:000005", "hello", "hello", 365.0, 365.3, 0.99, 0, 0)])
        self.index_service.ingest(source_004, [WordOccurrenceRecord(None, "demo_base:000004", "world", "world", 360.0, 360.2, 0.90, 0, 0)])
        self.index_service.ingest(source_001, [WordOccurrenceRecord(None, "demo_base:000001", "world", "world", 241.0, 241.2, 0.90, 0, 1)])
        self.index_service.ingest(source_007, [WordOccurrenceRecord(None, "demo_base:000007", "world", "world", 10.0, 10.2, 0.90, 0, 2)])

        nearest = self.mixing_service.build_mix_plan("hello world", base_name="demo_base", mix_mode="nearest_gap")
        farthest = self.mixing_service.build_mix_plan("hello world", base_name="demo_base", mix_mode="farthest_gap")
        self.assertEqual(nearest.items[1].source_audio_id, "demo_base:000004")
        self.assertEqual(farthest.items[1].source_audio_id, "demo_base:000001")

    def test_invalid_mix_mode_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            self.mixing_service.build_mix_plan("hello", base_name="demo_base", mix_mode="unsupported")

    def test_audio_base_import_overwrites_existing_base(self) -> None:
        def fake_split(
            source_path: Path,
            *,
            base_name: str,
            target_dir: Path,
            sequence_start: int,
            created_at: str,
            checkpoint_callback=None,
        ) -> tuple[list[AudioBaseFileRecord], int]:
            target_name = f"{sequence_start:06d}.wav"
            target_path = target_dir / target_name
            target_path.write_bytes(source_path.read_bytes())
            return (
                [
                    AudioBaseFileRecord(
                        source_audio_id=f"{base_name}:{sequence_start:06d}",
                        base_name=base_name,
                        sequence_number=sequence_start,
                        file_name=target_name,
                        file_path=str(target_path),
                        duration_sec=0.0,
                        file_size_bytes=target_path.stat().st_size,
                        created_at=created_at,
                    )
                ],
                sequence_start + 1,
            )

        self.audio_base_service._split_upload_into_speech_clips = fake_split  # type: ignore[method-assign]
        first_file = UploadFile(filename="first.wav", file=BytesIO(b"aaa"))
        self.audio_base_service.import_audio_files("speaker_a", [first_file])

        removed = self.audio_base_service.clear_base_storage("speaker_a")
        self.assertEqual(removed, 1)

        second_file = UploadFile(filename="second.wav", file=BytesIO(b"bbbb"))
        self.audio_base_service.import_audio_files("speaker_a", [second_file])

        target_file = self.settings.audio_base_dir / "speaker_a" / "000001.wav"
        self.assertTrue(target_file.exists())
        self.assertEqual(target_file.read_bytes(), b"bbbb")

    def test_stitch_segments_preserves_input_order(self) -> None:
        source_a = AudioSourceRecord(
            source_audio_id="demo_base:000001",
            base_name="demo_base",
            source_path="audio_base/demo_base/000001.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        source_b = AudioSourceRecord(
            source_audio_id="demo_base:000002",
            base_name="demo_base",
            source_path="audio_base/demo_base/000002.wav",
            language="en",
            model_tier="base",
            device="cpu",
            compute_type="int8",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self.database.upsert_audio_source(source_a)
        self.database.upsert_audio_source(source_b)

        segments = [
            MixPlanItem(token="a", source_audio_id="demo_base:000002", start_sec=1.0, end_sec=1.2),
            MixPlanItem(token="b", source_audio_id="demo_base:000001", start_sec=0.5, end_sec=0.7),
        ]

        with patch("app.services.mixing_service.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = ""
            run_mock.return_value.stderr = ""
            self.mixing_service.stitch_segments(base_name="demo_base", segments=segments)

            command = run_mock.call_args.args[0]
            self.assertEqual(command[2:6], ["-i", "audio_base/demo_base/000002.wav", "-i", "audio_base/demo_base/000001.wav"])
            filter_graph = command[command.index("-filter_complex") + 1]
            self.assertIn("atrim=start=1.000:end=1.200", filter_graph)
            self.assertIn("atrim=start=0.500:end=0.700", filter_graph)

    def test_stitch_segments_rejects_invalid_timestamps(self) -> None:
        segments = [MixPlanItem(token="bad", source_audio_id="demo_base:000001", start_sec=0.8, end_sec=0.8)]
        with self.assertRaises(ValueError):
            self.mixing_service.stitch_segments(base_name="demo_base", segments=segments)

    def test_task_queue_load_preserves_asr_checkpoint_after_restart(self) -> None:
        queue_path = self.settings.data_dir / "asr_task_queue.json"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(
            """
            [
              {
                "task_id": "task-1",
                "base_name": "demo_base",
                "status": "running",
                "total_files": 10,
                "processed_files": 4,
                "next_sequence_number": 5,
                "token_count": 100,
                "stage": "asr",
                "ready_for_asr": true,
                "vad_total_audio_sec": 12.0,
                "vad_processed_audio_sec": 12.0,
                "vad_source_dir": null,
                "vad_total_sources": 0,
                "vad_next_source_index": 1,
                "vad_next_segment_index": 0,
                "vad_next_sequence_number": 1,
                "vad_created_at": "2026-01-01T00:00:00+00:00",
                "asr_last_completed_sequence": 4,
                "model_tier": "large",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "last_error": null,
                "overwritten": false,
                "cleared_audio_files": 0,
                "cleared_index_sources": 0
              }
            ]
            """.strip(),
            encoding="utf-8",
        )

        queue_service = TaskQueueService(self.database, ASRService(self.settings), self.index_service, self.audio_base_service, self.settings)
        tasks = queue_service.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["status"], "paused")
        self.assertEqual(tasks[0]["stage"], "asr")
        self.assertEqual(tasks[0]["next_sequence_number"], 4)

    def test_prepare_for_shutdown_pauses_running_tasks(self) -> None:
        queue_service = TaskQueueService(self.database, ASRService(self.settings), self.index_service, self.audio_base_service, self.settings)
        now = "2026-01-01T00:00:00+00:00"
        running_task = QueueTask(
            task_id="task-run",
            base_name="demo_base",
            status="running",
            total_files=5,
            processed_files=1,
            next_sequence_number=2,
            token_count=10,
            stage="asr",
            ready_for_asr=True,
            vad_total_audio_sec=5.0,
            vad_processed_audio_sec=5.0,
            vad_source_dir=None,
            vad_total_sources=0,
            vad_next_source_index=1,
            vad_next_segment_index=0,
            vad_next_sequence_number=1,
            vad_created_at=now,
            asr_last_completed_sequence=1,
            model_tier="large",
            created_at=now,
            updated_at=now,
        )
        with queue_service._condition:
            queue_service._tasks = [running_task]
            queue_service._save_tasks()

        result = queue_service.prepare_for_shutdown(wait_timeout_sec=0.2)
        tasks = queue_service.list_tasks()
        self.assertEqual(result["paused_running"], 1)
        self.assertEqual(tasks[0]["status"], "paused")

    def test_resume_asr_rewinds_checkpoint_and_aligns_progress(self) -> None:
        queue_service = TaskQueueService(self.database, ASRService(self.settings), self.index_service, self.audio_base_service, self.settings)
        with queue_service._condition:
            queue_service._shutdown_requested = True
            queue_service._condition.notify_all()
        now = "2026-01-01T00:00:00+00:00"
        task = QueueTask(
            task_id="task-asr-resume",
            base_name="demo_base",
            status="paused",
            total_files=100,
            processed_files=377,
            next_sequence_number=376,
            token_count=10,
            stage="asr",
            ready_for_asr=True,
            vad_total_audio_sec=12.0,
            vad_processed_audio_sec=12.0,
            vad_source_dir=None,
            vad_total_sources=0,
            vad_next_source_index=1,
            vad_next_segment_index=0,
            vad_next_sequence_number=1,
            vad_created_at=now,
            asr_last_completed_sequence=376,
            model_tier="large",
            created_at=now,
            updated_at=now,
        )
        with queue_service._condition:
            queue_service._tasks = [task]
            queue_service._save_tasks()

        resumed = queue_service.resume_task("task-asr-resume")
        self.assertEqual(resumed["status"], "queued")
        self.assertEqual(resumed["next_sequence_number"], 376)
        self.assertEqual(resumed["processed_files"], 375)

    def test_resume_asr_purges_checkpoint_and_later_index_rows(self) -> None:
        queue_service = TaskQueueService(self.database, ASRService(self.settings), self.index_service, self.audio_base_service, self.settings)
        with queue_service._condition:
            queue_service._shutdown_requested = True
            queue_service._condition.notify_all()

        now = "2026-01-01T00:00:00+00:00"
        self.database.create_audio_base(
            AudioBaseRecord(
                base_name="demo_base",
                base_path=str(self.settings.audio_base_dir / "demo_base"),
                created_at=now,
                updated_at=now,
            )
        )
        self.database.replace_audio_base_files(
            "demo_base",
            [
                AudioBaseFileRecord("demo_base:000001", "demo_base", 1, "000001.wav", "p1", 1.0, 1, now),
                AudioBaseFileRecord("demo_base:000002", "demo_base", 2, "000002.wav", "p2", 1.0, 1, now),
                AudioBaseFileRecord("demo_base:000003", "demo_base", 3, "000003.wav", "p3", 1.0, 1, now),
            ],
        )
        for seq in (1, 2, 3):
            source_id = f"demo_base:{seq:06d}"
            self.database.upsert_audio_source(
                AudioSourceRecord(
                    source_audio_id=source_id,
                    base_name="demo_base",
                    source_path=f"audio_base/demo_base/{seq:06d}.wav",
                    language="en",
                    model_tier="large",
                    device="cpu",
                    compute_type="int8",
                    created_at=now,
                    updated_at=now,
                )
            )
            self.database.replace_occurrences(
                source_id,
                [WordOccurrenceRecord(None, source_id, "w", "w", 0.0, 0.1, 0.9, 0, 0)],
            )

        task = QueueTask(
            task_id="task-asr-purge",
            base_name="demo_base",
            status="paused",
            total_files=3,
            processed_files=2,
            next_sequence_number=3,
            token_count=2,
            stage="asr",
            ready_for_asr=True,
            vad_total_audio_sec=3.0,
            vad_processed_audio_sec=3.0,
            vad_source_dir=None,
            vad_total_sources=0,
            vad_next_source_index=1,
            vad_next_segment_index=0,
            vad_next_sequence_number=1,
            vad_created_at=now,
            asr_last_completed_sequence=2,
            model_tier="large",
            created_at=now,
            updated_at=now,
        )
        with queue_service._condition:
            queue_service._tasks = [task]
            queue_service._save_tasks()

        resumed = queue_service.resume_task("task-asr-purge")
        self.assertEqual(resumed["next_sequence_number"], 2)
        self.assertEqual(resumed["processed_files"], 1)
        self.assertIsNotNone(self.database.get_audio_source_path("demo_base:000001"))
        self.assertIsNone(self.database.get_audio_source_path("demo_base:000002"))
        self.assertIsNone(self.database.get_audio_source_path("demo_base:000003"))

    def test_enqueue_reasr_task_after_full_purge(self) -> None:
        queue_service = TaskQueueService(self.database, ASRService(self.settings), self.index_service, self.audio_base_service, self.settings)
        with queue_service._condition:
            queue_service._shutdown_requested = True
            queue_service._condition.notify_all()

        now = "2026-01-01T00:00:00+00:00"
        self.database.create_audio_base(
            AudioBaseRecord(
                base_name="demo_base",
                base_path=str(self.settings.audio_base_dir / "demo_base"),
                created_at=now,
                updated_at=now,
            )
        )
        self.database.replace_audio_base_files(
            "demo_base",
            [
                AudioBaseFileRecord("demo_base:000001", "demo_base", 1, "000001.wav", "p1", 1.0, 1, now),
                AudioBaseFileRecord("demo_base:000002", "demo_base", 2, "000002.wav", "p2", 1.0, 1, now),
            ],
        )
        for seq in (1, 2):
            source_id = f"demo_base:{seq:06d}"
            self.database.upsert_audio_source(
                AudioSourceRecord(
                    source_audio_id=source_id,
                    base_name="demo_base",
                    source_path=f"audio_base/demo_base/{seq:06d}.wav",
                    language="en",
                    model_tier="large",
                    device="cpu",
                    compute_type="int8",
                    created_at=now,
                    updated_at=now,
                )
            )
            self.database.replace_occurrences(
                source_id,
                [WordOccurrenceRecord(None, source_id, "w", "w", 0.0, 0.1, 0.9, 0, 0)],
            )

        purged = self.database.purge_asr_index_from_sequence("demo_base", 1)
        self.assertEqual(purged["purged_sources"], 2)

        queued = queue_service.enqueue_reasr_task(base_name="demo_base", total_files=2)
        self.assertEqual(queued["stage"], "asr")
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(queued["next_sequence_number"], 1)
        self.assertEqual(queued["total_files"], 2)

    def test_pause_task_accumulates_elapsed_seconds(self) -> None:
        queue_service = TaskQueueService(self.database, ASRService(self.settings), self.index_service, self.audio_base_service, self.settings)
        with queue_service._condition:
            queue_service._shutdown_requested = True
            queue_service._condition.notify_all()

        queued = queue_service.enqueue_reasr_task(base_name="demo_base", total_files=1)
        task_id = str(queued["task_id"])
        with queue_service._condition:
            task = next(item for item in queue_service._tasks if item.task_id == task_id)
            task.status = "running"
            task.stage = "asr"
            task.asr_running_since = "2026-01-01T00:00:00+00:00"
            queue_service._save_tasks()

        with patch.object(queue_service, "_now_iso", return_value="2026-01-01T00:00:03+00:00"):
            paused = queue_service.pause_task(task_id)

        self.assertEqual(paused["status"], "paused")
        self.assertGreaterEqual(float(paused.get("asr_elapsed_sec", 0.0)), 3.0)

    def test_update_base_metadata_writes_json(self) -> None:
        metadata_path = self.audio_base_service.update_base_metadata(
            "demo_base",
            {
                "vad_total_elapsed_sec": 12.3,
                "top_words": [{"token": "hello", "count": 4}],
            },
        )
        self.assertTrue(metadata_path.exists())
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["base_name"], "demo_base")
        self.assertAlmostEqual(float(payload["vad_total_elapsed_sec"]), 12.3, places=3)
        self.assertEqual(payload["top_words"][0]["token"], "hello")

    def test_delete_task_purges_temp_db_and_audio_base(self) -> None:
        queue_service = TaskQueueService(self.database, ASRService(self.settings), self.index_service, self.audio_base_service, self.settings)
        now = "2026-01-01T00:00:00+00:00"

        base_dir = self.settings.audio_base_dir / "demo_base"
        base_dir.mkdir(parents=True, exist_ok=True)
        clip_path = base_dir / "000001.wav"
        clip_path.write_bytes(b"wav")

        self.database.create_audio_base(
            AudioBaseRecord(
                base_name="demo_base",
                base_path=str(base_dir),
                created_at=now,
                updated_at=now,
            )
        )
        self.database.upsert_audio_source(
            AudioSourceRecord(
                source_audio_id="demo_base:000001",
                base_name="demo_base",
                source_path=str(clip_path),
                language="en",
                model_tier="large",
                device="cpu",
                compute_type="int8",
                created_at=now,
                updated_at=now,
            )
        )

        task_id = "task-delete"
        vad_job_dir = self.settings.temp_dir / "vad_jobs" / task_id
        vad_job_dir.mkdir(parents=True, exist_ok=True)
        (vad_job_dir / "manifest.json").write_text("[]", encoding="utf-8")

        task = QueueTask(
            task_id=task_id,
            base_name="demo_base",
            status="paused",
            total_files=0,
            processed_files=0,
            next_sequence_number=1,
            token_count=0,
            stage="vad",
            ready_for_asr=True,
            vad_total_audio_sec=0.0,
            vad_processed_audio_sec=0.0,
            vad_source_dir=str(vad_job_dir / "sources"),
            vad_total_sources=0,
            vad_next_source_index=1,
            vad_next_segment_index=0,
            vad_next_sequence_number=1,
            vad_created_at=now,
            asr_last_completed_sequence=0,
            model_tier="large",
            created_at=now,
            updated_at=now,
        )
        with queue_service._condition:
            queue_service._tasks = [task]
            queue_service._save_tasks()

        queue_service.delete_task(task_id)
        self.assertEqual(queue_service.list_tasks(), [])
        self.assertFalse(base_dir.exists())
        self.assertFalse(vad_job_dir.exists())
        self.assertIsNone(self.database.get_audio_base_stats("demo_base"))
        self.assertIsNone(self.database.get_audio_source_path("demo_base:000001"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

from app.core.config import Settings
from app.db import SQLiteDatabase
from app.models import AudioBaseFileRecord, AudioSourceRecord, MixPlanItem, WordOccurrenceRecord
from app.services.asr_service import ASRService
from app.services.audio_base_service import AudioBaseService
from app.services.index_service import IndexService
from app.services.mixing_service import MixingService


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

        result = self.mixing_service.mix_sentence("hello missing", base_name="demo_base")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.missing_tokens, ["missing"])

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

    def test_stitch_segments_rejects_invalid_timestamps(self) -> None:
        segments = [MixPlanItem(token="bad", source_audio_id="demo_base:000001", start_sec=0.8, end_sec=0.8)]
        with self.assertRaises(ValueError):
            self.mixing_service.stitch_segments(base_name="demo_base", segments=segments)


if __name__ == "__main__":
    unittest.main()

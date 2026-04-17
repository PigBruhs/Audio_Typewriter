from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.core.config import Settings
from app.db import SQLiteDatabase
from app.models import AudioSourceRecord, WordOccurrenceRecord
from app.services.asr_service import ASRService
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
            asr_model_cache_dir=base / "models",
            asr_device="cuda",
        )
        self.database = SQLiteDatabase(self.settings.database_path)
        self.database.initialize()
        self.index_service = IndexService(self.database, self.settings)
        self.mixing_service = MixingService(self.database, self.index_service, self.settings)

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


if __name__ == "__main__":
    unittest.main()

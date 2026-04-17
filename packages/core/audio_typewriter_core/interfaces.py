from typing import Protocol

from audio_typewriter_core.models import MixPlanItem, WordOccurrence


class ASREngine(Protocol):
    def transcribe(self, source_path: str, language: str, model_tier: str) -> list[WordOccurrence]:
        ...


class IndexStore(Protocol):
    def upsert_occurrences(self, occurrences: list[WordOccurrence]) -> int:
        ...

    def search_tokens(self, tokens: list[str]) -> list[list[WordOccurrence]]:
        ...


class AudioMixer(Protocol):
    def render(self, plan: list[MixPlanItem], output_path: str) -> str:
        ...


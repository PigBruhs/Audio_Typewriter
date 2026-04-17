from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..core.config import Settings, settings
from ..db import SQLiteDatabase
from ..models import AudioSourceRecord, WordOccurrenceRecord
from ..text import normalize_word


@dataclass(slots=True)
class TokenSearchResult:
    token: str
    candidates: list[WordOccurrenceRecord]


class IndexService:
    def __init__(self, database: SQLiteDatabase | None = None, runtime_settings: Settings | None = None) -> None:
        self.database = database or SQLiteDatabase()
        self.settings = runtime_settings or settings

    def ingest(self, audio_source: AudioSourceRecord, occurrences: list[WordOccurrenceRecord]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        audio_source.updated_at = now
        self.database.upsert_audio_source(audio_source)
        return self.database.replace_occurrences(audio_source.source_audio_id, occurrences)

    def search_tokens(self, tokens: list[str]) -> list[TokenSearchResult]:
        results: list[TokenSearchResult] = []
        for token in tokens:
            normalized = normalize_word(token)
            if not normalized:
                results.append(TokenSearchResult(token=token, candidates=[]))
                continue
            candidates = self.database.search_token(normalized, limit=self.settings.max_candidates_per_token)
            results.append(TokenSearchResult(token=token, candidates=candidates))
        return results

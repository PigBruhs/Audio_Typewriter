from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .core.config import settings
from .models import AudioSourceRecord, MixJobRecord, WordOccurrenceRecord

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS audio_sources (
    source_audio_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    language TEXT NOT NULL,
    model_tier TEXT NOT NULL,
    device TEXT NOT NULL,
    compute_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS word_occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_audio_id TEXT NOT NULL,
    token TEXT NOT NULL,
    normalized_token TEXT NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    confidence REAL NOT NULL,
    segment_index INTEGER NOT NULL,
    word_index INTEGER NOT NULL,
    FOREIGN KEY (source_audio_id) REFERENCES audio_sources(source_audio_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_word_occurrences_normalized_token
    ON word_occurrences(normalized_token);

CREATE INDEX IF NOT EXISTS idx_word_occurrences_source_audio_id
    ON word_occurrences(source_audio_id);

CREATE TABLE IF NOT EXISTS mix_jobs (
    job_id TEXT PRIMARY KEY,
    sentence TEXT NOT NULL,
    status TEXT NOT NULL,
    output_path TEXT,
    missing_tokens TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class SQLiteDatabase:
    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path or settings.database_path)

    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def initialize(self) -> None:
        connection = self.connect()
        try:
            connection.executescript(SCHEMA_SQL)
            connection.commit()
        finally:
            connection.close()

    def upsert_audio_source(self, record: AudioSourceRecord) -> None:
        connection = self.connect()
        try:
            connection.execute(
                """
                INSERT INTO audio_sources (
                    source_audio_id, source_path, language, model_tier, device,
                    compute_type, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_audio_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    language = excluded.language,
                    model_tier = excluded.model_tier,
                    device = excluded.device,
                    compute_type = excluded.compute_type,
                    updated_at = excluded.updated_at
                """,
                (
                    record.source_audio_id,
                    record.source_path,
                    record.language,
                    record.model_tier,
                    record.device,
                    record.compute_type,
                    record.created_at,
                    record.updated_at,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def replace_occurrences(self, source_audio_id: str, occurrences: Iterable[WordOccurrenceRecord]) -> int:
        occurrence_rows = list(occurrences)
        connection = self.connect()
        try:
            connection.execute("DELETE FROM word_occurrences WHERE source_audio_id = ?", (source_audio_id,))
            connection.executemany(
                """
                INSERT INTO word_occurrences (
                    source_audio_id, token, normalized_token, start_sec, end_sec,
                    confidence, segment_index, word_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.source_audio_id,
                        row.token,
                        row.normalized_token,
                        row.start_sec,
                        row.end_sec,
                        row.confidence,
                        row.segment_index,
                        row.word_index,
                    )
                    for row in occurrence_rows
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return len(occurrence_rows)

    def search_token(self, normalized_token: str, limit: int = 8) -> list[WordOccurrenceRecord]:
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT id, source_audio_id, token, normalized_token, start_sec, end_sec,
                       confidence, segment_index, word_index
                FROM word_occurrences
                WHERE normalized_token = ?
                ORDER BY confidence DESC, start_sec ASC, id ASC
                LIMIT ?
                """,
                (normalized_token, limit),
            ).fetchall()
        finally:
            connection.close()
        return [
            WordOccurrenceRecord(
                id=row["id"],
                source_audio_id=row["source_audio_id"],
                token=row["token"],
                normalized_token=row["normalized_token"],
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                confidence=float(row["confidence"]),
                segment_index=int(row["segment_index"]),
                word_index=int(row["word_index"]),
            )
            for row in rows
        ]

    def create_mix_job(self, record: MixJobRecord) -> None:
        connection = self.connect()
        try:
            connection.execute(
                """
                INSERT INTO mix_jobs (
                    job_id, sentence, status, output_path, missing_tokens, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    sentence = excluded.sentence,
                    status = excluded.status,
                    output_path = excluded.output_path,
                    missing_tokens = excluded.missing_tokens,
                    updated_at = excluded.updated_at
                """,
                (
                    record.job_id,
                    record.sentence,
                    record.status,
                    record.output_path,
                    record.missing_tokens,
                    record.created_at,
                    record.updated_at,
                ),
            )
            connection.commit()
        finally:
            connection.close()

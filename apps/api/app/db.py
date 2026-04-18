from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .core.config import settings
from .models import AudioBaseFileRecord, AudioBaseRecord, AudioBaseStats, AudioSourceRecord, MixJobRecord, WordOccurrenceRecord

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS audio_sources (
    source_audio_id TEXT PRIMARY KEY,
    base_name TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS audio_bases (
    base_name TEXT PRIMARY KEY,
    base_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_base_files (
    source_audio_id TEXT PRIMARY KEY,
    base_name TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    duration_sec REAL NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (base_name) REFERENCES audio_bases(base_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audio_sources_base_name ON audio_sources(base_name);
CREATE INDEX IF NOT EXISTS idx_audio_base_files_base_name ON audio_base_files(base_name);
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
            # Lightweight migration for old local DBs.
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(audio_sources)").fetchall()
            }
            if "base_name" not in columns:
                connection.execute("ALTER TABLE audio_sources ADD COLUMN base_name TEXT NOT NULL DEFAULT ''")
            connection.commit()
        finally:
            connection.close()

    def upsert_audio_source(self, record: AudioSourceRecord) -> None:
        connection = self.connect()
        try:
            connection.execute(
                """
                INSERT INTO audio_sources (
                    source_audio_id, base_name, source_path, language, model_tier, device,
                    compute_type, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_audio_id) DO UPDATE SET
                    base_name = excluded.base_name,
                    source_path = excluded.source_path,
                    language = excluded.language,
                    model_tier = excluded.model_tier,
                    device = excluded.device,
                    compute_type = excluded.compute_type,
                    updated_at = excluded.updated_at
                """,
                (
                    record.source_audio_id,
                    record.base_name,
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

    def create_audio_base(self, record: AudioBaseRecord) -> None:
        connection = self.connect()
        try:
            connection.execute(
                """
                INSERT INTO audio_bases (base_name, base_path, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(base_name) DO UPDATE SET
                    base_path = excluded.base_path,
                    updated_at = excluded.updated_at
                """,
                (record.base_name, record.base_path, record.created_at, record.updated_at),
            )
            connection.commit()
        finally:
            connection.close()

    def replace_audio_base_files(self, base_name: str, records: Iterable[AudioBaseFileRecord]) -> int:
        rows = list(records)
        connection = self.connect()
        try:
            connection.execute("DELETE FROM audio_base_files WHERE base_name = ?", (base_name,))
            connection.executemany(
                """
                INSERT INTO audio_base_files (
                    source_audio_id, base_name, sequence_number, file_name,
                    file_path, duration_sec, file_size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.source_audio_id,
                        row.base_name,
                        row.sequence_number,
                        row.file_name,
                        row.file_path,
                        row.duration_sec,
                        row.file_size_bytes,
                        row.created_at,
                    )
                    for row in rows
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return len(rows)

    def list_audio_base_files(self, base_name: str, start_sequence_number: int = 1) -> list[AudioBaseFileRecord]:
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT source_audio_id, base_name, sequence_number, file_name, file_path,
                       duration_sec, file_size_bytes, created_at
                FROM audio_base_files
                WHERE base_name = ? AND sequence_number >= ?
                ORDER BY sequence_number ASC
                """,
                (base_name, start_sequence_number),
            ).fetchall()
        finally:
            connection.close()

        return [
            AudioBaseFileRecord(
                source_audio_id=row["source_audio_id"],
                base_name=row["base_name"],
                sequence_number=int(row["sequence_number"]),
                file_name=row["file_name"],
                file_path=row["file_path"],
                duration_sec=float(row["duration_sec"]),
                file_size_bytes=int(row["file_size_bytes"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_audio_sources_for_base(self, base_name: str) -> int:
        connection = self.connect()
        try:
            cursor = connection.execute("DELETE FROM audio_sources WHERE base_name = ?", (base_name,))
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()

    def clear_audio_base_for_overwrite(self, base_name: str) -> dict[str, int]:
        connection = self.connect()
        try:
            source_row = connection.execute(
                "SELECT COUNT(1) AS c FROM audio_sources WHERE base_name = ?",
                (base_name,),
            ).fetchone()
            file_row = connection.execute(
                "SELECT COUNT(1) AS c FROM audio_base_files WHERE base_name = ?",
                (base_name,),
            ).fetchone()

            connection.execute("DELETE FROM audio_sources WHERE base_name = ?", (base_name,))
            connection.execute("DELETE FROM audio_base_files WHERE base_name = ?", (base_name,))
            connection.commit()
            return {
                "cleared_index_sources": int(source_row["c"] if source_row else 0),
                "cleared_base_files": int(file_row["c"] if file_row else 0),
            }
        finally:
            connection.close()

    def delete_audio_base(self, base_name: str) -> int:
        connection = self.connect()
        try:
            cursor = connection.execute("DELETE FROM audio_bases WHERE base_name = ?", (base_name,))
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()

    def purge_asr_index_from_sequence(self, base_name: str, start_sequence_number: int) -> dict[str, int]:
        connection = self.connect()
        try:
            source_row = connection.execute(
                """
                SELECT COUNT(1) AS c
                FROM audio_sources
                WHERE source_audio_id IN (
                    SELECT source_audio_id
                    FROM audio_base_files
                    WHERE base_name = ? AND sequence_number >= ?
                )
                """,
                (base_name, start_sequence_number),
            ).fetchone()
            word_row = connection.execute(
                """
                SELECT COUNT(1) AS c
                FROM word_occurrences
                WHERE source_audio_id IN (
                    SELECT source_audio_id
                    FROM audio_base_files
                    WHERE base_name = ? AND sequence_number >= ?
                )
                """,
                (base_name, start_sequence_number),
            ).fetchone()

            connection.execute(
                """
                DELETE FROM audio_sources
                WHERE source_audio_id IN (
                    SELECT source_audio_id
                    FROM audio_base_files
                    WHERE base_name = ? AND sequence_number >= ?
                )
                """,
                (base_name, start_sequence_number),
            )
            connection.commit()
            return {
                "purged_sources": int(source_row["c"] if source_row else 0),
                "purged_occurrences": int(word_row["c"] if word_row else 0),
            }
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

    def list_audio_bases(self) -> list[AudioBaseRecord]:
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT base_name, base_path, created_at, updated_at
                FROM audio_bases
                ORDER BY created_at DESC
                """
            ).fetchall()
        finally:
            connection.close()
        return [
            AudioBaseRecord(
                base_name=row["base_name"],
                base_path=row["base_path"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_audio_base_stats(self, base_name: str) -> AudioBaseStats | None:
        connection = self.connect()
        try:
            row = connection.execute(
                """
                SELECT ab.base_name AS base_name,
                       COUNT(abf.source_audio_id) AS audio_count,
                       COALESCE(SUM(abf.duration_sec), 0.0) AS total_duration_sec,
                       COALESCE(SUM(abf.file_size_bytes), 0) AS total_file_size_bytes
                FROM audio_bases ab
                LEFT JOIN audio_base_files abf ON abf.base_name = ab.base_name
                WHERE ab.base_name = ?
                GROUP BY ab.base_name
                """,
                (base_name,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return AudioBaseStats(
            base_name=row["base_name"],
            audio_count=int(row["audio_count"]),
            total_duration_sec=float(row["total_duration_sec"]),
            total_file_size_bytes=int(row["total_file_size_bytes"]),
        )

    def search_token(self, normalized_token: str, limit: int = 8, base_name: str | None = None) -> list[WordOccurrenceRecord]:
        connection = self.connect()
        try:
            if base_name:
                rows = connection.execute(
                    """
                    SELECT wo.id, wo.source_audio_id, wo.token, wo.normalized_token, wo.start_sec, wo.end_sec,
                           wo.confidence, wo.segment_index, wo.word_index
                    FROM word_occurrences wo
                    JOIN audio_sources src ON src.source_audio_id = wo.source_audio_id
                    WHERE wo.normalized_token = ? AND src.base_name = ?
                    ORDER BY wo.confidence DESC, wo.start_sec ASC, wo.id ASC
                    LIMIT ?
                    """,
                    (normalized_token, base_name, limit),
                ).fetchall()
            else:
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

    def get_audio_source_path(self, source_audio_id: str) -> str | None:
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT source_path FROM audio_sources WHERE source_audio_id = ?",
                (source_audio_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return str(row["source_path"])

    def get_audio_source_path_for_base(self, source_audio_id: str, base_name: str) -> str | None:
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT source_path FROM audio_sources WHERE source_audio_id = ? AND base_name = ?",
                (source_audio_id, base_name),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return str(row["source_path"])


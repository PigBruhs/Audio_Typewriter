from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .core.config import settings
from .models import (
    AudioBaseFileRecord,
    AudioBaseRecord,
    AudioBaseStats,
    AudioSourceRecord,
    MixJobRecord,
    PhraseOccurrenceRecord,
    WordOccurrenceRecord,
)

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

    def search_token(self, normalized_token: str, limit: int | None = 8, base_name: str | None = None) -> list[WordOccurrenceRecord]:
        connection = self.connect()
        try:
            if base_name and limit is not None:
                rows = connection.execute(
                    """
                    SELECT wo.id, wo.source_audio_id, wo.token, wo.normalized_token, wo.start_sec, wo.end_sec,
                           wo.confidence, wo.segment_index, wo.word_index
                    FROM word_occurrences wo
                    JOIN audio_sources src ON src.source_audio_id = wo.source_audio_id
                    WHERE wo.normalized_token = ? AND src.base_name = ?
                    ORDER BY wo.start_sec ASC, wo.id ASC
                    LIMIT ?
                    """,
                    (normalized_token, base_name, limit),
                ).fetchall()
            elif base_name:
                rows = connection.execute(
                    """
                    SELECT wo.id, wo.source_audio_id, wo.token, wo.normalized_token, wo.start_sec, wo.end_sec,
                           wo.confidence, wo.segment_index, wo.word_index
                    FROM word_occurrences wo
                    JOIN audio_sources src ON src.source_audio_id = wo.source_audio_id
                    WHERE wo.normalized_token = ? AND src.base_name = ?
                    ORDER BY wo.start_sec ASC, wo.id ASC
                    """,
                    (normalized_token, base_name),
                ).fetchall()
            elif limit is not None:
                rows = connection.execute(
                    """
                    SELECT id, source_audio_id, token, normalized_token, start_sec, end_sec,
                           confidence, segment_index, word_index
                    FROM word_occurrences
                    WHERE normalized_token = ?
                    ORDER BY start_sec ASC, id ASC
                    LIMIT ?
                    """,
                    (normalized_token, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, source_audio_id, token, normalized_token, start_sec, end_sec,
                           confidence, segment_index, word_index
                    FROM word_occurrences
                    WHERE normalized_token = ?
                    ORDER BY start_sec ASC, id ASC
                    """,
                    (normalized_token,),
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

    def search_phrase_tokens(
        self,
        normalized_tokens: list[str],
        *,
        base_name: str | None = None,
        limit: int | None = 8,
    ) -> list[PhraseOccurrenceRecord]:
        tokens = [str(token).strip() for token in normalized_tokens if str(token).strip()]
        if len(tokens) < 2:
            return []

        aliases = [f"w{idx}" for idx in range(len(tokens))]
        joins: list[str] = []
        for idx in range(1, len(tokens)):
            joins.append(
                f"JOIN word_occurrences {aliases[idx]} ON "
                f"{aliases[idx]}.source_audio_id = w0.source_audio_id AND "
                f"{aliases[idx]}.segment_index = w0.segment_index AND "
                f"{aliases[idx]}.word_index = w0.word_index + {idx}"
            )

        confidence_expr = " + ".join(f"{alias}.confidence" for alias in aliases)
        confidence_expr = f"(({confidence_expr}) / {len(aliases)})"
        phrase_text_expr = " || ' ' || ".join(f"{alias}.token" for alias in aliases)
        where_clause = " AND ".join(f"{alias}.normalized_token = ?" for alias in aliases)

        sql = (
            "SELECT "
            "w0.source_audio_id AS source_audio_id, "
            f"{phrase_text_expr} AS phrase_text, "
            "w0.start_sec AS start_sec, "
            f"{aliases[-1]}.end_sec AS end_sec, "
            f"{confidence_expr} AS confidence, "
            "w0.segment_index AS segment_index, "
            "w0.word_index AS start_word_index, "
            f"{aliases[-1]}.word_index AS end_word_index "
            "FROM word_occurrences w0 "
            f"{' '.join(joins)} "
        )

        params: list[object] = [*tokens]
        if base_name:
            sql += "JOIN audio_sources src ON src.source_audio_id = w0.source_audio_id "
            where_clause += " AND src.base_name = ?"
            params.append(base_name)

        sql += f"WHERE {where_clause} ORDER BY start_sec ASC, start_word_index ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        connection = self.connect()
        try:
            rows = connection.execute(sql, params).fetchall()
        finally:
            connection.close()

        normalized_phrase = " ".join(tokens)
        return [
            PhraseOccurrenceRecord(
                source_audio_id=str(row["source_audio_id"]),
                phrase_text=str(row["phrase_text"]),
                normalized_phrase=normalized_phrase,
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                confidence=float(row["confidence"]),
                segment_index=int(row["segment_index"]),
                start_word_index=int(row["start_word_index"]),
                end_word_index=int(row["end_word_index"]),
            )
            for row in rows
        ]

    def find_best_sentence_segment(self, normalized_tokens: list[str], *, base_name: str | None = None) -> tuple[str, int] | None:
        unique_tokens = sorted({str(token).strip() for token in normalized_tokens if str(token).strip()})
        if not unique_tokens:
            return None

        placeholders = ", ".join("?" for _ in unique_tokens)
        sql = (
            "SELECT wo.source_audio_id AS source_audio_id, "
            "wo.segment_index AS segment_index, "
            "COUNT(1) AS token_hits, "
            "MIN(wo.start_sec) AS first_start "
            "FROM word_occurrences wo "
        )
        params: list[object] = [*unique_tokens]
        if base_name:
            sql += "JOIN audio_sources src ON src.source_audio_id = wo.source_audio_id "
        sql += f"WHERE wo.normalized_token IN ({placeholders}) "
        if base_name:
            sql += "AND src.base_name = ? "
            params.append(base_name)
        sql += (
            "GROUP BY wo.source_audio_id, wo.segment_index "
            "ORDER BY token_hits DESC, first_start ASC "
            "LIMIT 1"
        )

        connection = self.connect()
        try:
            row = connection.execute(sql, params).fetchone()
        finally:
            connection.close()

        if row is None:
            return None
        return str(row["source_audio_id"]), int(row["segment_index"])

    def list_segment_words(self, source_audio_id: str, segment_index: int) -> list[WordOccurrenceRecord]:
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT id, source_audio_id, token, normalized_token, start_sec, end_sec,
                       confidence, segment_index, word_index
                FROM word_occurrences
                WHERE source_audio_id = ? AND segment_index = ?
                ORDER BY word_index ASC, id ASC
                """,
                (source_audio_id, int(segment_index)),
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

    def get_audio_source_duration_for_base(self, source_audio_id: str, base_name: str) -> float | None:
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT duration_sec FROM audio_base_files WHERE source_audio_id = ? AND base_name = ?",
                (source_audio_id, base_name),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        try:
            return float(row["duration_sec"])
        except (TypeError, ValueError):
            return None


    def get_base_index_summary(self, base_name: str) -> dict[str, int]:
        connection = self.connect()
        try:
            source_row = connection.execute(
                "SELECT COUNT(1) AS c FROM audio_sources WHERE base_name = ?",
                (base_name,),
            ).fetchone()
            occurrence_row = connection.execute(
                """
                SELECT COUNT(1) AS c
                FROM word_occurrences wo
                JOIN audio_sources src ON src.source_audio_id = wo.source_audio_id
                WHERE src.base_name = ?
                """,
                (base_name,),
            ).fetchone()
            token_row = connection.execute(
                """
                SELECT COUNT(DISTINCT wo.normalized_token) AS c
                FROM word_occurrences wo
                JOIN audio_sources src ON src.source_audio_id = wo.source_audio_id
                WHERE src.base_name = ?
                """,
                (base_name,),
            ).fetchone()
        finally:
            connection.close()
        return {
            "indexed_sources": int(source_row["c"] if source_row else 0),
            "indexed_occurrences": int(occurrence_row["c"] if occurrence_row else 0),
            "distinct_tokens": int(token_row["c"] if token_row else 0),
        }

    def list_top_words_for_base(self, base_name: str, limit: int = 50) -> list[dict[str, int | str]]:
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT wo.normalized_token AS token, COUNT(1) AS count
                FROM word_occurrences wo
                JOIN audio_sources src ON src.source_audio_id = wo.source_audio_id
                WHERE src.base_name = ?
                GROUP BY wo.normalized_token
                ORDER BY count DESC, token ASC
                LIMIT ?
                """,
                (base_name, int(limit)),
            ).fetchall()
        finally:
            connection.close()
        return [{"token": str(row["token"]), "count": int(row["count"])} for row in rows]


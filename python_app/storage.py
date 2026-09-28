"""SQLite and local-file persistence for call records."""

from __future__ import annotations

import copy
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from python_app.review import RESOLUTIONS, review_item_id
from python_app.search import search_calls as _search_calls


class StorageError(RuntimeError):
    """A persistence-layer error."""


def _deep_merge(current: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(current)
    for key, value in changes.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class LocalCallStore:
    """Persist call JSON in SQLite and source files under a local directory."""

    def __init__(self, base_dir: str | Path = "data", now_provider=None) -> None:
        self.base_dir = Path(base_dir)
        self.sources_dir = self.base_dir / "sources"
        self.database_path = self.base_dir / "call_intelligence.sqlite3"
        self.now_provider = now_provider or (lambda: datetime.now().astimezone())
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    uploaded_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    call_date TEXT,
                    record_json TEXT NOT NULL,
                    source_path TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_calls_uploaded_at ON calls(uploaded_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_calls_processing_status ON calls(processing_status)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS call_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    line_number INTEGER,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (call_id) REFERENCES calls(call_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_call_chunks_call_id ON call_chunks(call_id)"
            )

    @staticmethod
    def _safe_filename(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")
        return cleaned or "source"

    def _source_path(self, storage_key: str) -> Path:
        candidate = (self.base_dir / storage_key).resolve()
        sources_root = self.sources_dir.resolve()
        if candidate != sources_root and sources_root not in candidate.parents:
            raise StorageError("Source path escapes the local source directory.")
        return candidate

    def save_call(self, record: dict[str, Any], source_bytes: bytes | None = None) -> dict[str, Any]:
        if not isinstance(record, dict) or not record.get("call_id"):
            raise StorageError("A call record with call_id is required.")

        stored = copy.deepcopy(record)
        call_id = str(stored["call_id"])
        updated_at = self.now_provider().isoformat()
        stored.setdefault("timestamps", {})["updated_at"] = updated_at

        source_path: str | None = stored.get("source", {}).get("storage_key")
        if source_bytes is not None:
            original_name = stored.get("source", {}).get("file_name", "source")
            relative_name = f"sources/{call_id}_{self._safe_filename(original_name)}"
            path = self._source_path(relative_name)
            path.write_bytes(source_bytes)
            source_path = relative_name
            stored.setdefault("source", {})["storage_key"] = source_path

        uploaded_at = stored.get("timestamps", {}).get("uploaded_at", updated_at)
        processing_status = stored.get("processing", {}).get("status", "pending")
        call_date = stored.get("metadata", {}).get("call_date")

        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO calls (
                        call_id, uploaded_at, updated_at, processing_status,
                        call_date, record_json, source_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(call_id) DO UPDATE SET
                        uploaded_at = excluded.uploaded_at,
                        updated_at = excluded.updated_at,
                        processing_status = excluded.processing_status,
                        call_date = excluded.call_date,
                        record_json = excluded.record_json,
                        source_path = excluded.source_path
                    """,
                    (
                        call_id,
                        uploaded_at,
                        updated_at,
                        processing_status,
                        call_date,
                        json.dumps(stored, ensure_ascii=False),
                        source_path,
                    ),
                )
        except Exception:
            if source_bytes is not None and source_path:
                path = self._source_path(source_path)
                if path.exists():
                    path.unlink()
            raise

        return stored

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT record_json FROM calls WHERE call_id = ?", (call_id,)
            ).fetchone()
        return json.loads(row["record_json"]) if row else None

    def list_calls(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT record_json FROM calls ORDER BY uploaded_at DESC"
            ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def update_call(self, call_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        current = self.get_call(call_id)
        if current is None:
            raise StorageError(f"Call {call_id} does not exist.")
        updated = _deep_merge(current, changes)
        return self.save_call(updated)

    def update_review_item(
        self,
        call_id: str,
        item_id: str,
        resolution: str,
        *,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Record a human decision on exactly one review item.

        Every other review item on the call is left exactly as it was. Once
        every item on the call has a decision, the call's review status is
        automatically switched off.
        """

        if resolution not in RESOLUTIONS:
            raise StorageError(
                f"Unsupported review resolution '{resolution}'. Use one of: {sorted(RESOLUTIONS)}."
            )

        record = self.get_call(call_id)
        if record is None:
            raise StorageError(f"Call {call_id} does not exist.")

        items = record.get("review", {}).get("items", [])
        resolved_at = self.now_provider().isoformat()
        updated_items: list[dict[str, Any]] = []
        found = False
        for item in items:
            if isinstance(item, dict):
                # A record saved before the ticket system existed may not have
                # a stored id yet. Fall back to computing one from the item's
                # own content -- the same fallback the UI uses -- and save it
                # permanently once matched, instead of requiring a migration.
                current_id = item.get("id") or review_item_id(item)
                if current_id == item_id:
                    found = True
                    item = {
                        **item,
                        "id": current_id,
                        "resolution": resolution,
                        "resolution_note": note,
                        "resolved_at": resolved_at,
                    }
            updated_items.append(item)

        if not found:
            raise StorageError(f"Review item {item_id} does not exist on call {call_id}.")

        still_open = any(
            isinstance(item, dict) and item.get("resolution") is None for item in updated_items
        )
        changes: dict[str, Any] = {
            "review": {
                "items": updated_items,
                "required": still_open,
                "status": "required" if still_open else "completed",
            }
        }
        if not still_open and record.get("processing", {}).get("stage") == "review":
            changes["processing"] = {"stage": "complete"}

        return self.update_call(call_id, changes)

    def save_chunks(self, call_id: str, chunks: list[dict[str, Any]]) -> None:
        """Replace every stored search chunk for one call with a fresh set.

        Called after extraction completes (including re-analysis), so a
        chunk set never goes stale -- the old rows for this call are removed
        before the new ones are written, in one transaction.
        """

        created_at = self.now_provider().isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM call_chunks WHERE call_id = ?", (call_id,))
            connection.executemany(
                """
                INSERT INTO call_chunks (call_id, chunk_type, line_number, text, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        call_id,
                        chunk["chunk_type"],
                        chunk.get("line_number"),
                        chunk["text"],
                        json.dumps(chunk["embedding"]),
                        created_at,
                    )
                    for chunk in chunks
                ],
            )

    def list_all_chunks(self) -> list[dict[str, Any]]:
        """Return every stored search chunk across every call, embeddings included."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT call_id, chunk_type, line_number, text, embedding FROM call_chunks"
            ).fetchall()
        return [
            {
                "call_id": row["call_id"],
                "chunk_type": row["chunk_type"],
                "line_number": row["line_number"],
                "text": row["text"],
                "embedding": json.loads(row["embedding"]),
            }
            for row in rows
        ]

    def search_calls(self, query: str, *, min_score: float = 0.0, limit: int = 10) -> list[dict[str, Any]]:
        """Rank stored calls by similarity to ``query``. See python_app.search."""

        return _search_calls(
            query, self.list_calls(), self.list_all_chunks(), min_score=min_score, limit=limit
        )

    def get_original_file(self, call_id: str) -> bytes | None:
        record = self.get_call(call_id)
        if not record:
            return None
        storage_key = record.get("source", {}).get("storage_key")
        if not storage_key:
            return None
        path = self._source_path(storage_key)
        return path.read_bytes() if path.exists() else None

    def delete_call(self, call_id: str) -> None:
        record = self.get_call(call_id)
        if record is None:
            return

        storage_key = record.get("source", {}).get("storage_key")
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM calls WHERE call_id = ?", (call_id,))
            connection.execute("DELETE FROM call_chunks WHERE call_id = ?", (call_id,))

        if storage_key:
            path = self._source_path(storage_key)
            if path.exists():
                path.unlink()

    def count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM calls").fetchone()
        return int(row["count"])

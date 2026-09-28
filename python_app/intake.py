"""Upload validation and canonical pending call-record creation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
    ".wma",
}
TRANSCRIPT_EXTENSIONS = {".json", ".md", ".txt"}


class IntakeError(ValueError):
    """A user-correctable upload error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class NormalizedUpload:
    name: str
    mime_type: str | None
    size_bytes: int
    input_type: str


def _upload_value(upload: Any, key: str, default: Any = None) -> Any:
    if isinstance(upload, dict):
        return upload.get(key, default)
    return getattr(upload, key, default)


def _extension(name: str) -> str:
    return Path(name).suffix.lower()


def detect_input_type(name: str, mime_type: str | None = None) -> str | None:
    """Return ``audio``, ``transcript``, or ``None`` for an unsupported upload."""

    content_type = (mime_type or "").lower()
    extension = _extension(name)

    if content_type.startswith("audio/") or extension in AUDIO_EXTENSIONS:
        return "audio"
    if (
        content_type == "application/json"
        or content_type.startswith("text/")
        or extension in TRANSCRIPT_EXTENSIONS
    ):
        return "transcript"
    return None


def normalize_upload(upload: Any) -> NormalizedUpload:
    name = str(_upload_value(upload, "name", "")).strip()
    if not name:
        raise IntakeError("MISSING_FILENAME", "The uploaded file must have a filename.")

    mime_type = str(_upload_value(upload, "type", "") or "").lower() or None
    raw_size = _upload_value(upload, "size", None)
    if raw_size is None and hasattr(upload, "getvalue"):
        raw_size = len(upload.getvalue())
    try:
        size_bytes = int(raw_size or 0)
    except (TypeError, ValueError) as error:
        raise IntakeError("INVALID_FILE_SIZE", f"File size is invalid for {name}.") from error
    if size_bytes < 0:
        raise IntakeError("INVALID_FILE_SIZE", f"File size is invalid for {name}.")

    input_type = detect_input_type(name, mime_type)
    if input_type is None:
        raise IntakeError(
            "UNSUPPORTED_FILE_TYPE",
            f"Unsupported file type for {name}. Upload audio or a .txt, .md, or .json transcript.",
        )

    return NormalizedUpload(name, mime_type, size_bytes, input_type)


def _call_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("call_id") or value.get("id") or "")
    return str(getattr(value, "call_id", "") or getattr(value, "id", ""))


def next_call_id(existing_calls: Iterable[Any] = ()) -> str:
    existing_ids = {_call_id(value) for value in existing_calls if _call_id(value)}
    highest = 0
    for value in existing_ids:
        if value.startswith("CALL-") and value[5:].isdigit():
            highest = max(highest, int(value[5:]))

    candidate_number = highest + 1
    candidate = f"CALL-{candidate_number:04d}"
    while candidate in existing_ids:
        candidate_number += 1
        candidate = f"CALL-{candidate_number:04d}"
    return candidate


def create_pending_call_record(
    upload: Any,
    call_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    normalized = normalize_upload(upload)
    if not call_id.startswith("CALL-") or not call_id[5:].isdigit():
        raise IntakeError("INVALID_CALL_ID", "A valid generated call ID is required.")

    current_time = now or datetime.now().astimezone()
    uploaded_at = current_time.isoformat()
    return {
        "schema_version": 1,
        "call_id": call_id,
        "source": {
            "file_name": normalized.name,
            "mime_type": normalized.mime_type,
            "size_bytes": normalized.size_bytes,
            "input_type": normalized.input_type,
            "storage_key": None,
        },
        "metadata": {
            "title": f"{normalized.name} · awaiting analysis",
            "call_date": current_time.date().isoformat(),
            "date_source": "default_current_date",
            "language": "en",
            "speakers": ["Speaker 1", "Speaker 2"],
            "duration_seconds": None,
        },
        "processing": {"status": "pending", "stage": "ingestion", "error": None},
        "transcript": {
            "source": "transcription_pending" if normalized.input_type == "audio" else "uploaded",
            "text": None,
            "lines": [],
        },
        "insights": {
            "summary": None,
            "tag": None,
            "decisions": [],
            "action_items": [],
            "blockers": [],
            "sentiment": {"overall": None, "anger": None, "anger_evidence": [], "profanity": []},
            "confidence": None,
        },
        "compliance_findings": [],
        "review": {"required": False, "status": "not_started", "items": []},
        "search": {"indexed": False, "terms": []},
        "timestamps": {"uploaded_at": uploaded_at, "updated_at": uploaded_at},
    }


def prepare_call_intake(
    uploads: Iterable[Any],
    existing_calls: Iterable[Any] = (),
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str | None]]]:
    """Create one pending record per valid upload and collect per-file errors."""

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str | None]] = []
    known_calls = list(existing_calls)

    for upload in uploads:
        try:
            call_id = next_call_id([*known_calls, *records])
            records.append(create_pending_call_record(upload, call_id, now=now))
        except IntakeError as error:
            errors.append(
                {
                    "file_name": _upload_value(upload, "name"),
                    "code": error.code,
                    "message": str(error),
                }
            )

    return records, errors


"""Transcript parsing and normalization for uploaded transcript files."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from python_app.review import with_review_defaults


class TranscriptError(ValueError):
    """A transcript that cannot be safely normalized."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


TIMESTAMP_RE = re.compile(
    r"^\s*\[?(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]?\s+(?P<rest>.+?)\s*$"
)
LINE_NUMBER_RE = re.compile(r"^\s*(?P<line_number>\d+)\s*:\s*(?P<rest>.+?)\s*$")
SPEAKER_RE = re.compile(r"^(?P<speaker>[^:]{1,80}):\s*(?P<text>.+)$")


def _timestamp_to_seconds(value: Any, milliseconds: bool = False) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value) / 1000 if milliseconds else float(value)

    raw = str(value).strip()
    if not raw:
        return None
    if raw.replace(".", "", 1).isdigit():
        return float(raw) / 1000 if milliseconds else float(raw)

    parts = raw.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as error:
        raise TranscriptError("INVALID_TIMESTAMP", f"Invalid timestamp: {value}") from error

    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return hours * 3600 + minutes * 60 + seconds
    raise TranscriptError("INVALID_TIMESTAMP", f"Invalid timestamp: {value}")


def _line(line_number: int, speaker: str, text: str, start: Any = None, end: Any = None, *, start_ms=False, end_ms=False) -> dict[str, Any]:
    return {
        "line": line_number,
        "speaker": speaker,
        "start_seconds": _timestamp_to_seconds(start, milliseconds=start_ms),
        "end_seconds": _timestamp_to_seconds(end, milliseconds=end_ms),
        "text": text.strip(),
    }


def _plain_text_lines(text: str) -> tuple[list[dict[str, Any]], bool]:
    lines: list[dict[str, Any]] = []
    speaker_inferred = False

    for raw_line in text.splitlines():
        content = raw_line.strip()
        if not content:
            continue

        start = None
        timestamp_match = TIMESTAMP_RE.match(content)
        if timestamp_match:
            start = timestamp_match.group("timestamp")
            content = timestamp_match.group("rest")

        # Some exports use: [timestamp] line_number: speaker: text -- there the
        # leading number is just a source line count, not a speaker. But other
        # transcripts use a bare number AS the speaker id, e.g. "1: Hello.".
        # Only drop the number when a real "Speaker: text" shape still follows
        # it; otherwise the number is the speaker, so keep it as one.
        line_number_match = LINE_NUMBER_RE.match(content)
        if line_number_match:
            rest = line_number_match.group("rest")
            if SPEAKER_RE.match(rest):
                content = rest
            else:
                content = f"Speaker {line_number_match.group('line_number')}: {rest}"

        speaker = None
        speaker_match = SPEAKER_RE.match(content)
        if speaker_match:
            speaker = speaker_match.group("speaker").strip()
            content = speaker_match.group("text").strip()

        if not speaker:
            speaker = "Speaker 1"
            speaker_inferred = True

        lines.append(_line(len(lines) + 1, speaker, content, start=start))

    return lines, speaker_inferred


def _infer_missing_end_times(lines: list[dict[str, Any]]) -> None:
    """Use the next line's start as the current line's end when safe."""

    for current, following in zip(lines, lines[1:]):
        if (
            current["end_seconds"] is None
            and current["start_seconds"] is not None
            and following["start_seconds"] is not None
            and following["start_seconds"] >= current["start_seconds"]
        ):
            current["end_seconds"] = following["start_seconds"]


def _json_items(payload: Any) -> tuple[list[Any], str | None]:
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        for key in ("lines", "utterances", "segments"):
            if isinstance(payload.get(key), list):
                return payload[key], None
        for key in ("transcript", "text", "content"):
            if isinstance(payload.get(key), str):
                return [], payload[key]
    raise TranscriptError(
        "UNSUPPORTED_TRANSCRIPT_JSON",
        "JSON transcript must contain a lines, utterances, segments, transcript, or text field.",
    )


def _json_lines(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    items, text_fallback = _json_items(payload)
    if text_fallback is not None:
        return _plain_text_lines(text_fallback)

    lines: list[dict[str, Any]] = []
    speaker_inferred = False
    for item in items:
        if isinstance(item, str):
            text = item
            speaker = "Speaker 1"
            speaker_inferred = True
            lines.append(_line(len(lines) + 1, speaker, text))
            continue
        if not isinstance(item, dict):
            raise TranscriptError("INVALID_TRANSCRIPT_LINE", "Each JSON transcript line must be an object or string.")

        text = item.get("text") or item.get("content") or item.get("value")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptError("MISSING_TRANSCRIPT_TEXT", "Each JSON transcript line needs non-empty text.")

        speaker = item.get("speaker") or item.get("speaker_label") or item.get("role")
        if not speaker:
            speaker = "Speaker 1"
            speaker_inferred = True

        start_key = next((key for key in ("start_seconds", "start_ms", "start") if key in item), None)
        end_key = next((key for key in ("end_seconds", "end_ms", "end") if key in item), None)
        lines.append(
            _line(
                len(lines) + 1,
                str(speaker),
                text,
                start=item.get(start_key) if start_key else None,
                end=item.get(end_key) if end_key else None,
                start_ms=start_key == "start_ms",
                end_ms=end_key == "end_ms",
            )
        )

    return lines, speaker_inferred


def normalize_transcript(file_name: str, content: bytes) -> dict[str, Any]:
    """Normalize a transcript into stable lines while preserving exact source text."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TranscriptError("INVALID_TEXT_ENCODING", "Transcript must be UTF-8 text.") from error

    if not text.strip():
        raise TranscriptError("EMPTY_TRANSCRIPT", "The transcript file is empty.")

    if Path(file_name).suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise TranscriptError("INVALID_TRANSCRIPT_JSON", "Transcript JSON is invalid.") from error
        lines, speaker_inferred = _json_lines(payload)
    else:
        lines, speaker_inferred = _plain_text_lines(text)

    _infer_missing_end_times(lines)

    warnings: list[str] = []
    if not lines:
        warnings.append("No transcript lines were found.")
    if speaker_inferred:
        warnings.append("One or more speaker labels were inferred as Speaker 1.")

    timestamps_available = bool(lines) and all(
        line["start_seconds"] is not None and line["end_seconds"] is not None for line in lines
    )
    if lines and not timestamps_available:
        if lines[-1]["start_seconds"] is not None and lines[-1]["end_seconds"] is None:
            warnings.append(
                "The final transcript line has no end timestamp because no following line start is available."
            )
        else:
            warnings.append("One or more transcript lines do not have complete timestamps.")

    return {
        "source": "uploaded",
        "text": text,
        "lines": lines,
        "normalization": {
            "speaker_labels_inferred": speaker_inferred,
            "timestamps_available": timestamps_available,
            "warnings": warnings,
        },
    }


def apply_normalized_transcript(record: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    """Return a call record updated with a normalized transcript and review warnings."""

    updated = copy.deepcopy(record)
    updated["transcript"] = transcript
    speakers = list(dict.fromkeys(line["speaker"] for line in transcript["lines"]))
    if speakers:
        updated.setdefault("metadata", {})["speakers"] = speakers

    warnings = transcript["normalization"]["warnings"]
    review = updated.setdefault("review", {"required": False, "status": "not_started", "items": []})
    if warnings:
        review["required"] = True
        review["status"] = "required"
        review["items"] = [
            with_review_defaults(
                {
                    "type": "Transcript normalization warning",
                    "detail": warning,
                    "review_status": "Needs review",
                }
            )
            for warning in warnings
        ]

    updated["processing"] = {
        "status": "transcript_ready",
        "stage": "normalizing",
        "error": None,
    }
    return updated

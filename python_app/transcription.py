"""Provider-agnostic audio transcription with a diarized-result contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in the project venv
    load_dotenv = None


class TranscriptionError(RuntimeError):
    """A transcription failure that can be shown as a processing error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TranscriptionProvider(Protocol):
    """Contract used by the application, independent of provider response format."""

    def transcribe(self, source_bytes: bytes, file_name: str, mime_type: str | None) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class TranscriptionSettings:
    provider: str
    api_key: str | None
    base_url: str
    timeout_seconds: int
    poll_interval_seconds: float


def _load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def get_transcription_settings() -> TranscriptionSettings:
    _load_environment()
    try:
        timeout_seconds = int(os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", "600"))
    except ValueError:
        timeout_seconds = 600
    try:
        poll_interval_seconds = float(os.getenv("TRANSCRIPTION_POLL_INTERVAL_SECONDS", "3"))
    except ValueError:
        poll_interval_seconds = 3.0

    return TranscriptionSettings(
        provider=os.getenv("TRANSCRIPTION_PROVIDER", "").strip().lower(),
        api_key=os.getenv("TRANSCRIPTION_API_KEY") or None,
        base_url=os.getenv("TRANSCRIPTION_BASE_URL", "https://api.assemblyai.com").rstrip("/"),
        timeout_seconds=max(timeout_seconds, 1),
        poll_interval_seconds=max(poll_interval_seconds, 0.1),
    )


def _request_json(response: requests.Response, operation: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise TranscriptionError("INVALID_PROVIDER_RESPONSE", f"Provider returned invalid JSON during {operation}.") from error

    if not response.ok:
        raise TranscriptionError(
            "PROVIDER_REQUEST_FAILED",
            f"Provider request failed during {operation} with HTTP {response.status_code}.",
        )
    if not isinstance(payload, dict):
        raise TranscriptionError("INVALID_PROVIDER_RESPONSE", f"Provider returned an invalid response during {operation}.")
    return payload


def transcript_from_provider_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Map a diarized provider payload into the canonical transcript contract."""

    raw_utterances = payload.get("utterances")
    utterances = raw_utterances if isinstance(raw_utterances, list) else []
    lines: list[dict[str, Any]] = []
    warnings: list[str] = []
    speaker_labels_inferred = False

    for item in utterances:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        raw_speaker = item.get("speaker")
        if raw_speaker is None or str(raw_speaker).strip() == "":
            speaker = "Speaker 1"
            speaker_labels_inferred = True
        else:
            speaker = f"Speaker {str(raw_speaker).strip()}"

        start_ms = item.get("start")
        end_ms = item.get("end")
        lines.append(
            {
                "line": len(lines) + 1,
                "speaker": speaker,
                "start_seconds": float(start_ms) / 1000 if isinstance(start_ms, (int, float)) else None,
                "end_seconds": float(end_ms) / 1000 if isinstance(end_ms, (int, float)) else None,
                "text": text.strip(),
            }
        )

    if not lines:
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            lines = [
                {
                    "line": 1,
                    "speaker": "Speaker 1",
                    "start_seconds": None,
                    "end_seconds": None,
                    "text": text.strip(),
                }
            ]
            speaker_labels_inferred = True
            warnings.append("The provider returned text without diarized utterances.")
        else:
            raise TranscriptionError("EMPTY_TRANSCRIPT", "The provider returned no usable transcript text.")

    if speaker_labels_inferred and "Speaker labels were inferred because diarization was incomplete." not in warnings:
        warnings.append("Speaker labels were inferred because diarization was incomplete.")
    if any(line["start_seconds"] is None or line["end_seconds"] is None for line in lines):
        warnings.append("One or more provider transcript lines do not have complete timestamps.")

    # A speaker who is diarized into only one line, while the rest of the call
    # alternates between a couple of others, is a common diarization mistake --
    # not a new speaker, just the same voice mis-clustered on one turn. This
    # doesn't fix the provider's diarization; it makes the uncertainty visible
    # instead of a silent, confusing owner label reaching the report.
    speaker_line_counts: dict[str, int] = {}
    for line in lines:
        speaker_line_counts[line["speaker"]] = speaker_line_counts.get(line["speaker"], 0) + 1
    singleton_speakers = [speaker for speaker, count in speaker_line_counts.items() if count == 1]
    if len(speaker_line_counts) > 2 and singleton_speakers:
        warnings.append(
            f"Possible diarization inconsistency: {', '.join(sorted(singleton_speakers))} "
            f"{'appears' if len(singleton_speakers) == 1 else 'appear'} in only one line out of "
            f"{len(lines)} -- this may be the same person as another speaker rather than a "
            "separate one. Check the transcript before trusting owner attribution."
        )

    return {
        "source": "transcribed",
        "text": str(payload.get("text") or "\n".join(line["text"] for line in lines)),
        "lines": lines,
        "normalization": {
            "speaker_labels_inferred": speaker_labels_inferred,
            "timestamps_available": all(
                line["start_seconds"] is not None and line["end_seconds"] is not None for line in lines
            ),
            "warnings": warnings,
        },
    }


class AssemblyAITranscriptionProvider:
    """Provider implementation kept behind the generic application contract."""

    def __init__(
        self,
        settings: TranscriptionSettings,
        *,
        session: requests.Session | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.sleep_fn = sleep_fn

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        if not self.settings.api_key:
            raise TranscriptionError("MISSING_TRANSCRIPTION_KEY", "TRANSCRIPTION_API_KEY is not configured.")
        headers = {"authorization": self.settings.api_key}
        if content_type:
            headers["content-type"] = content_type
        return headers

    def _post(self, path: str, *, headers: dict[str, str], **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.settings.base_url}{path}",
                headers=headers,
                timeout=30,
                **kwargs,
            )
        except requests.RequestException as error:
            raise TranscriptionError("PROVIDER_UNAVAILABLE", "The transcription provider could not be reached.") from error
        return _request_json(response, path)

    def _get(self, path: str, *, headers: dict[str, str]) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.settings.base_url}{path}",
                headers=headers,
                timeout=30,
            )
        except requests.RequestException as error:
            raise TranscriptionError("PROVIDER_UNAVAILABLE", "The transcription provider could not be reached.") from error
        return _request_json(response, path)

    def transcribe(self, source_bytes: bytes, file_name: str, mime_type: str | None) -> dict[str, Any]:
        if not source_bytes:
            raise TranscriptionError("EMPTY_AUDIO", "The uploaded audio file is empty.")

        upload = self._post(
            "/v2/upload",
            headers=self._headers(mime_type or "application/octet-stream"),
            data=source_bytes,
        )
        audio_url = upload.get("upload_url")
        if not isinstance(audio_url, str) or not audio_url:
            raise TranscriptionError("INVALID_PROVIDER_RESPONSE", "The provider did not return an audio upload URL.")

        request = self._post(
            "/v2/transcript",
            headers=self._headers("application/json"),
            json={
                "audio_url": audio_url,
                "language_code": "en",
                "speaker_labels": True,
                "punctuate": True,
                "format_text": True,
            },
        )
        transcript_id = request.get("id")
        if not isinstance(transcript_id, str) or not transcript_id:
            raise TranscriptionError("INVALID_PROVIDER_RESPONSE", "The provider did not return a transcript ID.")

        deadline = time.monotonic() + self.settings.timeout_seconds
        while time.monotonic() < deadline:
            result = self._get(f"/v2/transcript/{transcript_id}", headers=self._headers())
            status = str(result.get("status") or "").lower()
            if status == "completed":
                return transcript_from_provider_payload(result)
            if status in {"error", "failed"}:
                raise TranscriptionError("TRANSCRIPTION_FAILED", "The transcription provider failed to process the audio.")
            self.sleep_fn(self.settings.poll_interval_seconds)

        raise TranscriptionError("TRANSCRIPTION_TIMEOUT", "The transcription provider timed out before completing.")


def create_transcription_provider() -> TranscriptionProvider:
    """Create the configured provider without exposing provider details to callers."""

    settings = get_transcription_settings()
    if settings.provider in {"assemblyai", "assembly_ai"}:
        return AssemblyAITranscriptionProvider(settings)
    if not settings.provider:
        raise TranscriptionError("MISSING_TRANSCRIPTION_PROVIDER", "TRANSCRIPTION_PROVIDER is not configured.")
    raise TranscriptionError("UNSUPPORTED_TRANSCRIPTION_PROVIDER", "The configured transcription provider is unsupported.")

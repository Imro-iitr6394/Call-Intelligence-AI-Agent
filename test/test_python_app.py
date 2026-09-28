"""Tests for the Python intake and persistence implementation."""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import jsonschema

from python_app.intake import (
    create_pending_call_record,
    detect_input_type,
    next_call_id,
    prepare_call_intake,
)
from python_app.observability import _safe_attributes, observe, trace_model_call
from python_app.extraction import (
    ExtractionError,
    GeminiExtractionModel,
    GeminiSettings,
    apply_extraction_to_record,
    extract_call_intelligence,
    build_extraction_prompt,
    normalize_model_response,
    parse_model_response,
    validate_extraction_response,
)
from python_app.dates import looks_like_resolved_date, resolve_relative_date
from python_app.review import (
    review_item_id,
    visible_compliance_findings,
    with_review_defaults,
)
from python_app.risk_rules import detect_risks, overall_compliance_status, run_risk_rules
from python_app.storage import LocalCallStore, StorageError
from python_app.transcription import (
    AssemblyAITranscriptionProvider,
    TranscriptionSettings,
    transcript_from_provider_payload,
)
from python_app.transcripts import TranscriptError, apply_normalized_transcript, normalize_transcript


NOW = datetime(2026, 9, 25, 10, 30, tzinfo=timezone.utc)
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "call-record.schema.json"


def test_intake_detects_supported_inputs():
    assert detect_input_type("call.mp3", "audio/mpeg") == "audio"
    assert detect_input_type("notes.md", "text/markdown") == "transcript"
    assert detect_input_type("call.exe", "application/octet-stream") is None


def test_intake_creates_generic_pending_record():
    record = create_pending_call_record(
        {"name": "call.wav", "type": "audio/wav", "size": 42},
        "CALL-0007",
        now=NOW,
    )

    assert record["call_id"] == "CALL-0007"
    assert record["source"]["input_type"] == "audio"
    assert record["transcript"]["source"] == "transcription_pending"
    assert record["metadata"]["date_source"] == "default_current_date"


def test_batch_intake_returns_valid_records_and_errors():
    records, errors = prepare_call_intake(
        [
            {"name": "one.mp3", "type": "audio/mpeg", "size": 10},
            {"name": "two.txt", "type": "text/plain", "size": 20},
            {"name": "bad.exe", "type": "application/octet-stream", "size": 30},
        ],
        existing_calls=["CALL-0001"],
        now=NOW,
    )

    assert [record["call_id"] for record in records] == ["CALL-0002", "CALL-0003"]
    assert errors[0]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_next_call_id_avoids_existing_ids():
    assert next_call_id(["CALL-0001", "CALL-0004"]) == "CALL-0005"


def test_sqlite_store_supports_lifecycle():
    with TemporaryDirectory(dir=Path.cwd(), prefix=".test-call-storage-") as temp_dir:
        store = LocalCallStore(
            Path(temp_dir) / "data",
            now_provider=lambda: datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc),
        )
        record = create_pending_call_record(
            {"name": "call.txt", "type": "text/plain", "size": 12},
            "CALL-0001",
            now=NOW,
        )

        saved = store.save_call(record, source_bytes=b"hello transcript")
        assert saved["source"]["storage_key"] == "sources/CALL-0001_call.txt"
        assert store.get_call("CALL-0001")["call_id"] == "CALL-0001"
        assert store.get_original_file("CALL-0001") == b"hello transcript"

        updated = store.update_call("CALL-0001", {"processing": {"status": "processing"}})
        assert updated["processing"]["status"] == "processing"
        assert updated["processing"]["stage"] == "ingestion"

        store.delete_call("CALL-0001")
        assert store.get_call("CALL-0001") is None
        assert store.get_original_file("CALL-0001") is None


def test_update_review_item_resolves_one_item_without_touching_others():
    with TemporaryDirectory(dir=Path.cwd(), prefix=".test-call-storage-") as temp_dir:
        store = LocalCallStore(
            Path(temp_dir) / "data",
            now_provider=lambda: datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc),
        )
        record = create_pending_call_record(
            {"name": "call.txt", "type": "text/plain", "size": 12}, "CALL-0001", now=NOW
        )
        first_item = with_review_defaults({"type": "legal_escalation", "detail": "Mentioned an attorney."})
        second_item = with_review_defaults({"type": "contact_restriction", "detail": "Asked to stop calling."})
        record["review"] = {"required": True, "status": "required", "items": [first_item, second_item]}
        record["processing"] = {"status": "completed", "stage": "review", "error": None}
        store.save_call(record)

        updated = store.update_review_item(
            "CALL-0001", first_item["id"], "approved", note="Confirmed with supervisor."
        )

        resolved = next(item for item in updated["review"]["items"] if item["id"] == first_item["id"])
        untouched = next(item for item in updated["review"]["items"] if item["id"] == second_item["id"])
        assert resolved["resolution"] == "approved"
        assert resolved["resolution_note"] == "Confirmed with supervisor."
        assert resolved["resolved_at"] is not None
        assert untouched["resolution"] is None
        assert updated["review"]["required"] is True  # second item is still open
        assert updated["processing"]["stage"] == "review"

        fully_resolved = store.update_review_item("CALL-0001", second_item["id"], "rejected")

        assert fully_resolved["review"]["required"] is False
        assert fully_resolved["review"]["status"] == "completed"
        assert fully_resolved["processing"]["stage"] == "complete"


def test_update_review_item_rejects_unknown_resolution_and_unknown_item():
    with TemporaryDirectory(dir=Path.cwd(), prefix=".test-call-storage-") as temp_dir:
        store = LocalCallStore(
            Path(temp_dir) / "data",
            now_provider=lambda: datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc),
        )
        record = create_pending_call_record(
            {"name": "call.txt", "type": "text/plain", "size": 12}, "CALL-0001", now=NOW
        )
        item = with_review_defaults({"type": "legal_escalation", "detail": "Mentioned an attorney."})
        record["review"] = {"required": True, "status": "required", "items": [item]}
        store.save_call(record)

        try:
            store.update_review_item("CALL-0001", item["id"], "not-a-real-resolution")
        except StorageError:
            pass
        else:
            raise AssertionError("Expected an unsupported resolution to be rejected")

        try:
            store.update_review_item("CALL-0001", "missing-item-id", "approved")
        except StorageError:
            pass
        else:
            raise AssertionError("Expected an unknown review item id to be rejected")


def test_apply_extraction_to_record_keeps_a_reviewers_earlier_decision():
    transcript = _risk_transcript()
    record = create_pending_call_record(
        {"name": "call.txt", "type": "text/plain", "size": 12}, "CALL-0001", now=NOW
    )
    record["transcript"] = transcript

    first_result = extract_call_intelligence(transcript)
    record = apply_extraction_to_record(record, first_result)
    finding_item = next(
        item for item in record["review"]["items"] if item["type"] == "contact_restriction"
    )
    record["review"]["items"] = [
        {**item, "resolution": "approved", "resolution_note": "Reviewed once already."}
        if item["id"] == finding_item["id"]
        else item
        for item in record["review"]["items"]
    ]

    second_result = extract_call_intelligence(transcript)
    record = apply_extraction_to_record(record, second_result)

    carried_forward = next(
        item for item in record["review"]["items"] if item["id"] == finding_item["id"]
    )
    assert carried_forward["resolution"] == "approved"
    assert carried_forward["resolution_note"] == "Reviewed once already."
    assert record["review"]["required"] is False
    assert record["processing"]["stage"] == "complete"


def test_normalizes_plain_text_speaker_and_timestamp_lines():
    transcript = normalize_transcript(
        "call.txt",
        b"00:00:04 Speaker 1: I can pay today.\nAgent: I will follow up.\n",
    )

    assert transcript["lines"][0]["start_seconds"] == 4.0
    assert transcript["lines"][0]["speaker"] == "Speaker 1"
    assert transcript["lines"][0]["end_seconds"] is None
    assert transcript["lines"][1]["speaker"] == "Agent"
    assert transcript["normalization"]["timestamps_available"] is False
    assert transcript["normalization"]["warnings"]


def test_normalizes_numbered_lines_without_treating_line_numbers_as_speakers():
    transcript = normalize_transcript(
        "call.txt",
        (
            b"[00:00:01] 1: Agent: Hello.\n"
            b"[00:00:06] 2: Consumer: I need help.\n"
            b"[00:00:10] 3: Agent: I will follow up.\n"
        ),
    )

    lines = transcript["lines"]
    assert [line["speaker"] for line in lines] == ["Agent", "Consumer", "Agent"]
    assert [line["start_seconds"] for line in lines] == [1.0, 6.0, 10.0]
    assert [line["end_seconds"] for line in lines] == [6.0, 10.0, None]
    assert transcript["normalization"]["speaker_labels_inferred"] is False


def test_normalizes_bare_numbered_speakers_without_collapsing_them():
    transcript = normalize_transcript(
        "call.txt",
        b"1: I need to pay.\n2: Sure, go ahead.\n1: Thank you.\n",
    )

    lines = transcript["lines"]
    assert [line["speaker"] for line in lines] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert [line["text"] for line in lines] == ["I need to pay.", "Sure, go ahead.", "Thank you."]


def test_normalizes_json_lines_and_marks_inferred_speaker():
    transcript = normalize_transcript(
        "call.json",
        b'{"lines": [{"speaker": "Customer", "start_ms": 1500, "end_ms": 2500, "text": "I need help."}]}',
    )

    assert transcript["lines"][0]["start_seconds"] == 1.5
    assert transcript["lines"][0]["end_seconds"] == 2.5
    assert transcript["normalization"]["timestamps_available"] is True


def test_apply_normalized_transcript_routes_warnings_to_review():
    record = create_pending_call_record({"name": "call.txt", "type": "text/plain", "size": 5}, "CALL-0001", now=NOW)
    transcript = normalize_transcript("call.txt", b"A line without a speaker.")
    updated = apply_normalized_transcript(record, transcript)

    assert updated["processing"]["status"] == "transcript_ready"
    assert updated["review"]["required"] is True
    assert updated["review"]["status"] == "required"


def test_rejects_invalid_transcript_json():
    try:
        normalize_transcript("call.json", b"not json")
    except TranscriptError as error:
        assert error.code == "INVALID_TRANSCRIPT_JSON"
    else:
        raise AssertionError("Expected invalid transcript JSON to be rejected")


def test_observability_attributes_exclude_sensitive_content():
    safe = _safe_attributes(
        {
            "call_id": "CALL-0001",
            "input_type": "transcript",
            "transcript_text": "customer account details",
            "prompt": "private prompt",
            "line_count": 3,
        }
    )

    assert safe == {"call_id": "CALL-0001", "input_type": "transcript", "line_count": 3}


def test_observability_is_non_blocking_when_disabled(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")

    with observe("test_operation", transcript_text="must not be exported"):
        result = trace_model_call("test_model_call", lambda: {"private": "output"})

    assert result == {"private": "output"}


def test_transcription_payload_maps_diarized_utterances_to_generic_lines():
    transcript = transcript_from_provider_payload(
        {
            "text": "Hello. I need help.",
            "utterances": [
                {"speaker": "A", "start": 0, "end": 1200, "text": "Hello."},
                {"speaker": "B", "start": 1200, "end": 2800, "text": "I need help."},
            ],
        }
    )

    assert transcript["source"] == "transcribed"
    assert [line["speaker"] for line in transcript["lines"]] == ["Speaker A", "Speaker B"]
    assert transcript["lines"][0]["end_seconds"] == 1.2
    assert transcript["lines"][1]["start_seconds"] == 1.2
    assert transcript["normalization"]["timestamps_available"] is True


def test_transcription_flags_a_speaker_who_appears_only_once_as_possible_misdiarization():
    """Regression test for a real bug found during independent QA: AssemblyAI
    diarized two real speakers into three labels, splitting one speaker's
    closing line off as a new "Speaker C" with no warning anywhere.
    """

    transcript = transcript_from_provider_payload(
        {
            "text": "Hello. I need help. Sure, one moment. Here's the update.",
            "utterances": [
                {"speaker": "A", "start": 0, "end": 1000, "text": "Hello."},
                {"speaker": "B", "start": 1000, "end": 2000, "text": "I need help."},
                {"speaker": "A", "start": 2000, "end": 3000, "text": "Sure, one moment."},
                {"speaker": "C", "start": 3000, "end": 4000, "text": "Here's the update."},
            ],
        }
    )

    warnings = transcript["normalization"]["warnings"]
    assert any("diarization inconsistency" in w for w in warnings)
    assert any("Speaker C" in w for w in warnings)


def test_transcription_does_not_flag_a_normal_two_speaker_call():
    transcript = transcript_from_provider_payload(
        {
            "text": "Hello. Hi there. How can I help. I have a question.",
            "utterances": [
                {"speaker": "A", "start": 0, "end": 1000, "text": "Hello."},
                {"speaker": "B", "start": 1000, "end": 2000, "text": "Hi there."},
                {"speaker": "A", "start": 2000, "end": 3000, "text": "How can I help."},
                {"speaker": "B", "start": 3000, "end": 4000, "text": "I have a question."},
            ],
        }
    )

    warnings = transcript["normalization"]["warnings"]
    assert not any("diarization inconsistency" in w for w in warnings)


def test_transcription_provider_uploads_polls_and_normalizes_response():
    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code
            self.ok = status_code < 400

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self):
            self.post_calls = []
            self.get_calls = []

        def post(self, url, **kwargs):
            self.post_calls.append((url, kwargs))
            if url.endswith("/v2/upload"):
                return FakeResponse({"upload_url": "https://provider.test/uploaded-audio"})
            return FakeResponse({"id": "transcript-1", "status": "queued"})

        def get(self, url, **kwargs):
            self.get_calls.append((url, kwargs))
            return FakeResponse(
                {
                    "id": "transcript-1",
                    "status": "completed",
                    "text": "Hello",
                    "utterances": [{"speaker": "A", "start": 0, "end": 900, "text": "Hello"}],
                }
            )

    session = FakeSession()
    provider = AssemblyAITranscriptionProvider(
        TranscriptionSettings("assemblyai", "test-key", "https://provider.test", 10, 0),
        session=session,
        sleep_fn=lambda _seconds: None,
    )

    transcript = provider.transcribe(b"audio-bytes", "demo.mp3", "audio/mpeg")

    assert transcript["lines"][0]["speaker"] == "Speaker A"
    assert session.post_calls[0][0].endswith("/v2/upload")
    assert session.post_calls[1][1]["json"]["speaker_labels"] is True
    assert session.get_calls[0][0].endswith("/v2/transcript/transcript-1")


def _risk_transcript():
    return {
        "source": "transcribed",
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker A",
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "text": "I did not contact an attorney.",
            },
            {
                "line": 2,
                "speaker": "Speaker B",
                "start_seconds": 2.0,
                "end_seconds": 5.0,
                "text": "Please stop calling me and remove me from your list.",
            },
        ],
    }


def test_risk_rules_handle_negation_and_active_contact_restriction():
    result = run_risk_rules(_risk_transcript())

    assert [finding["type"] for finding in result["findings"]] == ["contact_restriction"]
    assert result["observations"][0]["type"] == "legal_escalation_mention"
    assert result["observations"][0]["active"] is False
    assert detect_risks(_risk_transcript())[0]["evidence"]["line"] == 2


def test_risk_rules_do_not_suppress_attorney_mention_after_unrelated_negation():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker B",
                "start_seconds": 0.0,
                "end_seconds": 3.0,
                "text": "I do not know if my attorney called you yet.",
            }
        ]
    }

    categories = {finding["type"] for finding in detect_risks(transcript)}
    assert "attorney_representation" in categories


def test_risk_rules_detect_demo_call_categories():
    transcript = {
        "lines": [
            {
                "line": 8,
                "speaker": "Speaker B",
                "start_seconds": 52.0,
                "end_seconds": 65.0,
                "text": "I am going to have my attorney look at these charges. Take me off from your list and stop calling me.",
            }
        ]
    }

    categories = {finding["type"] for finding in detect_risks(transcript)}
    assert categories == {"legal_escalation", "attorney_representation", "contact_restriction"}


def test_review_item_id_is_stable_for_the_same_finding():
    finding_a = {"type": "contact_restriction", "detail": "Stop calling me.", "evidence": {"line": 2}}
    finding_b = {"type": "contact_restriction", "detail": "Stop calling me.", "evidence": {"line": 2}}
    different_finding = {"type": "contact_restriction", "detail": "Stop calling me.", "evidence": {"line": 3}}

    assert review_item_id(finding_a) == review_item_id(finding_b)
    assert review_item_id(finding_a) != review_item_id(different_finding)


def test_with_review_defaults_fills_gaps_without_overwriting_existing_decisions():
    fresh_item = with_review_defaults({"type": "contact_restriction", "detail": "Stop calling me."})
    assert fresh_item["resolution"] is None
    assert fresh_item["resolution_note"] is None
    assert fresh_item["resolved_at"] is None
    assert isinstance(fresh_item["id"], str) and fresh_item["id"]

    already_decided = with_review_defaults(
        {
            "type": "contact_restriction",
            "detail": "Stop calling me.",
            "id": "existing-id",
            "resolution": "approved",
            "resolution_note": "Checked with supervisor.",
            "resolved_at": "2026-07-01T10:00:00+00:00",
        }
    )
    assert already_decided["id"] == "existing-id"
    assert already_decided["resolution"] == "approved"
    assert already_decided["resolution_note"] == "Checked with supervisor."


def test_compliance_finding_shares_its_id_with_its_review_ticket():
    result = extract_call_intelligence(_risk_transcript())

    finding = result["compliance_findings"][0]
    ticket = next(item for item in result["review"]["items"] if item["type"] == finding["type"])

    assert finding["id"] == ticket["id"]


def test_visible_compliance_findings_drops_rejected_and_applies_corrections():
    findings = [
        {"id": "a", "type": "contact_restriction", "description": "Original wording.", "severity": "Red"},
        {"id": "b", "type": "wrong_number_or_person", "description": "Might be wrong number.", "severity": "Yellow"},
    ]
    review_items = [
        {"id": "a", "resolution": "rejected"},
        {"id": "b", "resolution": "corrected", "resolution_note": "Confirmed correct number, false alarm."},
    ]

    visible = visible_compliance_findings(findings, review_items)

    assert [item["id"] for item in visible] == ["b"]
    assert visible[0]["description"] == "Confirmed correct number, false alarm."
    assert overall_compliance_status(visible) == "Yellow"  # the Red finding was rejected


def test_extraction_baseline_is_safe_and_routes_deterministic_risks_to_review():
    result = extract_call_intelligence(_risk_transcript())

    assert result["status"] == "review_required"
    assert result["insights"]["summary"] is None
    assert result["review"]["required"] is True
    assert {item["type"] for item in result["compliance_findings"]} == {"contact_restriction"}


def test_risk_findings_use_red_yellow_green_severity():
    findings = detect_risks(_risk_transcript())

    assert findings[0]["severity"] == "Red"
    assert overall_compliance_status(findings) == "Red"
    assert overall_compliance_status([]) == "Green"
    assert overall_compliance_status([{"severity": "Yellow"}]) == "Yellow"


def test_resolve_relative_date_matches_assignment_example():
    call_date = date(2026, 7, 1)  # The assignment's own example call date.

    assert resolve_relative_date("Friday", call_date) == "2026-07-03"
    assert resolve_relative_date("by Friday", call_date) == "2026-07-03"
    assert resolve_relative_date("the 15th of next month", call_date) == "2026-08-15"
    assert resolve_relative_date("today", call_date) == "2026-07-01"
    assert resolve_relative_date("tomorrow", call_date) == "2026-07-02"
    assert resolve_relative_date("sometime soon", call_date) is None


def test_looks_like_resolved_date():
    assert looks_like_resolved_date("2026-07-03") is True
    assert looks_like_resolved_date("Friday") is False


def test_extraction_resolves_relative_due_date_phrase_from_model():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker A",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "text": "I can call you back on Friday.",
            }
        ]
    }

    def model(_prompt):
        return {
            "summary": "A follow-up was proposed.",
            "tag": "Follow-up",
            "decisions": [],
            "action_items": [
                {
                    "description": "Call the consumer back on Friday.",
                    "owner": "Speaker A",
                    "due_date": "Friday",
                    "evidence": {
                        "line": 1,
                        "excerpt": "I can call you back on Friday.",
                        "start_seconds": 1.0,
                        "end_seconds": 4.0,
                    },
                }
            ],
            "blockers": [],
            "sentiment": {"overall": None, "anger": None, "profanity": []},
            "confidence": "Medium",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    assert result["status"] == "completed"
    resolved_item = result["insights"]["action_items"][0]
    assert resolved_item["due_date"] == "2026-07-03"
    assert resolved_item["due_date_phrase"] == "Friday"


def test_extraction_rejects_unresolved_due_date_phrase():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker A",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "text": "I will call you back sometime soon.",
            }
        ]
    }

    def model(_prompt):
        return {
            "summary": "A vague follow-up was proposed.",
            "decisions": [],
            "action_items": [
                {
                    "description": "Call the consumer back.",
                    "owner": "Speaker A",
                    "due_date": "sometime soon",
                    "evidence": {
                        "line": 1,
                        "excerpt": "I will call you back sometime soon.",
                        "start_seconds": 1.0,
                        "end_seconds": 4.0,
                    },
                }
            ],
            "blockers": [],
            "sentiment": {"overall": None, "anger": None, "profanity": []},
            "confidence": "Medium",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    assert result["status"] == "review_required"
    assert result["insights"]["summary"] is None
    assert any("due_date" in error for error in result["validation"]["errors"])


def test_extraction_rejects_anger_claim_with_no_evidence():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker B",
                "start_seconds": 0.0,
                "end_seconds": 3.0,
                "text": "Please stop calling me.",
            }
        ]
    }
    response = {
        "summary": "The consumer sounded angry.",
        "decisions": [],
        "action_items": [],
        "blockers": [],
        "sentiment": {"overall": "Negative", "anger": "High", "profanity": []},
        "confidence": "Medium",
        "compliance_findings": [],
    }

    validation = validate_extraction_response(response, transcript)

    assert validation["valid"] is False
    assert any("anger_evidence" in error for error in validation["errors"])


def test_extraction_accepts_anger_claim_with_matching_evidence():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker B",
                "start_seconds": 0.0,
                "end_seconds": 3.0,
                "text": "Please stop calling me.",
            }
        ]
    }
    response = {
        "summary": "The consumer sounded angry.",
        "tag": "Angry Customer",
        "decisions": [],
        "action_items": [],
        "blockers": [],
        "sentiment": {
            "overall": "Negative",
            "anger": "High",
            "anger_evidence": [
                {
                    "line": 1,
                    "excerpt": "Please stop calling me.",
                    "start_seconds": 0.0,
                    "end_seconds": 3.0,
                }
            ],
            "profanity": [],
        },
        "confidence": "Medium",
        "compliance_findings": [],
    }

    validation = validate_extraction_response(response, transcript)

    assert validation["valid"] is True


def test_extraction_rejects_profanity_word_not_in_its_own_evidence():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker B",
                "start_seconds": 0.0,
                "end_seconds": 3.0,
                "text": "Please stop calling me.",
            }
        ]
    }
    response = {
        "summary": "The consumer used profanity.",
        "decisions": [],
        "action_items": [],
        "blockers": [],
        "sentiment": {
            "overall": "Negative",
            "anger": None,
            "profanity": [
                {
                    "word": "made up word",
                    "evidence": {
                        "line": 1,
                        "excerpt": "Please stop calling me.",
                        "start_seconds": 0.0,
                        "end_seconds": 3.0,
                    },
                }
            ],
        },
        "confidence": "Medium",
        "compliance_findings": [],
    }

    validation = validate_extraction_response(response, transcript)

    assert validation["valid"] is False
    assert any("does not appear in its own evidence excerpt" in error for error in validation["errors"])


def test_extraction_routes_missing_owner_to_review_without_inventing_one():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker A",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "text": "Someone needs to call the consumer back.",
            }
        ]
    }

    def model(_prompt):
        return {
            "summary": "A follow-up was mentioned but nobody claimed it.",
            "tag": "Follow-up",
            "decisions": [],
            "action_items": [
                {
                    "description": "Call the consumer back.",
                    "evidence": {
                        "line": 1,
                        "excerpt": "Someone needs to call the consumer back.",
                        "start_seconds": 1.0,
                        "end_seconds": 4.0,
                    },
                }
            ],
            "blockers": [],
            "sentiment": {"overall": None, "anger": None, "profanity": []},
            "confidence": "Medium",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    assert result["status"] == "review_required"
    assert result["insights"]["action_items"][0]["owner"] == "Unclear/Unassigned"
    assert result["review"]["required"] is True
    assert any(item["type"] == "unclear_owner" for item in result["review"]["items"])


def test_extraction_rejects_compliance_finding_with_invalid_severity():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker A",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "text": "I can call you back on Friday.",
            }
        ]
    }
    response = {
        "summary": "A follow-up was proposed.",
        "decisions": [],
        "action_items": [],
        "blockers": [],
        "sentiment": {"overall": None, "anger": None, "profanity": []},
        "confidence": None,
        "compliance_findings": [
            {
                "type": "contact_restriction",
                "severity": "high",
                "description": "Legacy severity value.",
                "evidence": {
                    "line": 1,
                    "excerpt": "I can call you back on Friday.",
                    "start_seconds": 1.0,
                    "end_seconds": 4.0,
                },
            }
        ],
    }

    validation = validate_extraction_response(response, transcript)

    assert validation["valid"] is False
    assert any("severity must be Red, Yellow, or Green" in error for error in validation["errors"])


def test_extraction_flags_a_dollar_amount_the_model_never_cited():
    """Regression test for a real bug found during independent QA: Gemini
    captured one payment commitment in a call but silently dropped a second,
    real one, with no error or warning. This is a heuristic safety net, not a
    guarantee -- it flags an uncited dollar amount for human review instead of
    letting it vanish invisibly.
    """

    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Consumer",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "text": "Could I pay $450 now and $450 on the 20th of next month?",
            },
            {
                "line": 2,
                "speaker": "Agent",
                "start_seconds": 4.0,
                "end_seconds": 8.0,
                "text": "I'll need my supervisor to approve that split.",
            },
        ]
    }

    def model(_prompt):
        return {
            "summary": "The agent will check with a supervisor.",
            "tag": "Payment Proposal",
            "decisions": [],
            "action_items": [],
            "blockers": [
                {
                    "description": "Needs supervisor approval.",
                    "evidence": {"line": 2, "excerpt": "I'll need my supervisor to approve that split."},
                }
            ],
            "sentiment": {"overall": None, "anger": None, "profanity": []},
            "confidence": "Medium",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    assert result["status"] == "review_required"
    flagged = [item for item in result["review"]["items"] if item["type"] == "possible_uncaptured_commitment"]
    assert len(flagged) == 1
    assert flagged[0]["evidence"]["line"] == 1
    assert "$450" in flagged[0]["detail"]


def test_extraction_does_not_flag_a_dollar_amount_that_was_cited():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Consumer",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "text": "I can pay $450 today.",
            }
        ]
    }

    def model(_prompt):
        return {
            "summary": "The consumer will pay $450 today.",
            "tag": "Payment",
            "decisions": [],
            "action_items": [
                {
                    "description": "Consumer to pay $450 today.",
                    "owner": "Consumer",
                    "evidence": {"line": 1, "excerpt": "I can pay $450 today."},
                }
            ],
            "blockers": [],
            "sentiment": {"overall": None, "anger": None, "profanity": []},
            "confidence": "High",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    assert result["status"] == "completed"
    assert not any(item["type"] == "possible_uncaptured_commitment" for item in result["review"]["items"])


def test_extraction_flags_a_severity_disagreement_between_rule_and_model():
    """The rule engine and the model can independently flag the same finding
    (same type, same line) with different severities. Merging used to keep
    only the rule's version with no record the model ever disagreed -- this
    checks that disagreement now surfaces as its own review item instead of
    being silently dropped.
    """

    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Consumer",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "text": "Please stop calling me, I'm about to lose my house over this.",
            }
        ]
    }

    def model(_prompt):
        return {
            "summary": "The consumer asked the agent to stop calling.",
            "tag": "Cease And Desist Request",
            "decisions": [],
            "action_items": [],
            "blockers": [],
            "sentiment": {"overall": "Negative", "anger": None, "profanity": []},
            "confidence": "High",
            "compliance_findings": [
                {
                    "type": "contact_restriction",
                    "severity": "Yellow",
                    "description": "A mild request to stop calling.",
                    "evidence": {
                        "line": 1,
                        "excerpt": "stop calling me",
                        "start_seconds": 0.0,
                        "end_seconds": 4.0,
                    },
                }
            ],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    # The rule's Red finding still wins the (type, line) slot -- the model
    # never removes or downgrades it.
    contact_findings = [f for f in result["compliance_findings"] if f["type"] == "contact_restriction"]
    assert len(contact_findings) == 1
    assert contact_findings[0]["severity"] == "Red"
    assert contact_findings[0]["source"] == "deterministic_rule"

    # But the model's disagreement is not lost -- it becomes its own ticket.
    disagreements = [item for item in result["review"]["items"] if item["type"] == "rule_model_disagreement"]
    assert len(disagreements) == 1
    assert disagreements[0]["severity"] == "Red"  # the worse of Red/Yellow
    assert "Red" in disagreements[0]["detail"]
    assert "Yellow" in disagreements[0]["detail"]


def _minimal_valid_response(**overrides):
    response = {
        "summary": "A short call summary.",
        "tag": "Routine Follow-up",
        "decisions": [],
        "action_items": [],
        "blockers": [],
        "sentiment": {"overall": None, "anger": None, "profanity": []},
        "confidence": "Medium",
        "compliance_findings": [],
    }
    response.update(overrides)
    return response


def test_tag_field_is_required_like_summary():
    response = _minimal_valid_response()
    del response["tag"]

    validation = validate_extraction_response(response, {"lines": []})

    assert validation["valid"] is False
    assert any("tag" in error for error in validation["errors"])


def test_tag_accepts_null_and_short_strings():
    validation_null = validate_extraction_response(_minimal_valid_response(tag=None), {"lines": []})
    validation_short = validate_extraction_response(
        _minimal_valid_response(tag="Legal Escalation"), {"lines": []}
    )

    assert validation_null["valid"] is True
    assert validation_short["valid"] is True


def test_tag_rejects_a_full_sentence_instead_of_a_short_label():
    long_tag = "This is a full sentence describing the call in detail, not a short label" * 2

    validation = validate_extraction_response(_minimal_valid_response(tag=long_tag), {"lines": []})

    assert validation["valid"] is False
    assert any("60 characters" in error for error in validation["errors"])


def test_extraction_accepts_valid_model_response_with_matching_evidence():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker A",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "text": "I can call you back on Friday.",
            }
        ]
    }

    def model(_prompt):
        return {
            "summary": "The agent offered a Friday follow-up.",
            "tag": "Follow-up",
            "decisions": [],
            "action_items": [
                {
                    "description": "Call the consumer back on Friday.",
                    "owner": "Speaker A",
                    "evidence": {
                        "line": 1,
                        "excerpt": "I can call you back on Friday.",
                        "start_seconds": 1.0,
                        "end_seconds": 4.0,
                    },
                }
            ],
            "blockers": [],
            "sentiment": {"overall": "Neutral", "anger": None, "profanity": []},
            "confidence": "Medium",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-09-26")

    assert result["status"] == "completed"
    assert result["validation"]["valid"] is True
    assert result["insights"]["action_items"][0]["evidence"]["line"] == 1
    assert result["review"]["required"] is False


def test_extraction_discards_model_claim_with_wrong_evidence():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker A",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "text": "I can call you back on Friday.",
            }
        ]
    }

    invalid = {
        "summary": "The consumer agreed to pay today.",
        "decisions": [
            {
                "description": "Confirmed payment today.",
                "evidence": {
                    "line": 1,
                    "excerpt": "The consumer agreed to pay today.",
                    "start_seconds": 1.0,
                    "end_seconds": 4.0,
                },
            }
        ],
        "action_items": [],
        "blockers": [],
        "sentiment": {"overall": None, "anger": None, "profanity": []},
        "confidence": "High",
        "compliance_findings": [],
    }

    result = extract_call_intelligence(transcript, model=lambda _prompt: invalid)

    assert result["status"] == "review_required"
    assert result["insights"]["summary"] is None
    assert result["validation"]["valid"] is False
    assert result["review"]["required"] is True


def test_extraction_survives_model_rounding_evidence_timestamps_slightly():
    """Regression test: a real bug seen with live Gemini output on audio calls.

    The model correctly identified the right line, but echoed back a rounded
    version of its timestamp (21.2 instead of the real 21.219). The old
    normalizer only filled in *missing* timestamps, so this mismatch caused
    the entire otherwise-correct extraction to be discarded. Timestamps are
    fully determined by a valid line reference, so they should always be
    taken from the transcript, not trusted from the model.
    """

    transcript = {
        "lines": [
            {
                "line": 4,
                "speaker": "Speaker B",
                "start_seconds": 21.219,
                "end_seconds": 30.240,
                "text": "I don't have $1,400. This is ridiculous!",
            }
        ]
    }

    def model(_prompt):
        return {
            "summary": "The consumer disputed the balance.",
            "tag": "Payment Dispute",
            "decisions": [],
            "action_items": [],
            "blockers": [
                {
                    "description": "Consumer disputes the balance amount.",
                    "evidence": {
                        "line": 4,
                        "excerpt": "I don't have $1,400. This is ridiculous!",
                        # Rounded, not exact -- this is what broke it live.
                        "start_seconds": 21.2,
                        "end_seconds": 30.24,
                    },
                }
            ],
            "sentiment": {"overall": "Negative", "anger": None, "profanity": []},
            "confidence": "High",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-09-27")

    assert result["status"] == "completed"
    assert result["insights"]["summary"] == "The consumer disputed the balance."
    assert result["insights"]["blockers"][0]["evidence"]["start_seconds"] == 21.219


def test_extraction_survives_model_rounding_anger_evidence_timestamps():
    """Same bug class as the blocker-evidence regression above, found live in
    sentiment.anger_evidence specifically -- it uses a separate code path
    from decisions/action_items/blockers/compliance_findings and was missed
    by the first fix.
    """

    transcript = {
        "lines": [
            {
                "line": 2,
                "speaker": "Speaker B",
                "start_seconds": 6.418,
                "end_seconds": 12.515,
                "text": "Why do you guys keep bothering me about this?",
            }
        ]
    }

    def model(_prompt):
        return {
            "summary": "The consumer sounded frustrated.",
            "tag": "Angry Customer",
            "decisions": [],
            "action_items": [],
            "blockers": [],
            "sentiment": {
                "overall": "Negative",
                "anger": "Moderate",
                "anger_evidence": [
                    {
                        "line": 2,
                        "excerpt": "Why do you guys keep bothering me about this?",
                        "start_seconds": 6.4,  # rounded, not exact
                        "end_seconds": 12.5,  # rounded, not exact
                    }
                ],
                "profanity": [],
            },
            "confidence": "High",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-09-27")

    assert result["status"] == "completed"
    assert result["insights"]["sentiment"]["anger_evidence"][0]["start_seconds"] == 6.418


def test_model_response_normalization_copies_authoritative_timestamps_and_sentiment_shape():
    transcript = {
        "lines": [
            {
                "line": 1,
                "speaker": "Speaker A",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "text": "I can call you back on Friday.",
            }
        ]
    }
    response = {
        "summary": "A follow-up was proposed.",
        "tag": "Follow-up",
        "decisions": [],
        "action_items": [
            {
                "action": "Call the consumer back on Friday.",
                "evidence": {"line": 1, "excerpt": "I can call you back on Friday."},
            }
        ],
        "blockers": [],
        "sentiment": "Neutral",
        "confidence": "medium",
        "compliance_findings": [],
    }

    normalized, warnings = normalize_model_response(response, transcript)
    validation = validate_extraction_response(normalized, transcript)

    assert validation["valid"] is True
    assert normalized["sentiment"] == {
        "overall": "Neutral",
        "anger": None,
        "anger_evidence": [],
        "profanity": [],
    }
    assert normalized["action_items"][0]["description"] == "Call the consumer back on Friday."
    assert normalized["action_items"][0]["evidence"]["start_seconds"] == 1.0
    assert normalized["action_items"][0]["evidence"]["end_seconds"] == 4.0
    assert warnings


def test_extraction_prompt_contains_few_shot_rules_and_output_contract():
    prompt = build_extraction_prompt({"lines": []}, call_date="2026-09-26")

    assert "Example valid output" in prompt
    assert "Do not treat a question, possibility, or proposal" in prompt
    assert "sentiment must be an object" in prompt
    assert "Return JSON only" in prompt


def test_extraction_parser_supports_json_code_fence_and_rejects_non_object():
    assert parse_model_response('```json\n{"summary": null}\n```') == {"summary": None}
    try:
        parse_model_response("[]")
    except ValueError as error:
        assert "JSON object" in str(error)
    else:
        raise AssertionError("Expected non-object model response to be rejected")


def test_full_pipeline_record_matches_json_schema():
    """The record the real app builds must always match the documented schema."""

    record = create_pending_call_record(
        {"name": "call.txt", "type": "text/plain", "size": 5},
        "CALL-0001",
        now=NOW,
    )
    transcript = normalize_transcript(
        "call.txt", b"Agent: Hello.\nConsumer: Please stop calling me.\n"
    )
    record = apply_normalized_transcript(record, transcript)

    def model(_prompt):
        return {
            "summary": "Test summary.",
            "decisions": [],
            "action_items": [],
            "blockers": [],
            "sentiment": {"overall": "Negative", "anger": None, "profanity": []},
            "confidence": "Medium",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(
        record["transcript"], model=model, call_date=record["metadata"]["call_date"]
    )
    record = apply_extraction_to_record(record, result)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(record, schema)


def test_gemini_adapter_rotates_to_next_key_when_one_hits_its_quota():
    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.ok = status_code < 400
            self._payload = payload

        def json(self):
            return self._payload

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append(kwargs["headers"]["x-goog-api-key"])
            if kwargs["headers"]["x-goog-api-key"] == "first-key":
                # This key's quota is exhausted.
                return Response(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
            return Response(200, {"candidates": [{"content": {"parts": [{"text": '{"summary": "ok"}'}]}}]})

    session = Session()
    model = GeminiExtractionModel(
        GeminiSettings(
            api_keys=("first-key", "second-key"),
            model="gemini-test",
            base_url="https://example.test/v1beta",
            timeout_seconds=10,
        ),
        session=session,
    )

    assert model("return JSON") == '{"summary": "ok"}'
    assert session.calls == ["first-key", "second-key"]


def test_gemini_adapter_does_not_rotate_keys_on_the_last_key_or_non_quota_errors():
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.ok = status_code < 400

        def json(self):
            return {"error": {"message": "RESOURCE_EXHAUSTED"}}

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append(kwargs["headers"]["x-goog-api-key"])
            return Response(429)

    session = Session()
    model = GeminiExtractionModel(
        GeminiSettings(
            api_keys=("only-key",),
            model="gemini-test",
            base_url="https://example.test/v1beta",
            timeout_seconds=10,
        ),
        session=session,
    )

    try:
        model("return JSON")
    except ExtractionError as error:
        assert error.code == "GEMINI_REQUEST_FAILED"
    else:
        raise AssertionError("Expected the last key's quota failure to raise, not retry forever")
    assert session.calls == ["only-key"]  # never rotates past the only configured key


def test_gemini_adapter_returns_candidate_text_without_exposing_key():
    class Response:
        ok = True
        status_code = 200

        def json(self):
            return {
                "candidates": [
                    {"content": {"parts": [{"text": '{"summary": "ok"}'}]}}
                ]
            }

    class Session:
        def __init__(self):
            self.call = None

        def post(self, url, **kwargs):
            self.call = (url, kwargs)
            return Response()

    session = Session()
    model = GeminiExtractionModel(
        GeminiSettings(
            api_keys=("test-secret",),
            model="gemini-test",
            base_url="https://example.test/v1beta",
            timeout_seconds=10,
        ),
        session=session,
    )

    assert model("return JSON") == '{"summary": "ok"}'
    assert session.call[0].endswith("/models/gemini-test:generateContent")
    assert session.call[1]["headers"]["x-goog-api-key"] == "test-secret"
    assert session.call[1]["json"]["generationConfig"]["responseMimeType"] == "application/json"

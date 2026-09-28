"""Golden call regression tests.

Each test loads one small, realistic transcript from ``test/golden_calls/`` and
runs it through the real pipeline -- the same normalization, risk rules, and
extraction/validation code the app uses -- then checks a known-correct answer.

This is deliberately not a benchmark score. It is a fixed, human-readable set
of cases matching the categories the project's own spec lists under "AI
regression tests", so a change that breaks one of these is caught immediately
and the exact expected behavior is documented right next to the test.

Where a case depends on the model's own judgment (for example: is this a
confirmed agreement or just a proposal?), no deterministic classifier exists in
this codebase yet -- that is intentionally out of scope, and the test says so.
What every case here checks is the part the code *is* responsible for:
evidence grounding, date resolution, negation handling, and review routing.
"""

from datetime import datetime, timezone
from pathlib import Path

from python_app.extraction import extract_call_intelligence
from python_app.intake import create_pending_call_record
from python_app.risk_rules import detect_risks, overall_compliance_status, run_risk_rules
from python_app.transcripts import normalize_transcript

GOLDEN_DIR = Path(__file__).parent / "golden_calls"


def _load(name: str) -> dict:
    return normalize_transcript(name, (GOLDEN_DIR / name).read_bytes())


# ---------------------------------------------------------------------------
# The assignment's own worked example, end to end.
# ---------------------------------------------------------------------------


def test_golden_assignment_example_resolves_dates_exactly_like_the_assignment():
    """Reproduces the assignment's own numbers: call date 2026-07-01,
    "the 15th of next month" -> 2026-08-15, "Friday" -> 2026-07-03.
    """

    transcript = _load("assignment_example.txt")
    settlement_line = next(line for line in transcript["lines"] if "supervisor" in line["text"])
    proposal_line = next(line for line in transcript["lines"] if "$700 now" in line["text"])

    def model(_prompt):
        return {
            "summary": "The consumer proposed a split payment; the agent needs supervisor approval.",
            "tag": "Payment Proposal",
            "decisions": [
                {
                    "description": "Consumer proposed paying $700 now and $700 next month.",
                    "evidence": {
                        "line": proposal_line["line"],
                        "excerpt": proposal_line["text"],
                    },
                }
            ],
            "action_items": [
                {
                    "description": "Consumer to pay the second $700 installment.",
                    "owner": "Consumer",
                    "due_date": "the 15th of next month",
                    "evidence": {
                        "line": settlement_line["line"],
                        "excerpt": "$700 on the 15th of next month",
                    },
                },
                {
                    "description": "Follow up on supervisor approval for the settlement split.",
                    "owner": "Agent",
                    "due_date": "Friday",
                    "evidence": {
                        "line": settlement_line["line"],
                        "excerpt": "I'll follow up by Friday",
                    },
                },
            ],
            "blockers": [
                {
                    "description": "Settlement split needs supervisor approval before it is final.",
                    "evidence": {
                        "line": settlement_line["line"],
                        "excerpt": "I'll need a supervisor to approve the settlement split",
                    },
                }
            ],
            "sentiment": {"overall": "Neutral", "anger": None, "profanity": []},
            "confidence": "Medium",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    assert result["status"] == "completed"
    action_items = result["insights"]["action_items"]
    installment, follow_up = action_items[0], action_items[1]
    assert installment["due_date"] == "2026-08-15"
    assert installment["due_date_phrase"] == "the 15th of next month"
    assert follow_up["due_date"] == "2026-07-03"
    assert follow_up["due_date_phrase"] == "Friday"
    assert result["insights"]["blockers"][0]["description"].startswith("Settlement split")


# ---------------------------------------------------------------------------
# Deterministic compliance/risk categories.
# ---------------------------------------------------------------------------


def test_golden_cease_and_desist_is_red_and_routes_to_review():
    transcript = _load("cease_and_desist.txt")
    findings = detect_risks(transcript)

    assert {f["type"] for f in findings} == {"contact_restriction"}
    assert overall_compliance_status(findings) == "Red"


def test_golden_legal_mention_is_red():
    transcript = _load("legal_mention.txt")
    findings = detect_risks(transcript)

    assert {"legal_escalation", "attorney_representation"} <= {f["type"] for f in findings}
    assert overall_compliance_status(findings) == "Red"


def test_golden_wrong_number_is_yellow_not_red():
    transcript = _load("wrong_number.txt")
    findings = detect_risks(transcript)

    assert {f["type"] for f in findings} == {"wrong_number_or_person"}
    assert overall_compliance_status(findings) == "Yellow"


def test_golden_recording_consent_uncertainty_is_red():
    transcript = _load("recording_consent.txt")
    findings = detect_risks(transcript)

    assert {f["type"] for f in findings} == {"recording_consent_uncertainty"}
    assert overall_compliance_status(findings) == "Red"


def test_golden_sensitive_information_is_red():
    transcript = _load("sensitive_information.txt")
    findings = detect_risks(transcript)

    assert {f["type"] for f in findings} == {"sensitive_financial_information"}
    assert overall_compliance_status(findings) == "Red"


def test_golden_negated_legal_mention_is_green_not_red():
    """"I did not contact an attorney" must not be treated as an active risk."""

    transcript = _load("negated_legal_mention.txt")
    result = run_risk_rules(transcript)

    assert result["findings"] == []
    assert result["observations"][0]["active"] is False
    assert overall_compliance_status(result["findings"]) == "Green"


# ---------------------------------------------------------------------------
# Dates, speakers, and grounding.
# ---------------------------------------------------------------------------


def test_golden_ambiguous_date_is_rejected_not_guessed():
    transcript = _load("ambiguous_date.txt")
    callback_line = next(line for line in transcript["lines"] if "next week" in line["text"])

    def model(_prompt):
        return {
            "summary": "The consumer promised to call back, with no specific date.",
            "decisions": [],
            "action_items": [
                {
                    "description": "Consumer to call back with a payment update.",
                    "owner": "Consumer",
                    "due_date": "sometime next week",
                    "evidence": {
                        "line": callback_line["line"],
                        "excerpt": "I'll call you back sometime next week",
                    },
                }
            ],
            "blockers": [],
            "sentiment": {"overall": None, "anger": None, "profanity": []},
            "confidence": "Low",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    assert result["status"] == "review_required"
    assert any("due_date" in error for error in result["validation"]["errors"])


def test_golden_unclear_speaker_roles_are_neutral_not_invented():
    transcript = _load("unclear_speaker_roles.txt")

    assert transcript["normalization"]["speaker_labels_inferred"] is True
    assert all(line["speaker"] == "Speaker 1" for line in transcript["lines"])


def test_golden_missing_call_date_falls_back_and_is_marked():
    record = create_pending_call_record(
        {"name": "call.txt", "type": "text/plain", "size": 10},
        "CALL-0001",
        now=datetime(2026, 9, 26, 9, 0, tzinfo=timezone.utc),
    )

    assert record["metadata"]["call_date"] == "2026-09-26"
    assert record["metadata"]["date_source"] == "default_current_date"


def test_golden_anger_is_accepted_only_with_real_evidence():
    transcript = _load("anger_and_profanity.txt")
    angry_line = next(line for line in transcript["lines"] if "angry" in line["text"])

    def model(_prompt):
        return {
            "summary": "The consumer expressed strong frustration about repeated calls.",
            "tag": "Angry Customer",
            "decisions": [],
            "action_items": [],
            "blockers": [],
            "sentiment": {
                "overall": "Negative",
                "anger": "High",
                "anger_evidence": [
                    {"line": angry_line["line"], "excerpt": "I am so angry about all these calls"}
                ],
                "profanity": [],
            },
            "confidence": "Medium",
            "compliance_findings": [],
        }

    result = extract_call_intelligence(transcript, model=model, call_date="2026-07-01")

    assert result["status"] == "completed"
    assert result["insights"]["sentiment"]["anger"] == "High"
    assert result["insights"]["sentiment"]["anger_evidence"][0]["line"] == angry_line["line"]

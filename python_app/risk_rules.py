"""Deterministic risk signals for debt-collection transcripts.

These rules are an explainable baseline, not a legal decision engine. They produce
reviewable observations with transcript evidence. Semantic model output can be merged
with these findings later by ``extraction.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


NEGATION_RE = re.compile(
    r"\b(?:no|not|never|didn['’]?t|do(?:es|id)?n['’]?t|without|neither)\b",
    re.IGNORECASE,
)

# A rough clause boundary: sentence punctuation or a connecting word that
# usually starts a new thought. A negation on the far side of one of these
# belongs to a different clause and should not suppress a match.
CLAUSE_BREAK_RE = re.compile(
    r"[.!?;,]|\b(?:but|and|because|if|although|though|so|which|who|that)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RiskRule:
    category: str
    severity: str
    description: str
    phrases: tuple[str, ...]


RISK_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        "legal_escalation",
        "Red",
        "The speaker may be considering legal advice, legal action, or a complaint.",
        (
            "attorney",
            "lawyer",
            "legal counsel",
            "legal advice",
            "seek counsel",
            "legal action",
            "lawsuit",
            "take this to court",
            "take you to court",
            "sue you",
            "file a complaint",
            "report you",
        ),
    ),
    RiskRule(
        "attorney_representation",
        "Red",
        "The speaker may be represented by an attorney or directing contact to counsel.",
        (
            "my attorney",
            "my lawyer",
            "my legal counsel",
            "represented by counsel",
            "represented by an attorney",
            "contact my attorney",
            "speak with my lawyer",
            "talk to my lawyer",
            "contact my counsel",
        ),
    ),
    RiskRule(
        "contact_restriction",
        "Red",
        "The speaker requested that contact stop or use a restricted channel.",
        (
            "stop calling me",
            "stop contacting me",
            "do not contact me",
            "don't contact me",
            "cease contact",
            "cease all communication",
            "remove me from your list",
            "take me off your list",
            "do not call my workplace",
            "do not call me at work",
            "only contact me by mail",
            "do not text me",
            "don't text me",
            "do not email me",
            "don't email me",
        ),
    ),
    RiskRule(
        "debt_dispute",
        "Red",
        "The speaker may dispute ownership, amount, or validity of the debt.",
        (
            "not my debt",
            "this is not my debt",
            "i do not owe this",
            "i don't owe this",
            "i dispute this debt",
            "dispute the debt",
            "the amount is wrong",
            "the amount is incorrect",
            "wrong amount",
            "prove that i owe this",
            "send me verification",
            "verify this debt",
            "show me the original creditor",
        ),
    ),
    RiskRule(
        "harassment_or_threat",
        "Red",
        "The speaker reports harassment, excessive contact, or a threat.",
        (
            "you are harassing me",
            "you're harassing me",
            "you keep harassing me",
            "you keep calling me",
            "you have called too many times",
            "you are threatening me",
            "you're threatening me",
            "threatening legal action",
        ),
    ),
    RiskRule(
        "wrong_number_or_person",
        "Yellow",
        "The speaker may be the wrong person or the number may be incorrect.",
        (
            "wrong person",
            "wrong number",
            "this is not the person",
            "this number does not belong to",
            "you have the wrong person",
            "i do not know that person",
        ),
    ),
    RiskRule(
        "recording_consent_uncertainty",
        "Red",
        "The speaker questions or objects to recording or consent.",
        (
            "i did not consent to being recorded",
            "i didn't consent to being recorded",
            "i do not consent to this recording",
            "i don't consent to this recording",
            "why are you recording me",
            "stop recording me",
            "i never agreed to be recorded",
        ),
    ),
    RiskRule(
        "financial_hardship",
        "Yellow",
        "The speaker indicates financial hardship or inability to pay.",
        (
            "i cannot afford this",
            "i can't afford this",
            "i am struggling financially",
            "i'm struggling financially",
            "i lost my job",
            "i have no money right now",
            "i cannot make the full payment",
            "i can't make the full payment",
        ),
    ),
    RiskRule(
        "sensitive_financial_information",
        "Red",
        "The transcript may contain sensitive payment or identity information.",
        (
            "social security number",
            "routing number",
            "bank account number",
            "debit card number",
            "credit card number",
            "card number",
            "security code",
            "pin number",
        ),
    ),
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _word_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", value.casefold())


def _current_clause(before: str) -> str:
    """Return only the part of ``before`` since the last clause break.

    A negation word in an earlier clause (for example "I do not know if my
    attorney called") does not apply to a phrase in a later clause, so the
    search for a negation must stay inside the same clause as the match.
    """

    breaks = list(CLAUSE_BREAK_RE.finditer(before))
    if not breaks:
        return before
    return before[breaks[-1].end():]


def _is_negated(text: str, match_start: int, phrase: str) -> bool:
    """Detect a simple negation immediately before a matched phrase, in the same clause.

    This intentionally stays conservative. Ambiguous cases should become review
    signals rather than being silently marked safe.
    """

    before = _current_clause(text[:match_start])
    preceding_tokens = _word_tokens(before)[-7:]
    if not preceding_tokens:
        return False

    normalized_phrase = _normalize(phrase)
    phrase_tokens = _word_tokens(normalized_phrase)
    # Phrases beginning with "do not" or "don't" are themselves active requests,
    # for example "do not contact me". Do not suppress those matches.
    if phrase_tokens and phrase_tokens[0] in {"do", "don't", "don’t"}:
        return False
    return bool(NEGATION_RE.search(" ".join(preceding_tokens)))


def _evidence(line: dict[str, Any], matched_phrase: str) -> dict[str, Any]:
    return {
        "line": line.get("line"),
        "speaker": line.get("speaker"),
        "start_seconds": line.get("start_seconds"),
        "end_seconds": line.get("end_seconds"),
        "excerpt": line.get("text", "").strip(),
        "matched_phrase": matched_phrase,
    }


def _lines(transcript: dict[str, Any]) -> Iterable[dict[str, Any]]:
    raw_lines = transcript.get("lines", []) if isinstance(transcript, dict) else []
    if not isinstance(raw_lines, list):
        return []
    return (line for line in raw_lines if isinstance(line, dict) and isinstance(line.get("text"), str))


def run_risk_rules(transcript: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return active findings and suppressed observations with source evidence."""

    findings: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, str]] = set()

    for line in _lines(transcript):
        text = line["text"]
        normalized_text = _normalize(text)
        for rule in RISK_RULES:
            for phrase in rule.phrases:
                normalized_phrase = _normalize(phrase)
                match_start = normalized_text.find(normalized_phrase)
                if match_start < 0:
                    continue

                key = (rule.category, line.get("line"), normalized_phrase)
                if key in seen:
                    continue
                seen.add(key)

                evidence = _evidence(line, phrase)
                if _is_negated(normalized_text, match_start, normalized_phrase):
                    observations.append(
                        {
                            "type": f"{rule.category}_mention",
                            "active": False,
                            "reason": "A negation was detected near the matched phrase.",
                            "evidence": evidence,
                        }
                    )
                    continue

                findings.append(
                    {
                        "type": rule.category,
                        "severity": rule.severity,
                        "confidence": "High",
                        "description": rule.description,
                        "source": "deterministic_rule",
                        "evidence": evidence,
                        "review_status": "Needs review" if rule.severity == "Red" else "Review recommended",
                    }
                )

    # Keep one finding per category and line; multiple matched phrases in the same
    # sentence should not create duplicate review work.
    deduplicated: list[dict[str, Any]] = []
    seen_findings: set[tuple[str, Any]] = set()
    for finding in findings:
        key = (finding["type"], finding["evidence"].get("line"))
        if key not in seen_findings:
            seen_findings.add(key)
            deduplicated.append(finding)

    return {"findings": deduplicated, "observations": observations}


def detect_risks(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Convenience function returning only active deterministic findings."""

    return run_risk_rules(transcript)["findings"]


def overall_compliance_status(findings: Iterable[dict[str, Any]]) -> str:
    """Reduce every active finding to one Red/Yellow/Green call-level status.

    Any Red finding wins. If there is no Red but there is a Yellow, the call is
    Yellow. With no findings at all, the call is Green.
    """

    severities = {finding.get("severity") for finding in findings if isinstance(finding, dict)}
    if "Red" in severities:
        return "Red"
    if "Yellow" in severities:
        return "Yellow"
    return "Green"

"""Shared helpers for building and updating human-review items.

Every review item is a small "ticket": something the system was not sure
about, with a stable id and a place to record what a human reviewer decided
about it. The id is based on what the item actually is (its type, the
transcript line it points at, and its message), not on when it was created,
so the same underlying issue keeps the same id across a re-analysis. That
lets a reviewer's decision be carried forward instead of silently wiped out
the next time extraction runs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

RESOLUTIONS = {"approved", "corrected", "rejected", "resolved"}


def review_item_id(item: dict[str, Any]) -> str:
    """Return a stable id for a review item, based on its content."""

    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    key = "|".join(
        [
            str(item.get("type", "")),
            str(evidence.get("line", "")),
            str(item.get("detail", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def with_review_defaults(item: dict[str, Any]) -> dict[str, Any]:
    """Attach a stable id and empty resolution fields to a fresh review item.

    Nothing here is overwritten if the item already carries these fields --
    this is only meant to fill in the gaps for a freshly generated item.
    """

    completed = dict(item)
    completed.setdefault("id", review_item_id(item))
    completed.setdefault("resolution", None)
    completed.setdefault("resolution_note", None)
    completed.setdefault("resolved_at", None)
    return completed


def rejected_finding_ids(review_items: Iterable[dict[str, Any]]) -> set[str]:
    """Ids of every review item a human has rejected."""

    return {
        item.get("id")
        for item in review_items
        if isinstance(item, dict) and item.get("resolution") == "rejected" and item.get("id")
    }


def corrected_descriptions(review_items: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Map of id -> replacement text for every review item a human has corrected."""

    return {
        item["id"]: item["resolution_note"]
        for item in review_items
        if isinstance(item, dict)
        and item.get("resolution") == "corrected"
        and item.get("id")
        and isinstance(item.get("resolution_note"), str)
        and item["resolution_note"].strip()
    }


def visible_compliance_findings(
    findings: Iterable[dict[str, Any]], review_items: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply human review decisions to the findings a report should show.

    A finding whose matching review item was rejected is dropped completely --
    it no longer counts toward the overall compliance status either. A finding
    whose matching review item was corrected shows the reviewer's replacement
    text instead of the original AI description, per the project rule that a
    human correction replaces the AI result.
    """

    review_items = list(review_items)
    excluded = rejected_finding_ids(review_items)
    corrections = corrected_descriptions(review_items)

    visible: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict) or finding.get("id") in excluded:
            continue
        if finding.get("id") in corrections:
            finding = {**finding, "description": corrections[finding["id"]]}
        visible.append(finding)
    return visible

"""Deterministic resolution of safe relative-date phrases against a call date.

Only a small, unambiguous set of phrases is handled here: weekday names
("Friday", "next Friday", "by Friday"), "today", "tomorrow", and
"<day> of next month" -- the exact patterns used in the assignment's own
example. Anything else is left alone so the caller can mark it unresolved
and route it to human review instead of guessing.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_PREFIX_RE = re.compile(r"^(?:by|on|this coming|next)\s+", re.IGNORECASE)
_NEXT_MONTH_ORDINAL_RE = re.compile(
    r"^(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+next\s+month$"
)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def looks_like_resolved_date(value: str) -> bool:
    """True when a value already looks like a plain calendar date, not a phrase."""

    return bool(_ISO_DATE_RE.match(value.strip()))


def _shift_month(source: date, months: int) -> tuple[int, int]:
    month_index = source.month - 1 + months
    year = source.year + month_index // 12
    month = month_index % 12 + 1
    return year, month


def resolve_relative_date(phrase: str, call_date: date) -> str | None:
    """Resolve one safe relative-date phrase against the call date.

    Returns an ISO ``YYYY-MM-DD`` string for a known safe pattern, otherwise
    ``None``. This function never guesses at anything ambiguous.
    """

    normalized = re.sub(r"\s+", " ", phrase.strip().lower())
    normalized = _PREFIX_RE.sub("", normalized)

    if normalized == "today":
        return call_date.isoformat()
    if normalized == "tomorrow":
        return (call_date + timedelta(days=1)).isoformat()

    if normalized in WEEKDAYS:
        target_weekday = WEEKDAYS[normalized]
        offset = (target_weekday - call_date.weekday()) % 7
        if offset == 0:
            offset = 7
        return (call_date + timedelta(days=offset)).isoformat()

    month_match = _NEXT_MONTH_ORDINAL_RE.match(normalized)
    if month_match:
        day = int(month_match.group(1))
        year, month = _shift_month(call_date, 1)
        last_day_of_month = calendar.monthrange(year, month)[1]
        return date(year, month, min(day, last_day_of_month)).isoformat()

    return None

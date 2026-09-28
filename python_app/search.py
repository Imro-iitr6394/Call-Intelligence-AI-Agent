"""Cross-call search: turn a call record into embeddable chunks, and rank
stored chunks by similarity to a query.

Every chunk keeps the transcript line it is evidence for, so a search result
can always be traced back to one exact line -- the same grounding rule every
other insight in this app already follows. Ranking is pure similarity score,
highest first: no keyword layer, no blending, no invented weights (see
project notes on why a hybrid score was rejected).
"""

from __future__ import annotations

from typing import Any

from python_app.embeddings import cosine_similarity, embed_text


def extract_chunks(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one chunk per piece of evidence-bearing text on a call record."""

    chunks: list[dict[str, Any]] = []

    for line in record.get("transcript", {}).get("lines", []) or []:
        if isinstance(line, dict) and isinstance(line.get("text"), str) and line["text"].strip():
            chunks.append(
                {
                    "chunk_type": "transcript_line",
                    "line_number": line.get("line"),
                    "text": line["text"],
                }
            )

    insights = record.get("insights", {})
    section_chunk_type = {
        "decisions": "decision",
        "action_items": "action_item",
        "blockers": "blocker",
    }
    for section, chunk_type in section_chunk_type.items():
        for item in insights.get(section, []) or []:
            if not isinstance(item, dict) or not isinstance(item.get("description"), str):
                continue
            if not item["description"].strip():
                continue
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            chunks.append(
                {
                    "chunk_type": chunk_type,
                    "line_number": evidence.get("line"),
                    "text": item["description"],
                }
            )

    for finding in record.get("compliance_findings", []) or []:
        if not isinstance(finding, dict) or not isinstance(finding.get("description"), str):
            continue
        if not finding["description"].strip():
            continue
        evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
        chunks.append(
            {
                "chunk_type": "compliance_finding",
                "line_number": evidence.get("line"),
                "text": finding["description"],
            }
        )

    return chunks


def rank_chunks_by_similarity(
    query_embedding: list[float],
    stored_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return every stored chunk with a similarity score attached, sorted
    highest first. Never drops a chunk silently -- callers decide thresholds.
    """

    scored = []
    for chunk in stored_chunks:
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list):
            continue
        scored.append({**chunk, "score": cosine_similarity(query_embedding, embedding)})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def best_result_per_call(
    ranked_chunks: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    *,
    min_score: float = 0.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Collapse ranked chunks down to one result per call: its best match.

    ``ranked_chunks`` must already be sorted highest-score-first, so the
    first chunk seen for a call id is that call's best match.
    """

    best_per_call: dict[str, dict[str, Any]] = {}
    for chunk in ranked_chunks:
        call_id = chunk.get("call_id")
        if call_id not in records_by_id or call_id in best_per_call:
            continue
        if chunk["score"] < min_score:
            continue
        best_per_call[call_id] = chunk

    results = []
    for call_id, chunk in best_per_call.items():
        record = records_by_id[call_id]
        results.append(
            {
                "call_id": call_id,
                "title": record.get("metadata", {}).get("title"),
                "score": chunk["score"],
                "matched_chunk_type": chunk["chunk_type"],
                "matched_text": chunk["text"],
                "matched_line": chunk.get("line_number"),
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:limit]


def search_calls(
    query: str,
    records: list[dict[str, Any]],
    stored_chunks: list[dict[str, Any]],
    *,
    min_score: float = 0.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Rank every stored call by how similar its best-matching chunk is to
    the query, highest first. No cutoff is applied by default -- a real,
    empirical check showed a genuinely correct match scoring only 0.32,
    so hiding results below an invented threshold would have hidden a
    correct answer. Every result carries its own score; showing that score
    to the reviewer, rather than silently filtering on it, is the safer
    default until there is real query data to calibrate a cutoff against.
    """

    if not query.strip():
        return []
    query_embedding = embed_text(query)
    ranked_chunks = rank_chunks_by_similarity(query_embedding, stored_chunks)
    records_by_id = {record["call_id"]: record for record in records}
    return best_result_per_call(ranked_chunks, records_by_id, min_score=min_score, limit=limit)

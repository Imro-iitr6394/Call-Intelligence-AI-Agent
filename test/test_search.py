"""Tests for chunk extraction and similarity ranking.

These tests never load the real embedding model -- they use hand-written
vectors so the ranking logic itself is verified quickly and deterministically.
The one test that needs the real model lives in test_search_live.py.
"""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from python_app.embeddings import cosine_similarity
from python_app.intake import create_pending_call_record
from python_app.search import best_result_per_call, extract_chunks, rank_chunks_by_similarity
from python_app.storage import LocalCallStore


def test_extract_chunks_covers_every_evidence_bearing_field():
    record = {
        "transcript": {
            "lines": [
                {"line": 1, "text": "Agent: Hello."},
                {"line": 2, "text": "Consumer: Please stop calling me."},
            ]
        },
        "insights": {
            "decisions": [{"description": "Payment plan agreed.", "evidence": {"line": 1}}],
            "action_items": [{"description": "Follow up Friday.", "evidence": {"line": 2}}],
            "blockers": [{"description": "Needs supervisor approval.", "evidence": {"line": 2}}],
        },
        "compliance_findings": [
            {
                "type": "contact_restriction",
                "description": "The speaker requested that contact stop.",
                "evidence": {"line": 2},
            }
        ],
    }

    chunks = extract_chunks(record)
    chunk_types = {chunk["chunk_type"] for chunk in chunks}

    assert chunk_types == {
        "transcript_line",
        "decision",
        "action_item",
        "blocker",
        "compliance_finding",
    }
    assert len(chunks) == 6  # 2 transcript lines + 4 insight/finding chunks
    transcript_chunk = next(c for c in chunks if c["chunk_type"] == "transcript_line" and c["line_number"] == 2)
    assert transcript_chunk["text"] == "Consumer: Please stop calling me."


def test_extract_chunks_skips_empty_or_missing_descriptions():
    record = {
        "transcript": {"lines": []},
        "insights": {
            "decisions": [{"description": "", "evidence": {"line": 1}}],
            "action_items": [{"evidence": {"line": 1}}],
            "blockers": [],
        },
        "compliance_findings": [],
    }

    assert extract_chunks(record) == []


def test_rank_chunks_by_similarity_orders_highest_score_first():
    query_embedding = [1.0, 0.0]
    stored_chunks = [
        {"call_id": "CALL-0001", "chunk_type": "transcript_line", "text": "unrelated", "embedding": [0.0, 1.0]},
        {"call_id": "CALL-0002", "chunk_type": "transcript_line", "text": "exact match", "embedding": [1.0, 0.0]},
        {"call_id": "CALL-0003", "chunk_type": "transcript_line", "text": "somewhat related", "embedding": [0.7, 0.7]},
    ]

    ranked = rank_chunks_by_similarity(query_embedding, stored_chunks)

    assert [chunk["call_id"] for chunk in ranked] == ["CALL-0002", "CALL-0003", "CALL-0001"]
    assert ranked[0]["score"] == 1.0


def test_best_result_per_call_keeps_only_the_strongest_chunk_per_call():
    records_by_id = {
        "CALL-0001": {"call_id": "CALL-0001", "metadata": {"title": "call one"}},
        "CALL-0002": {"call_id": "CALL-0002", "metadata": {"title": "call two"}},
    }
    # Already sorted highest score first, as rank_chunks_by_similarity would produce.
    ranked_chunks = [
        {"call_id": "CALL-0001", "chunk_type": "compliance_finding", "text": "best match", "line_number": 3, "score": 0.9},
        {"call_id": "CALL-0001", "chunk_type": "transcript_line", "text": "weaker match, same call", "line_number": 1, "score": 0.4},
        {"call_id": "CALL-0002", "chunk_type": "transcript_line", "text": "only match", "line_number": 2, "score": 0.6},
    ]

    results = best_result_per_call(ranked_chunks, records_by_id, min_score=0.35)

    assert [r["call_id"] for r in results] == ["CALL-0001", "CALL-0002"]
    assert results[0]["matched_text"] == "best match"  # not the weaker chunk from the same call


def test_best_result_per_call_drops_results_below_the_minimum_score():
    records_by_id = {"CALL-0001": {"call_id": "CALL-0001", "metadata": {}}}
    ranked_chunks = [
        {"call_id": "CALL-0001", "chunk_type": "transcript_line", "text": "weak", "line_number": 1, "score": 0.1},
    ]

    results = best_result_per_call(ranked_chunks, records_by_id, min_score=0.35)

    assert results == []


def test_cosine_similarity_handles_mismatched_or_empty_vectors_safely():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_storage_saves_lists_and_cascade_deletes_chunks():
    with TemporaryDirectory(dir=Path.cwd(), prefix=".test-call-storage-") as temp_dir:
        store = LocalCallStore(
            Path(temp_dir) / "data",
            now_provider=lambda: datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc),
        )
        record = create_pending_call_record(
            {"name": "call.txt", "type": "text/plain", "size": 5},
            "CALL-0001",
            now=datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc),
        )
        store.save_call(record, source_bytes=b"hello")

        store.save_chunks(
            "CALL-0001",
            [
                {"chunk_type": "transcript_line", "line_number": 1, "text": "Hello.", "embedding": [1.0, 0.0]},
                {"chunk_type": "transcript_line", "line_number": 2, "text": "Bye.", "embedding": [0.0, 1.0]},
            ],
        )

        chunks = store.list_all_chunks()
        assert len(chunks) == 2
        assert {c["text"] for c in chunks} == {"Hello.", "Bye."}
        assert chunks[0]["embedding"] == [1.0, 0.0] or chunks[1]["embedding"] == [1.0, 0.0]

        # Saving chunks again for the same call replaces the old set, not append.
        store.save_chunks(
            "CALL-0001",
            [{"chunk_type": "transcript_line", "line_number": 1, "text": "Hello again.", "embedding": [1.0, 0.0]}],
        )
        assert len(store.list_all_chunks()) == 1

        store.delete_call("CALL-0001")
        assert store.list_all_chunks() == []

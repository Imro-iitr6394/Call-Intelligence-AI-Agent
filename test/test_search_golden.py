"""Golden dataset for cross-call search ranking.

Six distinct calls, six queries. Each query is worded differently from its
target call's own text (testing semantic search, not keyword luck), and must
rank its target call **first out of all six** stored calls -- not just beat one
hand-picked distractor, the way the earlier ad hoc QA check did.

The dialogue text below is word-for-word the same as the scripts used to
generate the real audio files in ``.qa_run/inputs/`` (via
``demo/generate_demo_audio.py``), so this test checks the same content a human
uploading those audio files through the real UI would be searching over --
just without needing a live AssemblyAI call on every test run. This uses the
real local embedding model, not a mock; only the transcription step is
skipped, since search operates on the transcript text either way.
"""

from python_app.embeddings import embed_text
from python_app.search import best_result_per_call, extract_chunks, rank_chunks_by_similarity

# call key -> "Speaker: text" lines, exactly matching the .qa_run/inputs/*.txt
# scripts used to generate the matching golden audio files.
GOLDEN_CALLS = {
    "payment_proposal": [
        "Agent: Hello, this is Priya calling from AstraGlobal on a recorded line. Am I speaking with Robert?",
        "Consumer: Yes, this is Robert.",
        "Agent: I'm calling about your outstanding balance of nine hundred dollars. Are you able to make a payment today?",
        "Consumer: I can't pay all of it today. Could I pay four hundred fifty now and four hundred fifty on the twentieth of next month?",
        "Agent: I'll note that, four hundred fifty today and four hundred fifty on the twentieth of next month. I'll need my supervisor to approve this split, and I'll call you back by Friday with confirmation.",
        "Consumer: Okay, that works for me.",
    ],
    "cease_and_desist": [
        "Agent: Hello, this is calling about your account balance.",
        "Consumer: I've asked you before, please take me off your calling list and stop contacting me.",
    ],
    "legal_escalation": [
        "Agent: I'm calling about your outstanding balance.",
        "Consumer: If this continues, I will have my attorney look into this and consider legal action.",
    ],
    "wrong_number": [
        "Agent: Hello, I'm trying to reach Sarah Mitchell about an account.",
        "Consumer: There's no Sarah here, you have the wrong number.",
    ],
    "angry_customer": [
        "Agent: I understand your frustration, let's see how we can resolve this.",
        "Consumer: This is absolutely ridiculous, I am furious about these constant calls!",
    ],
    "financial_hardship": [
        "Agent: I understand this is a difficult time for you.",
        "Consumer: I lost my job last month and I simply cannot afford this right now.",
    ],
}

# query -> the call key it must rank #1, out of all six calls.
GOLDEN_QUERIES = {
    "customer wants no further contact": "cease_and_desist",
    "someone is threatening to get a lawyer involved": "legal_escalation",
    "nobody by that name lives at this number": "wrong_number",
    "the caller sounds extremely upset": "angry_customer",
    "customer recently became unemployed and can't pay": "financial_hardship",
    "consumer proposes splitting payment across two dates": "payment_proposal",
}


def _build_records_and_chunks():
    records = []
    chunks = []
    call_id_to_key = {}
    for index, (call_key, lines) in enumerate(GOLDEN_CALLS.items(), start=1):
        call_id = f"CALL-GOLDEN-{index:02d}"
        call_id_to_key[call_id] = call_key
        transcript_lines = []
        for line_number, raw_line in enumerate(lines, start=1):
            speaker, _, text = raw_line.partition(": ")
            transcript_lines.append({"line": line_number, "speaker": speaker, "text": text})
        record = {
            "call_id": call_id,
            "metadata": {"title": call_key},
            "transcript": {"lines": transcript_lines},
            "insights": {},
            "compliance_findings": [],
        }
        records.append(record)
        for chunk in extract_chunks(record):
            chunks.append({**chunk, "call_id": call_id, "embedding": embed_text(chunk["text"])})
    return records, chunks, call_id_to_key


def test_golden_search_dataset_ranks_the_relevant_call_first_for_every_query():
    """Real, empirical proof of ranking quality across a realistic pool of
    calls -- not just a two-call comparison. Reports every failing query at
    once (not just the first) so a regression is fully diagnosable in one run.
    """

    records, chunks, call_id_to_key = _build_records_and_chunks()
    records_by_id = {record["call_id"]: record for record in records}

    failures = []
    for query, expected_key in GOLDEN_QUERIES.items():
        query_embedding = embed_text(query)
        ranked_chunks = rank_chunks_by_similarity(query_embedding, chunks)
        results = best_result_per_call(ranked_chunks, records_by_id, limit=len(records))
        if not results:
            failures.append(f"{query!r}: expected {expected_key!r} first, got no results at all")
            continue
        top_key = call_id_to_key[results[0]["call_id"]]
        if top_key != expected_key:
            scores = ", ".join(f"{call_id_to_key[r['call_id']]}={r['score']:.3f}" for r in results)
            failures.append(
                f"{query!r}: expected {expected_key!r} to rank first, got {top_key!r}. Scores: {scores}"
            )

    assert not failures, "\n" + "\n".join(failures)

"""Live tests against the real embedding model.

Unlike test_search.py, these tests load the actual all-MiniLM-L6-v2 model
(downloaded once, then cached locally -- no network needed after that) and
prove real behavior on real sentences, rather than hand-written vectors.

One of these tests is marked ``xfail``: it documents a real, measured
weakness of pure similarity search (see limitations.md) rather than hiding
it. If a future model change makes it start passing, pytest will flag that
as an unexpected pass, which is the signal that the limitation may be
resolved.
"""

import pytest

from python_app.embeddings import cosine_similarity, embed_text


def test_similarity_search_finds_a_semantically_different_phrasing():
    """A genuinely correct match, worded completely differently from the source line."""

    query = "customer does not want any more calls"
    cease_and_desist_line = "Take me off your list and stop calling me."
    unrelated_line = "To verify your identity, can you confirm your details?"

    query_embedding = embed_text(query)
    relevant_score = cosine_similarity(query_embedding, embed_text(cease_and_desist_line))
    unrelated_score = cosine_similarity(query_embedding, embed_text(unrelated_line))

    assert relevant_score > unrelated_score
    assert relevant_score > 0.2  # meaningfully positive, not a coincidence


@pytest.mark.xfail(
    reason=(
        "Known limitation (see limitations.md): MiniLM leans on shared "
        "content words ('file', 'complaint') more than the negation 'not', "
        "so a negated statement can outscore a genuinely relevant but "
        "differently-worded one. Measured: negated text ~0.67 vs. the "
        "actually-relevant text ~0.39 for this exact query."
    ),
    strict=True,
)
def test_similarity_search_is_not_fooled_by_negation():
    """The case that motivated dropping keyword search -- does pure similarity fix it?"""

    query = "who can file a complaint"
    negated_text = "I will not file a complaint against you"
    active_threat_text = "I will take you to the court"

    query_embedding = embed_text(query)
    negated_score = cosine_similarity(query_embedding, embed_text(negated_text))
    threat_score = cosine_similarity(query_embedding, embed_text(active_threat_text))

    # This is the desired behavior. It currently does not hold -- see the
    # xfail reason above and limitations.md for the measured numbers.
    assert threat_score > negated_score

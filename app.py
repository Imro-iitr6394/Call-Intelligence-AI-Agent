"""Streamlit entry point for the Python Call Intelligence prototype."""

from __future__ import annotations

import html
import logging
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from python_app.intake import prepare_call_intake
from python_app.embeddings import embed_text
from python_app.extraction import (
    apply_extraction_to_record,
    create_extraction_model,
    extract_call_intelligence,
)
from python_app.observability import observe, status_text
from python_app.review import review_item_id, visible_compliance_findings
from python_app.risk_rules import overall_compliance_status
from python_app.search import extract_chunks
from python_app.storage import LocalCallStore
from python_app.transcription import TranscriptionError, create_transcription_provider
from python_app.transcripts import TranscriptError, apply_normalized_transcript, normalize_transcript


LOGGER = logging.getLogger("astra.search")
BASE_DIR = Path(__file__).parent / "data"

# (background, text) colors for each compliance status/severity level.
SEVERITY_COLORS = {
    "Red": ("#fee2e2", "#7f1d1d"),
    "Yellow": ("#fff7d6", "#7a4d00"),
    "Green": ("#dcfce7", "#14532d"),
}


@st.cache_resource
def get_store() -> LocalCallStore:
    return LocalCallStore(BASE_DIR)


def file_signature(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}:{uploaded_file.type}"


def _save_search_chunks(record: dict, store: LocalCallStore) -> None:
    """Embed and store this call's searchable chunks.

    Best-effort only: search indexing must never block ingestion or
    extraction. If the local embedding model can't be loaded, the call is
    still saved normally -- it just won't be findable by search yet.
    """

    try:
        chunks = extract_chunks(record)
        embedded_chunks = [{**chunk, "embedding": embed_text(chunk["text"])} for chunk in chunks]
        store.save_chunks(record["call_id"], embedded_chunks)
    except Exception as error:  # pragma: no cover - depends on local model availability
        LOGGER.warning("Search indexing skipped for %s: %s", record.get("call_id"), type(error).__name__)


def save_transcript_analysis(
    record: dict,
    transcript: dict,
    store: LocalCallStore,
) -> tuple[dict, dict]:
    """Normalize a transcript and run Gemini plus deterministic validation."""

    with observe("transcript_extraction", call_id=record["call_id"]):
        normalized_record = apply_normalized_transcript(record, transcript)
        model = create_extraction_model()
        result = extract_call_intelligence(
            transcript,
            model=model,
            call_date=record.get("metadata", {}).get("call_date"),
        )
        analyzed_record = apply_extraction_to_record(normalized_record, result)
        store.save_call(analyzed_record)
        _save_search_chunks(analyzed_record, store)
    return analyzed_record, result


def _process_transcript_upload(
    record: dict,
    source_bytes: bytes,
    stored_record: dict,
    store: LocalCallStore,
) -> None:
    try:
        with observe(
            "transcript_normalization",
            call_id=record["call_id"],
            input_type=record["source"]["input_type"],
        ):
            transcript = normalize_transcript(record["source"]["file_name"], source_bytes)
            stored_record, extraction_result = save_transcript_analysis(
                stored_record, transcript, store
            )
        if extraction_result["review"]["required"]:
            st.warning(f"Stored {record['call_id']} and routed it to review.")
        elif transcript["normalization"]["warnings"]:
            st.warning(f"Stored {record['call_id']} with transcript review warnings.")
        else:
            st.success(
                f"Stored and analyzed {record['call_id']}: "
                f"{record['source']['file_name']}"
            )
    except TranscriptError as error:
        store.update_call(
            record["call_id"],
            {"processing": {"status": "failed", "stage": "failed", "error": str(error)}},
        )
        st.error(f"{record['source']['file_name']}: {error}")


def _process_audio_upload(
    record: dict,
    source_bytes: bytes,
    stored_record: dict,
    store: LocalCallStore,
) -> None:
    try:
        with observe(
            "audio_transcription",
            call_id=record["call_id"],
            input_type=record["source"]["input_type"],
            size_bytes=record["source"]["size_bytes"],
        ):
            store.update_call(
                record["call_id"],
                {"processing": {"status": "processing", "stage": "transcribing", "error": None}},
            )
            provider = create_transcription_provider()
            transcript = provider.transcribe(
                source_bytes,
                record["source"]["file_name"],
                record["source"]["mime_type"],
            )
            stored_record, extraction_result = save_transcript_analysis(
                stored_record, transcript, store
            )
        if extraction_result["review"]["required"]:
            st.warning(f"Transcribed {record['call_id']} and routed it to review.")
        elif transcript["normalization"]["warnings"]:
            st.warning(f"Transcribed {record['call_id']} with review warnings.")
        else:
            st.success(
                f"Transcribed and analyzed {record['call_id']}: "
                f"{record['source']['file_name']}"
            )
    except TranscriptionError as error:
        store.update_call(
            record["call_id"],
            {
                "processing": {
                    "status": "failed",
                    "stage": "failed",
                    "error": str(error),
                }
            },
        )
        st.error(f"{record['source']['file_name']}: {error}")


def process_uploads(uploaded_files, store: LocalCallStore) -> None:
    with observe("call_batch_upload", batch_size=len(uploaded_files)):
        existing_calls = store.list_calls()
        records, errors = prepare_call_intake(uploaded_files, existing_calls=existing_calls)
        uploads_by_name = {}
        # Intake keeps only the filename, so duplicate names are consumed in upload order.
        for uploaded_file in uploaded_files:
            uploads_by_name.setdefault(uploaded_file.name, []).append(uploaded_file)

        for record in records:
            input_type = record["source"]["input_type"]
            with observe(
                "call_ingestion",
                call_id=record["call_id"],
                input_type=input_type,
                size_bytes=record["source"]["size_bytes"],
            ):
                uploaded_file = uploads_by_name[record["source"]["file_name"]].pop(0)
                source_bytes = uploaded_file.getvalue()
                stored_record = store.save_call(record, source_bytes=source_bytes)

                if input_type == "transcript":
                    _process_transcript_upload(record, source_bytes, stored_record, store)
                else:
                    _process_audio_upload(record, source_bytes, stored_record, store)

        for error in errors:
            st.error(f"{error['file_name'] or 'Upload'}: {error['message']}")


def evidence_link(
    evidence: dict | None,
    label: str | None = None,
    *,
    call_id: str | None = None,
) -> str:
    """Return a link that selects and highlights the cited transcript line."""

    line_number = evidence.get("line") if isinstance(evidence, dict) else None
    if not isinstance(line_number, int):
        return "Evidence line unavailable"
    link_label = label or f"View ground-truth line {line_number}"
    call_query = f"?call_id={quote(str(call_id or ''), safe='')}&evidence_line={line_number}"
    return (
        f'<a href="{call_query}#transcript-line-{line_number}" target="_self">'
        f"{html.escape(link_label)}</a>"
    )


def _format_seconds(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}s"
    return "time unavailable"


def render_ground_truth_transcript(transcript: dict, focused_line: int | None = None) -> None:
    """Render transcript lines as stable browser anchors for evidence navigation."""

    lines = transcript.get("lines", []) if isinstance(transcript, dict) else []
    if not isinstance(lines, list) or not lines:
        return

    st.markdown("---")
    st.subheader("Ground-truth transcript")
    st.caption("Evidence links above navigate to the exact transcript line used for that insight.")
    if focused_line is not None:
        st.info(f"Evidence focus: transcript line {focused_line} is highlighted below.")
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_number = line.get("line")
        if not isinstance(line_number, int):
            continue
        speaker = html.escape(str(line.get("speaker") or "Unknown speaker"))
        text = html.escape(str(line.get("text") or ""))
        time_range = (
            f"{_format_seconds(line.get('start_seconds'))} - "
            f"{_format_seconds(line.get('end_seconds'))}"
        )
        is_focused = line_number == focused_line
        border_color = "#dc2626" if is_focused else "#d1d5db"
        background_color = "#fff3a3" if is_focused else "#f8fafc"
        focus_label = " <strong>← selected evidence</strong>" if is_focused else ""
        st.markdown(
            f'<div id="transcript-line-{line_number}" '
            'style="padding:0.7rem 0.9rem; margin:0.45rem 0; '
            f'border-left:6px solid {border_color}; background:{background_color};">'
            f'<b>Line {line_number}</b> · {speaker} · {html.escape(time_range)}{focus_label}<br>'
            f"{text}</div>",
            unsafe_allow_html=True,
        )


# A review item that mentions any of these is treated as a legal/cease-and-desist
# matter for priority ordering, even if its own severity is not Red.
_LEGAL_OR_CEASE_TYPES = {"legal_escalation", "attorney_representation", "contact_restriction"}


def _review_item_priority(item: dict) -> tuple[int, str]:
    """Sort key matching the spec's stated review order.

    Red findings first, then legal/cease-and-desist, then low-confidence
    (validation failures), then Yellow, then everything else.
    """

    item_type = str(item.get("type") or "")
    severity = item.get("severity")
    if severity == "Red":
        rank = 0
    elif item_type in _LEGAL_OR_CEASE_TYPES:
        rank = 1
    elif item_type == "Extraction validation failure":
        rank = 2
    elif severity == "Yellow":
        rank = 3
    else:
        rank = 4
    return (rank, item_type)


def render_review_queue(record: dict, store: LocalCallStore) -> None:
    """Let a reviewer approve, correct, reject, or resolve each flagged item."""

    call_id = record["call_id"]
    items = [item for item in record.get("review", {}).get("items", []) if isinstance(item, dict)]
    if not items:
        st.info("Nothing is waiting for review on this call.")
        return

    open_items = [item for item in items if item.get("resolution") is None]
    resolved_items = [item for item in items if item.get("resolution") is not None]

    if open_items:
        st.caption(
            "Priority order: Red findings first, then legal/cease-and-desist, "
            "then low-confidence, then Yellow, then everything else."
        )
        for item in sorted(open_items, key=_review_item_priority):
            _render_review_item(item, call_id, store, is_open=True)
    else:
        st.success("Every flagged item on this call has been reviewed.")

    if resolved_items:
        with st.expander(f"Already resolved ({len(resolved_items)})"):
            for item in resolved_items:
                _render_review_item(item, call_id, store, is_open=False)


def _render_review_item(item: dict, call_id: str, store: LocalCallStore, *, is_open: bool) -> None:
    # Records saved before this ticket system existed may not have a stored
    # id yet -- fall back to computing one from the item's own content so
    # buttons still get a stable, unique key instead of colliding.
    item_id = str(item.get("id") or review_item_id(item))
    item_type = html.escape(str(item.get("type") or "Review item"))
    detail = html.escape(str(item.get("detail") or ""))
    severity = item.get("severity")

    badge = ""
    if severity in SEVERITY_COLORS:
        background, text_color = SEVERITY_COLORS[severity]
        badge = (
            f'<span style="background:{background}; color:{text_color}; '
            'padding:0.1rem 0.5rem; border-radius:0.3rem; font-size:0.85rem; margin-right:0.5rem;">'
            f"{html.escape(str(severity))}</span>"
        )
    st.markdown(f"{badge}<b>{item_type}</b> — {detail}", unsafe_allow_html=True)

    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        st.markdown(
            evidence_link(evidence, "View transcript evidence", call_id=call_id),
            unsafe_allow_html=True,
        )

    if not is_open:
        note = f" — {html.escape(str(item['resolution_note']))}" if item.get("resolution_note") else ""
        st.caption(f"Resolved: {item.get('resolution')}{note}")
        st.markdown("---")
        return

    columns = st.columns(4)
    if columns[0].button("Approve", key=f"approve-{item_id}"):
        store.update_review_item(call_id, item_id, "approved")
        st.rerun()
    correction_text = columns[1].text_input(
        "Correction", key=f"correction-text-{item_id}", label_visibility="collapsed",
        placeholder="Replacement text...",
    )
    if columns[1].button("Correct", key=f"correct-{item_id}"):
        if correction_text.strip():
            store.update_review_item(call_id, item_id, "corrected", note=correction_text.strip())
            st.rerun()
        else:
            st.warning("Type a replacement description before saving a correction.")
    if columns[2].button("Reject", key=f"reject-{item_id}"):
        store.update_review_item(call_id, item_id, "rejected")
        st.rerun()
    if columns[3].button("Resolve", key=f"resolve-{item_id}"):
        store.update_review_item(call_id, item_id, "resolved")
        st.rerun()
    st.markdown("---")


def render_call(record: dict, store: LocalCallStore, focused_line: int | None = None) -> None:
    call_id = record["call_id"]
    processing = record.get("processing", {})
    status = processing.get("status")
    stage = processing.get("stage")
    if status == "transcript_ready" and stage == "extracting":
        title = f"{record['source']['file_name']} · transcript ready"
    elif status == "completed" and stage == "review":
        title = f"{record['source']['file_name']} · review required"
    elif status == "completed":
        title = f"{record['source']['file_name']} · analysis complete"
    else:
        title = record["metadata"]["title"]
    st.subheader(title)
    st.caption(
        f"{call_id} · {record['metadata']['call_date']} · "
        f"{record['processing']['status']} · {record['source']['input_type']}"
    )

    review_items = record.get("review", {}).get("items", [])
    visible_findings = visible_compliance_findings(record.get("compliance_findings", []), review_items)
    overall_status = overall_compliance_status(visible_findings)
    badge_background, badge_text = SEVERITY_COLORS[overall_status]
    st.markdown(
        '<span style="display:inline-block; padding:0.25rem 0.75rem; '
        f'border-radius:0.4rem; background:{badge_background}; color:{badge_text}; '
        f'font-weight:600;">Compliance: {overall_status}</span>',
        unsafe_allow_html=True,
    )

    report_tab, review_tab, source_tab, json_tab = st.tabs(["Report", "Review", "Source", "JSON"])
    with report_tab:
        if status == "failed":
            st.error(f"Processing failed: {processing.get('error') or 'Unknown error.'}")
        elif status == "processing":
            st.info(f"Call is currently being processed ({stage or 'unknown stage'}).")
        elif status == "transcript_ready" and stage == "extracting":
            st.info("Transcript is ready. The local risk scan is complete; AI extraction is not configured.")
        elif status == "completed" and stage == "review":
            st.warning("Analysis completed, but this call requires human review.")
        elif status == "completed":
            st.success("Analysis completed.")
        else:
            st.info("Call is waiting to be processed.")

        insights = record.get("insights", {})
        if insights.get("tag"):
            st.markdown(
                f'<span style="color:#000000; font-weight:700;">Tag: {html.escape(str(insights["tag"]))}</span>',
                unsafe_allow_html=True,
            )
        if insights.get("summary"):
            st.subheader("Summary")
            st.write(insights["summary"])

        sentiment = insights.get("sentiment", {})
        if any(sentiment.get(key) is not None for key in ("overall", "anger")) or sentiment.get("profanity"):
            st.subheader("Conversation signals")
            signal_columns = st.columns(3)
            signal_columns[0].metric("Overall", sentiment.get("overall") or "Unclear")
            signal_columns[1].metric("Anger", sentiment.get("anger") or "Unclear")
            signal_columns[2].metric("Profanity", len(sentiment.get("profanity") or []))

            anger_evidence = sentiment.get("anger_evidence") or []
            if anger_evidence:
                anger_links = " · ".join(
                    evidence_link(evidence, f"line {evidence.get('line')}", call_id=call_id)
                    for evidence in anger_evidence
                    if isinstance(evidence, dict)
                )
                st.caption("Anger evidence:")
                st.markdown(anger_links, unsafe_allow_html=True)

            profanity_items = sentiment.get("profanity") or []
            if profanity_items:
                profanity_links = " · ".join(
                    evidence_link(item.get("evidence"), f"“{item.get('word')}”", call_id=call_id)
                    for item in profanity_items
                    if isinstance(item, dict)
                )
                st.caption("Profanity evidence:")
                st.markdown(profanity_links, unsafe_allow_html=True)

        if insights.get("decisions"):
            st.subheader("Decisions")
            for item in insights["decisions"]:
                if isinstance(item, dict):
                    description = str(item.get("description", "Unclear decision"))
                    st.markdown(
                        f"- {evidence_link(item.get('evidence'), description, call_id=call_id)}",
                        unsafe_allow_html=True,
                    )

        if insights.get("action_items"):
            st.subheader("Action items")
            for item in insights["action_items"]:
                if isinstance(item, dict):
                    owner = item.get("owner") or "Unassigned"
                    due_date = item.get("due_date") or "No date"
                    due_date_phrase = item.get("due_date_phrase")
                    due_date_text = html.escape(str(due_date))
                    if due_date_phrase:
                        due_date_text += f" (said: “{html.escape(str(due_date_phrase))}”)"
                    description = str(item.get("description", "Unclear action"))
                    st.markdown(
                        f"- {evidence_link(item.get('evidence'), description, call_id=call_id)} "
                        f"(owner: {html.escape(str(owner))}; due: {due_date_text}) "
                        f"· {evidence_link(item.get('evidence'), call_id=call_id)}",
                        unsafe_allow_html=True,
                    )

        if insights.get("blockers"):
            st.subheader("Blockers")
            for item in insights["blockers"]:
                if isinstance(item, dict):
                    description = str(item.get("description", "Unclear blocker"))
                    st.markdown(
                        f"- {evidence_link(item.get('evidence'), description, call_id=call_id)}",
                        unsafe_allow_html=True,
                    )

        if insights.get("confidence"):
            st.caption(f"Model confidence: {insights['confidence']}")

        if visible_findings:
            st.subheader("Risk signals")
            for finding in visible_findings:
                finding_description = str(finding.get("description", "Review transcript evidence."))
                finding_label = (
                    f"{finding.get('type', 'risk')} · "
                    f"{finding.get('severity', 'review')} · "
                    f"{finding_description}"
                )
                finding_text = evidence_link(
                    finding.get("evidence"), finding_label, call_id=call_id
                )
                finding_background, finding_text_color = SEVERITY_COLORS.get(
                    finding.get("severity"), SEVERITY_COLORS["Yellow"]
                )
                st.markdown(
                    '<div style="padding:0.8rem 1rem; margin:0.5rem 0; '
                    f'border-radius:0.4rem; background:{finding_background}; color:{finding_text_color};">'
                    f"{finding_text}</div>",
                    unsafe_allow_html=True,
                )

        render_ground_truth_transcript(record.get("transcript", {}), focused_line=focused_line)

        has_empty_insights = not any(
            [
                insights.get("summary"),
                insights.get("decisions"),
                insights.get("action_items"),
                insights.get("blockers"),
                insights.get("sentiment", {}).get("overall"),
            ]
        )
        can_retry_analysis = status == "completed" and stage == "review" and has_empty_insights
        if (status == "transcript_ready" and stage == "normalizing") or can_retry_analysis:
            button_label = "Retry AI analysis" if can_retry_analysis else "Run AI analysis"
            button_key = f"retry-analysis-{call_id}" if can_retry_analysis else f"analysis-{call_id}"
            if st.button(button_label, key=button_key):
                transcript = record.get("transcript", {})
                analyzed_record, _ = save_transcript_analysis(record, transcript, store)
                st.rerun()

    with review_tab:
        render_review_queue(record, store)

    with source_tab:
        source_bytes = store.get_original_file(call_id)
        if source_bytes is not None:
            st.download_button(
                "Download original file",
                data=source_bytes,
                file_name=record["source"]["file_name"],
                mime=record["source"]["mime_type"] or "application/octet-stream",
            )
        else:
            st.warning("The original source file is not available.")

    with json_tab:
        st.json(record)

    if st.button("Delete call", type="secondary", key=f"delete-{call_id}"):
        with observe("call_deletion", call_id=call_id):
            store.delete_call(call_id)
        st.session_state.pop("selected_call_id", None)
        st.rerun()


def render_search_results(query: str, store: LocalCallStore) -> None:
    """Show every stored call ranked by similarity to the query, highest first.

    Every result carries its own score, shown next to it -- results are never
    silently hidden behind an invented cutoff (see limitations.md). Clicking
    a result jumps straight to the matching transcript line, reusing the same
    evidence-link mechanism the report already uses.
    """

    with st.spinner("Searching..."):
        try:
            results = store.search_calls(query)
        except Exception as error:  # pragma: no cover - depends on local model availability
            st.sidebar.warning(f"Search is unavailable right now ({type(error).__name__}).")
            return

    if not results:
        st.sidebar.info("No matches. Try different words -- this is a similarity search, not exact text.")
        return

    st.sidebar.caption(f"{len(results)} result(s), ranked by similarity (highest first).")
    for result in results:
        snippet = (result["matched_text"] or "")[:120]
        label = (
            f"{result['call_id']} · Match score: {result['score']:.2f} "
            f"— {result['matched_chunk_type']}: “{snippet}”"
        )
        if result.get("matched_line") is not None:
            st.sidebar.markdown(
                evidence_link({"line": result["matched_line"]}, label, call_id=result["call_id"]),
                unsafe_allow_html=True,
            )
        elif st.sidebar.button(label, key=f"search-result-{result['call_id']}"):
            st.session_state["selected_call_id"] = result["call_id"]
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Call Intelligence", layout="wide")
    st.title("Call Intelligence AI Agent")
    st.caption("Local Python prototype · evidence-backed debt-collection call processing")

    store = get_store()
    st.sidebar.header("Upload calls")
    st.sidebar.caption(f"Observability: {status_text()}")
    uploaded_files = st.sidebar.file_uploader(
        "Choose audio or transcript files",
        type=["aac", "flac", "m4a", "mp3", "mp4", "ogg", "wav", "webm", "wma", "json", "md", "txt"],
        accept_multiple_files=True,
    )

    processed_uploads = st.session_state.setdefault("processed_uploads", set())
    if uploaded_files:
        new_uploads = [
            uploaded_file
            for uploaded_file in uploaded_files
            if file_signature(uploaded_file) not in processed_uploads
        ]
        if new_uploads:
            process_uploads(new_uploads, store)
            processed_uploads.update(file_signature(uploaded_file) for uploaded_file in new_uploads)

    calls = store.list_calls()
    st.sidebar.metric("Stored calls", len(calls))
    if not calls:
        st.info("Upload an audio recording or transcript to create the first call record.")
        return

    call_ids = [record["call_id"] for record in calls]

    search_query = st.sidebar.text_input(
        "Search calls", placeholder="e.g. customer wants no more calls"
    )
    if search_query.strip():
        render_search_results(search_query.strip(), store)

    requested_call_id = st.query_params.get("call_id")
    if requested_call_id in call_ids:
        # Set widget state before the selectbox is instantiated. This lets an
        # evidence link open the correct call without the widget-state error.
        st.session_state["selected_call_id"] = requested_call_id
    selected_id = st.sidebar.selectbox("Select a call", call_ids, key="selected_call_id")
    selected = next(record for record in calls if record["call_id"] == selected_id)
    focused_line = None
    if requested_call_id == selected_id:
        raw_line = st.query_params.get("evidence_line")
        try:
            parsed_line = int(raw_line) if raw_line is not None else None
        except (TypeError, ValueError):
            parsed_line = None
        if parsed_line is not None and parsed_line > 0:
            focused_line = parsed_line
    render_call(selected, store, focused_line=focused_line)


if __name__ == "__main__":
    main()

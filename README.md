# Astra — Call Intelligence AI Agent

Astra turns a debt-collection call (audio or a typed transcript) into a
structured, evidence-backed report: a summary, decisions, action items,
blockers, sentiment, and compliance risk findings — plus a review queue for
anything the system isn't confident about. Every claim it makes points at
the exact transcript line it came from.

## What it does

1. **Ingest** an audio file or a text/JSON transcript.
2. **Transcribe** audio via AssemblyAI (speaker-labeled, diarized).
3. **Normalize** the transcript into numbered, speaker-tagged lines.
4. **Detect risk** with fixed, deterministic phrase rules (works even
   without an AI key) — legal escalation, contact restriction, debt
   dispute, financial hardship, wrong number, and more, each rated
   Red/Yellow/Green.
5. **Extract insights** with Gemini: summary, tag, decisions, action items,
   blockers, sentiment — every claim is checked against the transcript
   before being trusted; anything that fails validation is discarded and
   flagged for a human instead of shown as fact.
6. **Route to human review** anything uncertain: validation failures,
   unclear task owners, missing timestamps, findings a human hasn't
   confirmed yet.
7. **Search across calls** using semantic (meaning-based) search, so a
   query like "customer wants no more calls" finds "please stop calling
   me" even with no shared words.

## Tech stack

| Piece | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Storage | SQLite (local file, no server) |
| Transcription | [AssemblyAI](https://www.assemblyai.com) |
| Extraction (summary/insights) | Google Gemini |
| Semantic search | `sentence-transformers` (`all-MiniLM-L6-v2`), brute-force cosine similarity |
| Optional monitoring | Logfire, LangSmith |
| Tests | pytest, jsonschema |

## Getting started

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux
```

### 2. Configure

Create a `.env` file in the project root (never commit this file — it's
already in `.gitignore`):

```env
# Transcription (AssemblyAI)
TRANSCRIPTION_PROVIDER=assemblyai
TRANSCRIPTION_API_KEY=your_assemblyai_key

# Extraction (Gemini) — up to 3 keys, rotated automatically on quota errors (HTTP 429)
GEMINI_API_KEY=your_gemini_key
GEMINI_API_KEY_2=optional_backup_key
GEMINI_API_KEY_3=optional_backup_key

# Optional monitoring — safe to leave unset
OBSERVABILITY_ENABLED=false
LOGFIRE_TOKEN=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=
LANGSMITH_PROJECT=
```

The app still works with no Gemini key — you just get the deterministic
risk findings without an AI summary. Audio upload needs an AssemblyAI key;
typed transcripts don't.

### 3. Run

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a `.wav/.mp3/.m4a/...` audio file
or a `.txt/.md/.json` transcript from the sidebar.

### 4. Test

```bash
.venv\Scripts\python.exe -m pytest -q
```

## Project layout

```
app.py                     Streamlit UI and the pipeline orchestration
python_app/
  intake.py                 Validates uploads, creates the initial call record
  transcription.py          Talks to AssemblyAI, maps its response to our line format
  transcripts.py             Parses uploaded text/JSON transcripts into lines
  dates.py                   Resolves relative dates ("Friday", "next month") to real dates
  extraction.py              Builds the Gemini prompt, validates its answer, merges results
  risk_rules.py               Fixed phrase-based compliance risk detection
  review.py                  Review item ("ticket") helpers
  embeddings.py               Local text-to-vector model for search
  search.py                   Chunking and cosine-similarity ranking for cross-call search
  storage.py                  SQLite persistence (calls + search index)
  observability.py            Optional, privacy-safe monitoring
schemas/call-record.schema.json   The official shape every saved call record must match
test/                         pytest suite, including golden datasets
demo/generate_demo_audio.py   Generates test audio from a script (edge-tts)
data/                          SQLite DB + original uploaded files (gitignored)
```

## Design principles

- **Evidence or nothing.** Every AI claim must cite a real transcript line
  and quote words that actually appear in it, or it's discarded.
- **Fail closed.** When the AI's answer can't be verified, the app keeps
  the deterministic risk-rule results and routes the call to a human
  instead of showing an unverified answer.
- **Never invent.** Unclear task owners become `Unclear/Unassigned`
  instead of a guessed name; unresolvable dates are rejected rather than
  approximated.
- **Human decisions are final.** A reviewer's approve/correct/reject always
  overrides the automated finding, and survives re-running the analysis.

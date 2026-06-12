# Open-Bug Triage RAG

A retrieval system that lets support and engineering staff search an **open**
ticket backlog. It is built for triage — locating and grouping related open
work — rather than suggesting fixes.

The sample dataset is themed around a house-search / home-buying platform, but
you can drop in any tickets with the same shape.

## What it does

An LLM agent (OpenAI via LangChain) chooses among these tools:

| Tool | Purpose |
| --- | --- |
| `SearchSimilarTickets` | Find open tickets semantically similar to a described problem |
| `FindDuplicateTickets` | Flag likely duplicates of a description **or** an existing ticket ID |
| `SearchByPriority` | List all open tickets at a priority level (Critical/High/Medium/Low) |
| `SearchByCategory` | List all open tickets in a category |
| `GetTicketByID` | Full details of one ticket |
| `GetTicketStatistics` | Backlog overview by category and priority |

`SearchSimilarTickets` and `FindDuplicateTickets` are semantic (they call the
embedding API). The rest are plain in-memory filters.

## How embeddings work

- **At startup**, every ticket's `title + description + category + priority` is
  embedded once (`text-embedding-3-small`) and stored in an in-memory Chroma
  collection — see `_setup_vectorstore`.
- **At query time**, the two semantic tools embed the incoming query and run a
  nearest-neighbour search.

## Project layout

```
open-bug-triage-rag/
├── .env.example          # copy to .env and add your key
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── house_search_tickets_open.json   # ticket dataset
└── src/
    └── open_bug_triage_rag.py           # the app
```

## Setup

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure your API key
cp .env.example .env
#   then edit .env and set OPENAI_API_KEY
```

### Environment variables

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | yes | — | Used by LangChain for both chat and embeddings |
| `OPENAI_CHAT_MODEL` | no | `gpt-4o-mini` | Chat model for the triage agent |

## Run

```bash
python src/open_bug_triage_rag.py
```

This runs the built-in demo queries (similar search, duplicate detection by
description and by ticket ID, priority/category sweeps, lookup, and stats).

For an interactive prompt, uncomment the loop at the bottom of
`src/open_bug_triage_rag.py`.

## Data format

Each ticket is a JSON object:

```json
{
  "ticket_id": "TICK-001",
  "title": "Property search returns listings outside selected radius",
  "description": "Users searching within a 5-mile radius receiving results 20+ miles away...",
  "category": "Search",
  "priority": "High",
  "created_date": "2026-01-14",
  "resolved_date": "open"
}
```

## Tuning duplicate detection

`DUPLICATE_THRESHOLD` (top of `open_bug_triage_rag.py`, default `0.78`) controls
how similar two tickets must be to count as duplicates. Calibrate it against a
few known duplicate pairs in your real data. If your Chroma version lacks
`similarity_search_with_relevance_scores`, switch to `similarity_search_with_score`
and invert the comparison (lower distance = more similar).

## Notes

- The Chroma store is **in-memory** and rebuilt on each run. To persist it, pass
  `persist_directory=...` to `Chroma`; the `.gitignore` already excludes common
  Chroma data folders.
- Never commit your real `.env`. If a key is ever committed, rotate it.

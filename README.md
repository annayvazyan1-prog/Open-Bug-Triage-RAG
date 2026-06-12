# Open-Bug Triage RAG

A retrieval system that lets support and engineering staff search an **open**
ticket backlog — to find similar tickets, spot likely duplicates, and sweep by
priority or category. It's built for triage (locating and grouping open work),
not for proposing fixes.

The sample dataset is themed around a house-search / home-buying platform, but
you can drop in any tickets with the same shape.

## What it does

An LLM agent (OpenAI via LangChain) chooses among these tools:

| Tool | Purpose |
| --- | --- |
| `SearchSimilarTickets` | Find open tickets similar to a described problem |
| `FindDuplicateTickets` | Flag likely duplicates of a description **or** a ticket ID |
| `SearchByPriority` | List open tickets at a priority level |
| `SearchByCategory` | List open tickets in a category |
| `GetTicketByID` | Full details of one ticket |
| `GetTicketStatistics` | Backlog overview by category and priority |

`SearchSimilarTickets` and `FindDuplicateTickets` are semantic (they use
embeddings). The rest are plain in-memory filters.

## Project layout

```
open-bug-triage-rag/
├── pyproject.toml        # dependencies (managed by uv)
├── .env.example          # copy to .env and add your key
├── .gitignore
├── README.md
├── data/
│   └── house_search_tickets_open.json
└── src/
    ├── triage_core.py            # shared logic (class, agent loop)
    ├── open_bug_triage_rag.py    # command-line demo
    └── app.py                    # Streamlit web app
```

## Setup with uv

[uv](https://docs.astral.sh/uv/) is an all-in-one Python package/venv manager.

```bash
# install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
# or:  pip install uv

# from the project folder, install everything
uv sync

# add your API key
cp .env.example .env        # then edit .env and set OPENAI_API_KEY
```

`uv sync` reads `pyproject.toml`, creates a `.venv`, and writes a `uv.lock` for
reproducible installs. Commit `uv.lock`; don't commit `.venv` or `.env`.

### Environment variables

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | yes | — | Used for both chat and embeddings |
| `OPENAI_CHAT_MODEL` | no | `gpt-4o-mini` | Chat model for the agent |

## Run the Streamlit app

```bash
uv run streamlit run src/app.py
```

This opens a browser UI with tabs for: asking the agent in plain language,
finding similar tickets, checking for duplicates (with an adjustable similarity
cutoff in the sidebar), browsing by priority/category, and looking up a ticket.
If no `OPENAI_API_KEY` is set, the app lets you paste one for the session.

## Run the command-line demo

```bash
uv run python src/open_bug_triage_rag.py
```

## pip alternative (no uv)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs from pyproject.toml
cp .env.example .env
streamlit run src/app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. At https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Choose this repo, branch `main`, main file path `src/app.py`.
4. Under **Advanced settings → Secrets**, add your key in TOML form:
   ```toml
   OPENAI_API_KEY = "sk-your-real-key"
   ```
5. Deploy.

Streamlit Cloud installs from `requirements.txt` (it does not run `uv`). The app
reads the key from the environment, `st.secrets`, or a session input — in that
order — so the secret above is picked up automatically.

## Data format

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

`DUPLICATE_THRESHOLD` in `src/triage_core.py` (default `0.78`) sets how similar
two tickets must be to count as duplicates; the Streamlit sidebar can override
it per session. Calibrate against a few known duplicate pairs in your data. If
your Chroma version lacks `similarity_search_with_relevance_scores`, switch to
`similarity_search_with_score` and invert the comparison.

## Notes

- The Chroma store is in-memory and rebuilt on each run (the Streamlit app caches
  it across reruns). To persist it, pass `persist_directory=...` to `Chroma`.
- Never commit your real `.env`. If a key is ever committed, rotate it.

# JobScout

A job search agent that reads your CV first. Upload a CV, review the profile it builds, choose how jobs should be scored, and let a LangGraph agent search real openings, broaden its own query when results are thin, and rank the best matches with evidence. Every step is traced in Langfuse.


## Stack

| Layer | Choice |
|---|---|
| Agent | LangGraph (state machine with a self-correcting search loop and a human review step) |
| LLM | Gemini for profiles and search planning; Claude for scoring (Gemini fallback) |
| Embeddings | Gemini embeddings via `GoogleGenerativeAIEmbeddings` |
| Jobs | JSearch API |
| Tracing | Langfuse |
| Storage | Postgres (paused runs, presets, results), Redis (search cache, rate limits) |
| Web | Next.js on Vercel, FastAPI backend on Vercel Functions |

## Status

Phases 0-4 are done: CV intake with redaction, a fact-checked profile, a cached job search, a LangGraph agent that pauses for review and broadens its own search, and evidence-based scoring with editable presets. See [docs/phases.md](docs/phases.md) for the full plan and [docs/architecture.md](docs/architecture.md) for how the code is organised.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env            # add GOOGLE_API_KEY, and Langfuse keys to see traces
cd backend
uv sync
uv run pytest                   # unit tests, no keys needed
uv run jobscout cv path/to/cv.pdf        # see exactly what the LLM will receive
uv run jobscout profile path/to/cv.pdf   # build the profile with Gemini
```

Put real CVs in `samples/private/`. That folder is git-ignored.

## Repository layout

```
backend/                 Python: agent, modules, API (FastAPI from Phase 5)
  src/jobscout/
    cv/                  read_cv(): bytes -> safe, redacted text
    profile/             ProfileBuilder: text -> fact-checked CandidateProfile
    llm/                 structured_llm(): the only place that knows the providers
    jobs/                JobSearch: validated query -> real postings (JSearch + Redis cache)
    agent/               JobScoutAgent: LangGraph graph, pause/resume, search loop
    trust/               scam check: rules for every ad, Claude Haiku for uncertain ones
    scoring/             presets, embedding shortlist, evidence-checked scoring, model comparison
    storage/             shared Redis cache (searches, profiles, embeddings, scores)
    observability/       trace_config(): Langfuse wiring and masking
    cli.py               `jobscout` command for trying each phase
  tests/
frontend/                Next.js app (added in Phase 6)
docs/                    architecture and phase plan
samples/private/         your own CVs (not committed)
```

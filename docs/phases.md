# Build phases

Each phase ends with something you can run, see in Langfuse, and test. Stop at the end of each phase, try it on real CVs, and note what to change before starting the next one.

| Phase | What you build | Main thing you learn | Status |
|---|---|---|---|
| 0 | Repo, settings, LLM gateway, Langfuse wiring, CI | How LangChain runnables, fallbacks and callbacks fit together | Done |
| 1 | CV intake and fact-checked profile | Structured output, prompt-injection defence, keeping maths out of the LLM | Done |
| 2 | Job search tool with Redis cache | Wrapping an external API as a safe, cached tool | Done |
| 3 | The LangGraph agent | State, nodes, conditional edges, the rewrite loop, interrupt and resume | Done |
| 4 | Scoring presets and ranking | Embeddings, batched LLM judging, weights you can edit | Done |
| 5 | FastAPI backend | Two-request runs, streaming progress, per-user limits | Next |
| 6 | Next.js front end | Upload, review, weight editor, results | |
| 7 | Deploy and evaluate | Vercel, Neon, Upstash, Langfuse datasets | |

---

## Phase 0: Foundation (done)

**Built:** `backend/` Python project (uv), `settings.py`, `jobscout.llm` (Gemini Flash → Flash-Lite → OpenRouter fallback chain), `jobscout.observability` (Langfuse config with masking), ruff, pytest, GitHub Actions.

**Try it:** `cd backend && uv run pytest`

## Phase 1: CV to profile (done)

**Built:**
- `jobscout.cv.read_cv(bytes)`: detects PDF/DOCX by content, drops hidden text (white, tiny, Word-hidden, zero-width), normalizes Unicode, enforces size/page/length limits, and redacts name, email, phone, links, address and protected attributes.
- `jobscout.profile.ProfileBuilder().build(cv)`: one Gemini call that extracts only what is written, then code fact-checks every skill against the CV, parses dates, computes years of experience (overlaps counted once) and derives seniority.
- `jobscout` CLI with `cv` and `profile` commands.

**Try it:**
```bash
cd backend
uv run jobscout cv ../samples/private/my_cv.pdf        # what the LLM will see; no key needed
uv run jobscout profile ../samples/private/my_cv.pdf   # needs GOOGLE_API_KEY; traced if Langfuse keys are set
uv run pytest --live -k live                            # one real Gemini call
```

**Done when:** 90 unit tests pass; the `profile` command on 3–5 real CVs gives sensible skills; flagged skills really are missing from each CV; Langfuse shows a `build_profile` trace with the Gemini call nested inside.

**Feedback to give before Phase 2:**
- Did redaction miss anything personal, or remove something useful (a city, a date)?
- Were any real skills wrongly flagged? Add them to `ALIAS_GROUPS` in `profile/_factcheck.py`.
- Are years of experience and seniority right for each CV?

## Phase 2: Job search tool (done)

**Try it:** `uv run jobscout search "data analyst in Dubai" --country ae` (run it twice: the second answer comes from Redis).

**Goal:** `jobscout.jobs.JobSearch.search(query) -> SearchResult` hides the JSearch HTTP call, input validation, Redis cache (24 h), timeouts, retries with backoff and a per-run call budget.

**Learn:** turning an external API into a LangChain tool the agent can't misuse; cache keys; why the LLM never builds HTTP requests itself.

**Done when:** a repeated search is served from Redis (`cache_hit=true` in Langfuse); invalid queries are rejected before any network call; tests use recorded JSearch responses, not the live API.

## Phase 3: The LangGraph agent (done)

**Try it:**
```bash
uv run jobscout start ../samples/private/my_cv.pdf      # builds the profile, saves the run, pauses
uv run jobscout resume <run-id> --title "Data Analyst" --country ae --city Dubai
```
Paused runs are saved to Postgres when `DATABASE_URL` is set, otherwise to `backend/.jobscout/checkpoints.sqlite`.

**Goal:** `ingest → extract_profile → confirm_profile (interrupt) → plan_query → search_jobs → filter_dedupe → [rewrite_query ↺ | score]` with a Postgres checkpointer.

**Learn:** graph state, thin nodes calling deep modules, the conditional edge, `interrupt()` and `Command(resume=...)`, recursion limits, one Langfuse trace per graph run.

**Done when:** a rare job title triggers rewrites (visible in Langfuse), the loop stops after 3 attempts, and a paused run resumes from Postgres in a new process.

## Phase 4: Scoring presets and ranking (done)

**Try it:**
```bash
uv run jobscout resume <run-id> --title "Data Scientist" --country ae --city Dubai --preset student
uv run jobscout resume <run-id> ... --preset professional --weight skills=60 --weight domain=0
uv run jobscout compare <finished-run-id> --models claude-sonnet-5 claude-opus-5
```

**Goal:** editable presets (Professional, Student, custom criteria), Gemini-embedding shortlist, batched scoring (5 jobs per call) with quoted evidence, weighted totals in code, top 10.

**Done when:** changing weights changes the ranking; scores for criteria a job ad doesn't mention are "not applicable"; no job appears that JSearch didn't return.

## Phase 5: FastAPI backend

**Goal:** `POST /runs` (upload → profile), `POST /runs/{id}/resume` (streamed progress → results), `/rubrics` CRUD, per-user rate limits in Redis.

## Phase 6: Next.js front end

**Goal:** upload page, profile review with flagged-skill chips, weight sliders and custom criteria, live progress, result cards with ratings sent to Langfuse.

## Phase 7: Deploy and evaluate

**Goal:** two Vercel projects, Neon and Upstash from the Vercel Marketplace, invite code, a Langfuse dataset of CVs with expected profiles to test prompt changes.

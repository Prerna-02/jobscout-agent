# Architecture

## The rule: deep modules, thin glue

Each package under `backend/src/jobscout/` exposes a small interface from its `__init__.py` and keeps everything else in underscore-prefixed files. A caller should never need to know how a module does its job.

| Module | Interface | What it hides |
|---|---|---|
| `jobscout.cv` | `read_cv(bytes) -> CVDocument` | File-type detection by content, PDF/DOCX parsing, hidden-text removal, Unicode cleanup, size limits, redaction |
| `jobscout.profile` | `ProfileBuilder().build(cv) -> CandidateProfile` | Prompt, structured output, injection defence, skill fact-check with aliases, date parsing, years and seniority |
| `jobscout.llm` | `structured_llm(Schema) -> Runnable` | Providers, fallback order, retries, timeouts |
| `jobscout.observability` | `trace_config(...)`, `flush()` | Langfuse client, callback handler, masking, "off when no keys" |
| `jobscout.jobs` *(Phase 2)* | `JobSearch.search(query)` | JSearch HTTP, validation, Redis cache, retries, call budget |
| `jobscout.scoring` *(Phase 4)* | `score_jobs(profile, jobs, rubric)` | Embedding shortlist, batching, evidence checks, weighted totals |
| `jobscout.agent` *(Phase 3)* | `build_graph(checkpointer)` | LangGraph wiring only; each node is a few lines calling a module above |

LangGraph nodes stay thin on purpose. The graph shows *order and decisions*; the modules hold *behaviour*, and each module is tested without running the graph.

## Dependency direction

```
agent  ──>  profile, jobs, scoring  ──>  llm, cv  ──>  settings
   └──────────────────────────────────>  observability
```

Lower layers never import higher ones. `observability` only depends on `cv` for its masking function.

## What leaves the machine

| Destination | What is sent | What is never sent |
|---|---|---|
| Gemini / OpenRouter | Redacted CV text, job descriptions | Name, email, phone, links, address, date of birth, gender, nationality… |
| JSearch | A short search string built from confirmed skills and target titles | Anything from the CV beyond that |
| Langfuse | Traces of every step, masked again for emails, phones and links | The original file |

## Code, not the LLM, decides

| Decision | Made by |
|---|---|
| Is this skill really in the CV? | `SkillMatcher` (string matching with aliases) |
| Years of experience | `total_years` (date parsing, overlaps merged) |
| Seniority | `derive_seniority` (title keywords, then years) |
| Weighted match score *(Phase 4)* | Python arithmetic over the LLM's per-criterion scores |
| Which jobs exist *(Phase 4)* | Only IDs returned by JSearch; unknown IDs are dropped |

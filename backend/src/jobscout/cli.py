"""Command-line entry point for trying each phase without the web app.

uv run jobscout cv path/to/cv.pdf          # Phase 1a: what the LLM will see (no API key needed)
uv run jobscout profile path/to/cv.pdf     # Phase 1b: the fact-checked profile, traced in Langfuse
uv run jobscout search "data analyst in Dubai" --country ae   # Phase 2: real jobs, cached 24 h
uv run jobscout start path/to/cv.pdf       # Phase 3: agent builds the profile, then pauses
uv run jobscout resume <run-id> --title "Data Analyst" --country ae --city Dubai
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from jobscout.agent import FinishedRun, JobScoutAgent, PausedRun, RunNotFound, SearchPreferences
from jobscout.cv import CVDocument, CVRejected, read_cv
from jobscout.jobs import (
    DatePosted,
    FilterRules,
    JobSearch,
    JSearchError,
    JSearchNotConfigured,
    SearchQuery,
    SearchResult,
    filter_jobs,
)
from jobscout.llm import LLMNotConfigured
from jobscout.observability import flush, trace_config
from jobscout.profile import CandidateProfile, ProfileBuilder
from jobscout.scoring import (
    PRESETS,
    JobMatch,
    Rubric,
    agreement,
    build_rubric,
    compare_scorers,
)
from jobscout.scoring._scorer import MIN_COVERAGE
from jobscout.trust import TrustReport

EXIT_REJECTED = 2
EXIT_NOT_CONFIGURED = 3
EXIT_SEARCH_FAILED = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jobscout", description="JobScout agent tools")
    commands = parser.add_subparsers(dest="command", required=True)

    cv_cmd = commands.add_parser("cv", help="Show the redacted text that would be sent to the LLM")
    cv_cmd.add_argument("path", type=Path)

    profile_cmd = commands.add_parser("profile", help="Build a fact-checked profile from a CV")
    profile_cmd.add_argument("path", type=Path)
    profile_cmd.add_argument("--json", action="store_true", help="Print the profile as JSON")

    search_cmd = commands.add_parser("search", help="Search real job postings (cached for 24 h)")
    search_cmd.add_argument("text", help='e.g. "data analyst in Dubai"')
    search_cmd.add_argument("--country", default="us", help="two-letter code, e.g. ae")
    search_cmd.add_argument("--date", default="week", choices=get_args(DatePosted))
    search_cmd.add_argument("--remote", action="store_true", help="remote jobs only")
    search_cmd.add_argument("--json", action="store_true")

    start_cmd = commands.add_parser("start", help="Agent: build the profile, then pause for review")
    start_cmd.add_argument("path", type=Path)

    resume_cmd = commands.add_parser("resume", help="Agent: confirm preferences and search")
    resume_cmd.add_argument("run_id")
    resume_cmd.add_argument("--title", action="append", required=True, help="repeat for up to 3")
    resume_cmd.add_argument("--country", required=True, help="two-letter code, e.g. ae")
    resume_cmd.add_argument("--city")
    resume_cmd.add_argument("--remote", action="store_true")
    resume_cmd.add_argument("--date", default="week", choices=get_args(DatePosted))
    resume_cmd.add_argument("--skills", help="comma-separated; default: skills found in the CV")
    resume_cmd.add_argument("--preset", choices=sorted(PRESETS), help="default: fits the profile")
    resume_cmd.add_argument(
        "--weight", action="append", default=[], metavar="KEY=N", help="e.g. --weight skills=50"
    )
    resume_cmd.add_argument("--rubric", type=Path, help="JSON file with your own criteria")

    compare_cmd = commands.add_parser("compare", help="Score a finished run with several models")
    compare_cmd.add_argument("run_id")
    compare_cmd.add_argument(
        "--models", nargs="+", default=["claude-sonnet-5", "claude-opus-5"], metavar="MODEL"
    )

    args = parser.parse_args(argv)
    _utf8_stdout()
    if args.command == "search":
        return _run_search(args)
    if args.command == "start":
        return _run_start(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "resume":
        return _run_resume(args)

    try:
        cv = read_cv(args.path.read_bytes())
    except CVRejected as exc:
        print(f"CV rejected ({exc.reason}): {exc.message}", file=sys.stderr)
        return EXIT_REJECTED

    if args.command == "cv":
        _print_cv(cv)
        return 0

    try:
        profile = ProfileBuilder().build(
            cv,
            config=trace_config(run_name="build_profile", session_id=cv.fingerprint, tags=["cli"]),
        )
    except LLMNotConfigured as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    finally:
        flush()

    print(profile.model_dump_json(indent=2) if args.json else _format_profile(profile))
    return 0


def _run_start(args: argparse.Namespace) -> int:
    try:
        with JobScoutAgent.open() as agent:
            paused = agent.start(args.path.read_bytes())
    except CVRejected as exc:
        print(f"CV rejected ({exc.reason}): {exc.message}", file=sys.stderr)
        return EXIT_REJECTED
    except LLMNotConfigured as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    print(_format_paused(paused))
    return 0


def _run_resume(args: argparse.Namespace) -> int:
    skills = [s.strip() for s in args.skills.split(",")] if args.skills else None
    try:
        with JobScoutAgent.open() as agent:
            if skills is None:  # default to the skills the CV really contains
                skills = agent.saved_profile(args.run_id).confirmed_skills
            prefs = SearchPreferences(
                target_titles=args.title,
                country=args.country,
                city=args.city,
                remote_only=args.remote,
                date_posted=args.date,
                skills=skills,
                rubric=_rubric_from_args(args),
            )
            result = agent.resume(args.run_id, prefs)
    except ValidationError as exc:
        print(f"Invalid preferences: {exc.errors()[0]['msg']}", file=sys.stderr)
        return EXIT_REJECTED
    except RunNotFound as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REJECTED
    except (JSearchNotConfigured, LLMNotConfigured, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    print(_format_finished(result))
    return 0


def _format_paused(paused: PausedRun) -> str:
    s = paused.suggested
    titles = ", ".join(s["target_titles"]) or "(none in your CV: add at least one)"
    return "\n".join(
        [
            _format_profile(paused.profile),
            "",
            "=" * 60,
            f"Run {paused.run_id} is paused for your review.",
            f"Suggested job titles: {titles}",
            f"Suggested city: {s['city'] or '-'}",
            "",
            "Continue with your choices, for example:",
            f'  uv run jobscout resume {paused.run_id} --title "Data Analyst" --country ae'
            + (f' --city "{s["city"]}"' if s["city"] else ""),
        ]
    )


def _format_finished(result: FinishedRun) -> str:
    lines = [f"Run {result.run_id} finished: {result.outcome.replace('_', ' ')}", "", "Searches:"]
    for i, search in enumerate(result.searches, 1):
        status = search.get("error") or (
            f"found {search['found']}, kept {search.get('kept', 0)}"
            + (" (cache)" if search.get("cache_hit") else "")
        )
        lines.append(
            f'  {i}. [{search["rung"]}] "{search["text"]}" ({search["date_posted"]}): {status}'
        )
    requests = sum(1 for x in result.searches if x.get("cache_hit") is False)
    cached = sum(1 for x in result.searches if x.get("cache_hit"))
    tokens = ", ".join(
        f"{model}: {t['total']:,} tokens ({t['thinking']:,} thinking)"
        for model, t in result.llm_tokens.items()
    )
    lines += [f"JSearch: {requests} request(s), {cached} from cache", f"LLM: {tokens or 'none'}"]
    if result.matches:
        lines += [
            "",
            f"Top {len(result.matches)} of {len(result.jobs)} jobs ({result.rubric_name}):",
        ]
        lines += _format_matches(result.matches, result.trust)
    else:
        lines += ["", f"{len(result.jobs)} jobs:"]
        for i, job in enumerate(result.jobs, 1):
            details = [job.company, job.location, job.posted_text]
            lines += [f"{i:>2}. {job.title}", f"    {' | '.join(d for d in details if d)}"]
    for risky in result.removed_as_risky:
        lines.append(
            f"x Removed as likely scam: {risky['title']} ({risky['company']}): {risky['reason']}"
        )
    if result.dropped:
        lines += ["", f"Removed: {result.dropped}"]
    lines += [f"! {w}" for w in result.warnings]
    return "\n".join(lines)


def _format_matches(
    matches: list[JobMatch], trust: dict[str, TrustReport] | None = None
) -> list[str]:
    lines: list[str] = []
    for i, m in enumerate(matches, 1):
        total = f"{m.total:>3}/100" if m.total is not None else "  n/a  "
        flag = "  (low evidence)" if m.total is not None and m.coverage < MIN_COVERAGE else ""
        details = " | ".join(d for d in [m.job.company, m.job.location, m.job.posted_text] if d)
        lines += [f"{i:>2}. {total}  {m.job.title}{flag}", f"           {details}"]
        for c in m.criteria:
            if c.score is not None:
                lines.append(f'           {c.label:<16} {c.score:>2}/10  ad:  "{c.job_quote}"')
                if c.candidate_quote:
                    lines.append(f'           {"":<16}        you: "{c.candidate_quote}"')
            else:
                lines.append(f"           {c.label:<16}  n/a   {c.note}")
        if m.missing_skills:
            lines.append(f"           Missing: {', '.join(m.missing_skills)}")
        lines.append(f"           {m.summary}  ({m.coverage}% of weights judged)")
        report = (trust or {}).get(m.job.job_id)
        if report and report.level == "caution":
            reasons = "; ".join(dict.fromkeys(f.text for f in report.flags))
            lines.append(f"           ! Check before applying: {reasons}")
        elif report and report.good_signs:
            lines.append(f"           + {report.good_signs[0]}")
    return lines


def _rubric_from_args(args: argparse.Namespace) -> Rubric | None:
    """--rubric file wins; else --preset with any --weight changes; else None (auto preset)."""
    custom = (
        Rubric.model_validate_json(args.rubric.read_text(encoding="utf-8")) if args.rubric else None
    )
    weights = {key: int(value) for key, value in (item.split("=", 1) for item in args.weight)}
    return build_rubric(args.preset, weights, custom)


def _run_compare(args: argparse.Namespace) -> int:
    try:
        with JobScoutAgent.open() as agent:
            profile, prefs, rubric, pairs = agent.scoring_inputs(args.run_id)
        config = trace_config(run_name="compare_scorers", tags=["compare"])
        reports = compare_scorers(profile, prefs, rubric, pairs, args.models, config=config)
    except (RunNotFound, LLMNotConfigured) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    finally:
        flush()

    print(f"Scored the same {len(pairs)} jobs with {len(reports)} models ({rubric.name}):\n")
    print(f"{'model':<18}{'time':>7}{'input':>9}{'cached':>9}{'output':>9}{'cost':>10}")
    for r in reports:
        cost = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "-"
        print(
            f"{r.model:<18}{r.seconds:>6}s{r.input_tokens:>9,}{r.cache_read_tokens:>9,}"
            f"{r.output_tokens:>9,}{cost:>10}" + (f"   ERROR {r.error}" if r.error else "")
        )
    ok = [r for r in reports if not r.error]
    if len(ok) >= 2:
        a, b = ok[0], ok[1]
        agree = agreement(a.matches, b.matches)
        print(
            f"\nAgreement {a.model} vs {b.model}: {agree['same_top']} of the top {agree['top']} "
            f"jobs are the same; scores differ by {agree['mean_score_gap']} points on average."
        )
        print("\nPer job (total score):")
        for ma, mb in zip(a.matches, b.matches, strict=True):
            ta = "-" if ma.total is None else ma.total
            tb = "-" if mb.total is None else mb.total
            print(f"  {ta:>4} {tb:>4}  {ma.job.title[:60]}")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    try:
        query = SearchQuery(
            text=args.text, country=args.country, date_posted=args.date, remote_only=args.remote
        )
    except ValidationError as exc:
        print(f"Invalid search: {exc.errors()[0]['msg']}", file=sys.stderr)
        return EXIT_REJECTED

    try:
        search = JobSearch.from_settings()
        result = search.search(query, config=trace_config(run_name="search_jobs", tags=["cli"]))
    except (JSearchNotConfigured, ValueError) as exc:  # ValueError: malformed REDIS_URL
        print(str(exc), file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    except JSearchError as exc:
        print(exc.message, file=sys.stderr)
        return EXIT_SEARCH_FAILED
    finally:
        flush()

    print(result.model_dump_json(indent=2) if args.json else _format_search(result))
    return 0


def _format_search(result: SearchResult) -> str:
    filtered = filter_jobs(result.jobs, FilterRules(), now=datetime.now(UTC))
    q = result.query
    source = "yes" if result.cache_hit else "no (1 JSearch request used)"
    lines = [
        f'{len(filtered.kept)} jobs for "{q.text}" ({q.country}, posted: {q.date_posted})'
        f"   from cache: {source}",
        "",
    ]
    for i, job in enumerate(filtered.kept, 1):
        mode = "remote" if job.is_remote else "on-site"
        details = [job.company, job.location, job.posted_text, mode, job.salary]
        lines += [
            f"{i:>2}. {job.title}",
            f"    {' | '.join(d for d in details if d)}",
            f"    {job.apply_link}",
        ]
    if filtered.dropped:
        lines += ["", f"Removed: {filtered.dropped}"]
    return "\n".join(lines)


def _print_cv(cv: CVDocument) -> None:
    print(cv.text)
    print("\n" + "-" * 60)
    pages = cv.page_count or "-"
    print(f"format: {cv.source_format}   pages: {pages}   fingerprint: {cv.fingerprint}")
    print(f"redacted: {cv.redactions or 'nothing'}")
    print(f"hidden characters dropped: {cv.dropped_hidden_chars}")


def _format_profile(p: CandidateProfile) -> str:
    years = "unknown" if p.years_experience is None else f"{p.years_experience} years"
    lines = [
        f"Title:       {p.current_title or '-'}",
        f"Seniority:   {p.seniority}   Experience: {years}",
        f"Summary:     {p.summary or '-'}",
        f"Locations:   {', '.join(p.locations) or '-'}   Work mode: {p.work_mode}",
        "",
        f"Skills found in CV ({len(p.confirmed_skills)}): {', '.join(p.confirmed_skills) or '-'}",
        f"Skills NOT in CV ({len(p.flagged_skills)}):   {', '.join(p.flagged_skills) or '-'}",
        "",
        "Roles:",
        *[
            f"  - {r.title or '(no title)'}{f' @ {r.employer}' if r.employer else ''} ({r.kind}) "
            f"{r.start_text or '?'} -> {r.end_text or '?'}"
            for r in p.roles
        ],
    ]
    for label, items in (
        ("Education", p.education),
        ("Certifications", p.certifications),
        ("Projects", p.projects),
        ("Extracurriculars", p.extracurriculars),
    ):
        if items:
            lines += ["", f"{label}:", *[f"  - {item}" for item in items]]
    if p.warnings:
        lines += ["", "Warnings:", *[f"  ! {w}" for w in p.warnings]]
    return "\n".join(lines)


def _utf8_stdout() -> None:
    # Windows consoles default to a legacy code page; CV text is often not ASCII.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())

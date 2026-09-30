"""Command-line entry point for trying each phase without the web app.

uv run jobscout cv path/to/cv.pdf          # Phase 1a: what the LLM will see (no API key needed)
uv run jobscout profile path/to/cv.pdf     # Phase 1b: the fact-checked profile, traced in Langfuse
uv run jobscout search "data analyst in Dubai" --country ae   # Phase 2: real jobs, cached 24 h
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

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

    args = parser.parse_args(argv)
    _utf8_stdout()
    if args.command == "search":
        return _run_search(args)

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

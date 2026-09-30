"""Command-line entry point for trying each phase without the web app.

uv run jobscout cv path/to/cv.pdf          # Phase 1a: what the LLM will see (no API key needed)
uv run jobscout profile path/to/cv.pdf     # Phase 1b: the fact-checked profile, traced in Langfuse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobscout.cv import CVDocument, CVRejected, read_cv
from jobscout.llm import LLMNotConfigured
from jobscout.observability import flush, trace_config
from jobscout.profile import CandidateProfile, ProfileBuilder

EXIT_REJECTED = 2
EXIT_NOT_CONFIGURED = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jobscout", description="JobScout agent tools")
    commands = parser.add_subparsers(dest="command", required=True)

    cv_cmd = commands.add_parser("cv", help="Show the redacted text that would be sent to the LLM")
    cv_cmd.add_argument("path", type=Path)

    profile_cmd = commands.add_parser("profile", help="Build a fact-checked profile from a CV")
    profile_cmd.add_argument("path", type=Path)
    profile_cmd.add_argument("--json", action="store_true", help="Print the profile as JSON")

    args = parser.parse_args(argv)
    _utf8_stdout()

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

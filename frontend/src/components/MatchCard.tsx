import { AlertTriangle, ArrowUpRight, CheckCircle2, ChevronDown, Quote, UserRound } from "lucide-react";

import type { JobMatch, TrustReport } from "@/lib/api";

import { ScoreRing, scoreTone } from "./ScoreRing";

const LOW_EVIDENCE = 50; // same threshold the API uses for ranking

export function MatchCard({ match, rank, trust }: { match: JobMatch; rank: number; trust?: TrustReport }) {
  const { job } = match;
  const tone = scoreTone(match.total);
  const details = [job.company, job.location, job.posted_text, job.is_remote ? "Remote" : null].filter(Boolean);
  const missing = match.missing_skills ?? [];
  const judged = match.criteria.filter((c) => c.score != null);
  const notJudged = match.criteria.filter((c) => c.score == null);

  return (
    <article className="rounded-2xl border border-line bg-surface p-5 transition-shadow hover:shadow-lg hover:shadow-black/5 sm:p-6">
      <div className="flex items-start gap-4">
        <ScoreRing total={match.total} />
        <div className="min-w-0 flex-1">
          <p className="font-mono text-xs text-muted">#{rank}</p>
          <h3 className="font-display text-lg font-semibold leading-snug">{job.title}</h3>
          <p className="text-sm text-muted">{details.join(" · ")}</p>
          <p className="mt-1.5 flex flex-wrap items-center gap-x-2 text-xs">
            <span className={`font-medium ${tone.text}`}>{tone.label}</span>
            {match.total != null && match.coverage < LOW_EVIDENCE && (
              <span className="rounded-full bg-fair-soft px-2 py-0.5 text-fair">low evidence</span>
            )}
            <span className="text-muted">{match.coverage}% of your scorecard judged</span>
          </p>
        </div>
      </div>

      {match.summary && <p className="mt-4 text-sm">{match.summary}</p>}

      <ul className="mt-4 space-y-3">
        {judged.map((c) => (
          <li key={c.key} className="text-sm">
            <div className="flex items-center gap-3">
              <span className="w-28 shrink-0 truncate sm:w-32">{c.label}</span>
              <div className="h-1.5 flex-1 rounded-full bg-sunken" aria-hidden>
                <div
                  className="h-1.5 rounded-full bg-gradient-to-r from-accent to-accent-2"
                  style={{ width: `${(c.score ?? 0) * 10}%` }}
                />
              </div>
              <span className="w-11 shrink-0 text-right font-mono text-xs tabular-nums">{c.score}/10</span>
            </div>
            {c.job_quote && (
              <p className="mt-1 flex gap-1.5 text-xs text-muted sm:pl-35">
                <Quote className="mt-0.5 size-3 shrink-0" aria-label="From the ad:" />
                {c.job_quote}
              </p>
            )}
            {c.candidate_quote && (
              <p className="mt-0.5 flex gap-1.5 text-xs text-muted sm:pl-35">
                <UserRound className="mt-0.5 size-3 shrink-0" aria-label="From your CV:" />
                {c.candidate_quote}
              </p>
            )}
          </li>
        ))}
      </ul>

      {notJudged.length > 0 && (
        <details className="mt-3 text-xs text-muted">
          <summary className="flex cursor-pointer list-none items-center gap-1 hover:text-ink">
            <ChevronDown className="size-3.5" aria-hidden />
            {notJudged.length} not judged: {notJudged.map((c) => c.label).join(", ")}
          </summary>
          <ul className="mt-1 space-y-0.5 pl-5">
            {notJudged.map((c) => (
              <li key={c.key}>
                {c.label}: {c.note}
              </li>
            ))}
          </ul>
        </details>
      )}

      {missing.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center gap-1.5 text-sm">
          <span className="font-medium">Skills to build:</span>
          {missing.map((s) => (
            <span key={s} className="rounded-full border border-line px-2 py-0.5 text-xs text-muted">
              {s}
            </span>
          ))}
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
        <TrustLine trust={trust} />
        <a
          href={job.apply_link}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto inline-flex items-center gap-1 rounded-xl bg-gradient-to-r from-accent to-accent-2 px-4 py-2 text-sm font-medium text-accent-ink"
        >
          View and apply <ArrowUpRight className="size-4" aria-hidden />
        </a>
      </div>
    </article>
  );
}

function TrustLine({ trust }: { trust?: TrustReport }) {
  if (!trust) return <span />;
  if (trust.level === "caution") {
    const reasons = [...new Set((trust.flags ?? []).map((f) => f.text))].join("; ");
    return (
      <p className="flex max-w-md items-start gap-1.5 text-sm text-fair">
        <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden /> Check before applying: {reasons}
      </p>
    );
  }
  const good = trust.good_signs ?? [];
  if (good.length > 0) {
    return (
      <p className="flex items-center gap-1.5 text-sm text-good">
        <CheckCircle2 className="size-4" aria-hidden /> {good[0]}
      </p>
    );
  }
  return <span />;
}

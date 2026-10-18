import { Search, ShieldX, Sparkles, Trophy } from "lucide-react";
import Link from "next/link";

import type { ResultResponse } from "@/lib/api";

import { MatchCard } from "./MatchCard";

export function Results({ result }: { result: ResultResponse }) {
  const searches = result.searches as { text: string; date_posted: string; found?: number; kept?: number; cache_hit?: boolean; error?: string }[];

  return (
    <div className="space-y-6">
      <header className="space-y-4">
        <h1 className="font-display text-3xl font-bold tracking-tight sm:text-4xl">
          {result.matches.length > 0 ? (
            <>
              Your top <span className="text-gradient">{result.matches.length} jobs</span>
            </>
          ) : (
            "No matching jobs yet"
          )}
        </h1>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat icon={Trophy} label="Best score" value={result.matches[0]?.total ?? "–"} />
          <Stat icon={Search} label="Jobs found" value={result.jobs_found} />
          <Stat icon={ShieldX} label="Scams removed" value={result.removed_as_risky.length} />
          <Stat icon={Sparkles} label="Scorecard" value={result.rubric_name ?? "Default"} />
        </dl>
      </header>

      {result.warnings.map((w) => (
        <p key={w} className="rounded-lg bg-fair-soft px-3 py-2 text-sm text-fair">
          {w}
        </p>
      ))}

      {result.removed_as_risky.length > 0 && (
        <div className="rounded-lg bg-weak-soft px-3 py-2 text-sm text-weak">
          <p className="font-medium">Removed as likely scams:</p>
          <ul className="mt-1 list-disc pl-5">
            {result.removed_as_risky.map((r) => (
              <li key={`${r.title}-${r.company}`}>
                {r.title} ({r.company}): {r.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      <ol className="space-y-4">
        {result.matches.map((match, i) => (
          <li key={match.job.job_id}>
            <MatchCard match={match} rank={i + 1} trust={result.trust[match.job.job_id]} />
          </li>
        ))}
      </ol>

      <details className="rounded-2xl border border-line bg-surface p-4 text-sm">
        <summary className="cursor-pointer font-medium">How the agent searched</summary>
        <ol className="mt-3 space-y-1 text-muted">
          {searches.map((s, i) => (
            <li key={i}>
              {i + 1}. “{s.text}” ({s.date_posted}):{" "}
              {s.error ?? `found ${s.found}, kept ${s.kept ?? 0}${s.cache_hit ? " (from cache)" : ""}`}
            </li>
          ))}
        </ol>
      </details>

      <p className="text-center text-xs text-muted">
        No check can prove a posting is genuine. Never pay to apply for a job.
      </p>
      <p className="text-center">
        <Link href="/" className="text-sm text-accent underline">
          Search again with another CV or new choices
        </Link>
      </p>
    </div>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-xl border border-line bg-surface px-3 py-2.5">
      <dt className="flex items-center gap-1.5 text-xs text-muted">
        <Icon className="size-3.5" aria-hidden /> {label}
      </dt>
      <dd className="mt-0.5 truncate font-mono text-lg tabular-nums">{value}</dd>
    </div>
  );
}

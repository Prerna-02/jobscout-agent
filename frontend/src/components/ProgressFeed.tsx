import { CheckCircle2, Loader2 } from "lucide-react";

import type { Progress } from "@/lib/api";

const STAGES = [
  { label: "Search", steps: ["plan_query", "search_jobs", "filter_dedupe", "rewrite_query"] },
  { label: "Scam check", steps: ["trust_check"] },
  { label: "Shortlist", steps: ["shortlist"] },
  { label: "Score", steps: ["score_jobs", "rank_jobs"] },
];

/** Live view of the agent's work: a stage bar on top, then each finished step as it happens. */
export function ProgressFeed({ steps }: { steps: Progress[] }) {
  const reached = Math.max(
    0,
    ...steps.map((s) => STAGES.findIndex((stage) => stage.steps.includes(s.step))),
  );

  return (
    <section className="mx-auto max-w-2xl rounded-2xl border border-line bg-surface p-6 shadow-xl shadow-black/5">
      <div className="flex items-center gap-3">
        <Loader2 className="size-6 animate-spin text-accent" aria-hidden />
        <div>
          <h2 className="font-display text-xl font-semibold">Finding your jobs…</h2>
          <p className="text-sm text-muted">Usually about a minute. You can keep this page open.</p>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-4 gap-2" aria-hidden>
        {STAGES.map((stage, i) => (
          <div key={stage.label}>
            <div
              className={`h-1.5 rounded-full ${
                i < reached ? "bg-accent" : i === reached ? "animate-pulse bg-accent/60" : "bg-sunken"
              }`}
            />
            <p className={`mt-1.5 text-xs ${i <= reached ? "text-ink" : "text-muted"}`}>{stage.label}</p>
          </div>
        ))}
      </div>

      <ol aria-live="polite" className="mt-6 space-y-2.5">
        {steps.length === 0 && (
          <li className="flex items-center gap-2 text-sm font-medium">
            <Loader2 className="size-4 animate-spin text-accent" aria-hidden /> Planning your search…
          </li>
        )}
        {steps.map((step, i) => {
          const latest = i === steps.length - 1;
          return (
            <li key={`${step.step}-${i}`} className="animate-rise flex items-start gap-2 text-sm">
              {latest ? (
                <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-accent" aria-hidden />
              ) : (
                <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-good" aria-hidden />
              )}
              <span className={latest ? "font-medium" : "text-muted"}>{step.message}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

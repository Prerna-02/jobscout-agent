"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ProgressFeed } from "@/components/ProgressFeed";
import { Results } from "@/components/Results";
import { ReviewForm } from "@/components/ReviewForm";
import { StepIndicator } from "@/components/StepIndicator";
import {
  ApiError,
  getRun,
  type Progress,
  type ResultResponse,
  type ResumeRequest,
  resumeRun,
  type StartResponse,
} from "@/lib/api";

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string; canRetry: boolean }
  | { kind: "review"; run: StartResponse; notice?: string }
  | { kind: "searching"; run: StartResponse; steps: Progress[] }
  | { kind: "done"; result: ResultResponse };

export function RunView({ runId }: { runId: string }) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    getRun(runId)
      .then((run) => {
        if (cancelled) return;
        setState(run.status === "paused" ? { kind: "review", run: run as StartResponse } : { kind: "done", result: run as ResultResponse });
      })
      .catch((err) => {
        if (!cancelled) setState({ kind: "error", message: messageOf(err), canRetry: false });
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  async function search(run: StartResponse, choices: ResumeRequest) {
    setState({ kind: "searching", run, steps: [] });
    window.scrollTo({ top: 0, behavior: "smooth" });
    try {
      const result = await resumeRun(runId, choices, (step) =>
        setState((s) => (s.kind === "searching" ? { ...s, steps: [...s.steps, step] } : s)),
      );
      setState({ kind: "done", result });
    } catch (err) {
      const busy = err instanceof ApiError && (err.status === 503 || err.status === 429);
      // A busy server or a daily limit leaves the run paused: let the user try again.
      setState({ kind: "review", run, notice: messageOf(err) + (busy ? "" : " Your choices are kept below.") });
    }
  }

  switch (state.kind) {
    case "loading":
      return <p className="text-muted">Loading your run…</p>;
    case "error":
      return (
        <div className="space-y-3">
          <p role="alert" className="rounded-lg bg-weak-soft px-3 py-2 text-weak">
            {state.message}
          </p>
          <Link href="/" className="text-sm text-accent underline">
            Upload a CV
          </Link>
        </div>
      );
    case "review":
      return (
        <div className="mx-auto max-w-3xl space-y-6">
          <StepIndicator current="Review" />
          <header>
            <h1 className="font-display text-3xl font-bold tracking-tight sm:text-4xl">Check your profile</h1>
            <p className="text-muted">Fix anything that&apos;s wrong, then tell JobScout what to look for.</p>
          </header>
          {state.notice && (
            <p role="alert" className="rounded-lg bg-fair-soft px-3 py-2 text-sm text-fair">
              {state.notice}
            </p>
          )}
          <ReviewForm run={state.run} onSubmit={(choices) => search(state.run, choices)} />
        </div>
      );
    case "searching":
      return (
        <div className="mx-auto max-w-3xl space-y-6">
          <StepIndicator current="Search" />
          <ProgressFeed steps={state.steps} />
        </div>
      );
    case "done":
      return (
        <div className="mx-auto max-w-3xl space-y-6">
          <StepIndicator current="Results" />
          <Results result={state.result} />
        </div>
      );
  }
}

function messageOf(err: unknown): string {
  return err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
}

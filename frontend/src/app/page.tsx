import { FileSearch, ShieldAlert, SlidersHorizontal } from "lucide-react";
import Link from "next/link";

import { ExampleMatch } from "@/components/ExampleMatch";
import { UploadForm } from "@/components/UploadForm";

const FEATURES = [
  {
    icon: FileSearch,
    title: "Scores you can check",
    text: "Every score quotes the job ad, and your CV where it helps. No black-box numbers.",
  },
  {
    icon: ShieldAlert,
    title: "Scam check built in",
    text: "Ads asking for fees are removed. Vague or chat-app-only ads are flagged before you apply.",
  },
  {
    icon: SlidersHorizontal,
    title: "Your scorecard",
    text: "Students and professionals weigh different things. Change the weights or add your own.",
  },
];

export default function Home() {
  return (
    <div className="space-y-16">
      <section className="grid items-center gap-10 lg:grid-cols-[1.1fr_1fr]">
        <div className="animate-rise space-y-6">
          <p className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1 text-xs text-muted">
            <span className="size-1.5 rounded-full bg-good" aria-hidden />
            Live job listings, checked and ranked in about a minute
          </p>
          <h1 className="font-display text-4xl font-bold leading-[1.05] tracking-tight sm:text-5xl">
            Find jobs that <span className="text-gradient">actually fit</span> your CV.
          </h1>
          <p className="max-w-xl text-lg text-muted">
            Upload your CV. JobScout searches real openings, checks each one for scam signs, and
            ranks them against your profile, with the evidence behind every score.
          </p>
          <UploadForm />
        </div>
        <div className="animate-rise lg:pl-6" style={{ animationDelay: "120ms" }}>
          <ExampleMatch />
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {FEATURES.map(({ icon: Icon, title, text }) => (
          <div key={title} className="rounded-2xl border border-line bg-surface p-5">
            <span className="grid size-9 place-items-center rounded-lg bg-accent-soft text-accent">
              <Icon className="size-4.5" aria-hidden />
            </span>
            <p className="mt-3 font-display text-lg font-semibold">{title}</p>
            <p className="mt-1 text-sm text-muted">{text}</p>
          </div>
        ))}
      </section>

      <p className="text-center text-sm text-muted">
        Curious how it works?{" "}
        <Link href="/about" className="font-medium text-accent hover:underline">
          Read about JobScout →
        </Link>
      </p>
    </div>
  );
}

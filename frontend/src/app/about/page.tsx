import {
  FileUp,
  type LucideIcon,
  ListChecks,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Trophy,
} from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "About",
  description: "What JobScout is, how it finds and ranks jobs, and what happens to your CV.",
};

const STEPS: { icon: LucideIcon; title: string; text: string }[] = [
  {
    icon: FileUp,
    title: "Upload your CV",
    text: "JobScout reads your PDF or Word file and removes your name, email, phone and address before any AI model sees it. Gemini then turns the text into a profile, and plain code checks every skill really appears in your CV.",
  },
  {
    icon: ListChecks,
    title: "Review your profile",
    text: "You correct anything that's wrong, add or remove skills, and choose the job titles, country and city you want. Nothing is searched until you confirm.",
  },
  {
    icon: SlidersHorizontal,
    title: "Choose how jobs are scored",
    text: "Pick the Professional or Student scorecard, move the sliders, or add criteria of your own, such as languages or certifications.",
  },
  {
    icon: Search,
    title: "The agent searches",
    text: "JobScout searches live listings from job boards and company sites. If too few jobs come back, it rewrites its own search: similar titles, older postings, then the whole country.",
  },
  {
    icon: ShieldCheck,
    title: "Every ad is checked for scam signs",
    text: "Ads asking for fees are removed. Chat-app-only contact, personal email addresses or hidden employers are flagged, and unclear ads get a second look from an AI reviewer.",
  },
  {
    icon: Trophy,
    title: "Jobs are scored and ranked",
    text: "The closest jobs are scored on each criterion from 0 to 10 by Claude, with a quote from the ad as proof. Code checks every quote and does the maths, so you get a top 10 you can verify.",
  },
];

const PRIVACY = [
  "Your CV file is read once and never stored.",
  "Contact details are removed before any AI model sees your CV.",
  "Each search belongs to your browser only, through a secret key no one else has.",
  "Daily limits keep the service fair and free to run.",
];

const STACK = ["LangGraph", "LangChain", "Langfuse", "Gemini", "Claude", "JSearch", "FastAPI", "Next.js", "Postgres", "Redis"];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-14">
      <section className="animate-rise space-y-4">
        <p className="font-mono text-xs uppercase tracking-widest text-accent">About JobScout</p>
        <h1 className="font-display text-4xl font-bold leading-tight tracking-tight sm:text-5xl">
          A job search agent that <span className="text-gradient">reads your CV first</span>.
        </h1>
        <p className="text-lg text-muted">
          Job boards show you hundreds of listings and leave the matching to you. JobScout turns
          that around: it starts from your CV, searches real openings for you, throws out the
          suspicious ones, and explains exactly why each remaining job does or doesn&apos;t fit.
        </p>
      </section>

      <section className="space-y-6">
        <h2 className="font-display text-2xl font-semibold">How it works</h2>
        <ol className="relative space-y-6 border-l border-line pl-8">
          {STEPS.map(({ icon: Icon, title, text }, i) => (
            <li key={title} className="relative">
              <span className="absolute -left-[3.05rem] grid size-9 place-items-center rounded-full border border-line bg-surface text-accent">
                <Icon className="size-4" aria-hidden />
              </span>
              <p className="font-mono text-xs text-muted">Step {i + 1}</p>
              <h3 className="font-display text-lg font-semibold">{title}</h3>
              <p className="mt-1 text-muted">{text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-2xl border border-line bg-surface p-6">
          <h2 className="font-display text-xl font-semibold">What happens to your data</h2>
          <ul className="mt-3 space-y-2 text-sm text-muted">
            {PRIVACY.map((item) => (
              <li key={item} className="flex gap-2">
                <ShieldCheck className="mt-0.5 size-4 shrink-0 text-good" aria-hidden />
                {item}
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-2xl border border-line bg-surface p-6">
          <h2 className="font-display text-xl font-semibold">Built with</h2>
          <p className="mt-1 text-sm text-muted">
            One LangGraph agent with a review pause and a self-correcting search loop, traced
            end to end in Langfuse.
          </p>
          <ul className="mt-3 flex flex-wrap gap-1.5">
            {STACK.map((name) => (
              <li key={name} className="rounded-full bg-sunken px-2.5 py-1 font-mono text-xs">
                {name}
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="rounded-2xl bg-gradient-to-br from-accent to-accent-2 p-8 text-center text-accent-ink">
        <h2 className="font-display text-2xl font-semibold">Ready to see your matches?</h2>
        <p className="mt-1 opacity-90">It takes about a minute, and the first step is just your CV.</p>
        <Link
          href="/"
          className="mt-5 inline-block rounded-xl bg-surface px-5 py-2.5 font-medium text-ink shadow-lg transition-transform hover:-translate-y-0.5"
        >
          Upload your CV
        </Link>
      </section>
    </div>
  );
}

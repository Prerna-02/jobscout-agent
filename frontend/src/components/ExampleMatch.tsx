"use client";

import { CheckCircle2, MousePointer2, Quote } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { ScoreRing } from "./ScoreRing";

const TOTAL = 81;
const CRITERIA = [
  { label: "Skills", score: 8, quote: "hands-on experience with SQL, Excel and Python" },
  { label: "Projects", score: 7, quote: "build dashboards and reports for leadership" },
  { label: "Location", score: 10, quote: "This role is based in Dubai" },
];

/** One loop of the demo. `focus` is which evidence line the cursor points at
 * (0–2 the quotes, 3 the trust line); `full` is whether the bars are filled. */
const STEPS = [
  { ms: 700, full: false, focus: null },
  { ms: 1500, full: true, focus: null },
  { ms: 1700, full: true, focus: 0 },
  { ms: 1700, full: true, focus: 1 },
  { ms: 1700, full: true, focus: 2 },
  { ms: 1900, full: true, focus: 3 },
  { ms: 1200, full: true, focus: null },
] as const;

type Point = { x: number; y: number };

/** A clearly labelled example result card. It plays a short loop (bars fill, the score
 * counts up, a cursor checks each quote) so visitors see what they'll get. It shows the
 * finished card before JavaScript loads and for people who prefer reduced motion. */
export function ExampleMatch() {
  const box = useRef<HTMLElement>(null);
  const [step, setStep] = useState<number | null>(null); // null: still, finished card
  const [cursor, setCursor] = useState<Point | null>(null);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let timer: number;
    const play = (i: number) => {
      const focus = STEPS[i].focus;
      const target = focus == null ? null : box.current?.querySelector<HTMLElement>(`[data-demo="${focus}"]`);
      setCursor(pointAt(box.current, target));
      setStep(i);
      timer = window.setTimeout(() => play((i + 1) % STEPS.length), STEPS[i].ms);
    };
    timer = window.setTimeout(() => play(0), 1200);
    return () => window.clearTimeout(timer);
  }, []);

  const current = step == null ? null : STEPS[step];
  const full = current?.full ?? true;
  const focus = current?.focus ?? null;
  const shown = useCountTo(full ? TOTAL : 0);
  const lit = (i: number) => (focus === i ? "bg-accent-soft text-ink" : "");

  return (
    <figure ref={box} className="relative">
      <div className="absolute -inset-3 -z-10 rounded-3xl bg-gradient-to-br from-accent/25 to-accent-2/20 blur-2xl" aria-hidden />
      <div className="rounded-2xl border border-line bg-surface p-5 shadow-xl shadow-black/5">
        <div className="flex items-center justify-between">
          <span className="rounded-full bg-sunken px-2.5 py-0.5 text-xs font-medium text-muted">Example result</span>
          <span className="font-mono text-xs text-muted">#1 of 10</span>
        </div>
        <div className="mt-4 flex items-center gap-4">
          <ScoreRing total={TOTAL} shown={shown} />
          <div className="min-w-0">
            <p className="font-display text-lg font-semibold leading-snug">Data Analyst, Insights & Dashboards</p>
            <p className="text-sm text-muted">Example Co · Dubai · 2 days ago</p>
            <p className="mt-1 text-xs font-medium text-good">Strong match</p>
          </div>
        </div>
        <ul className="mt-5 space-y-3">
          {CRITERIA.map((c, i) => (
            <li key={c.label} className="text-sm">
              <div className="flex items-center gap-3">
                <span className="w-20 shrink-0">{c.label}</span>
                <div className="h-1.5 flex-1 rounded-full bg-sunken">
                  <div
                    className="h-1.5 rounded-full bg-gradient-to-r from-accent to-accent-2 transition-[width] duration-1000 ease-out"
                    style={{ width: full ? `${c.score * 10}%` : "0%", transitionDelay: full ? `${i * 150}ms` : "0ms" }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-xs tabular-nums">{c.score}/10</span>
              </div>
              <p
                data-demo={i}
                className={`-mx-1 mt-1 flex gap-1.5 rounded px-1 text-xs text-muted transition-colors duration-300 ${lit(i)}`}
              >
                <Quote className="mt-0.5 size-3 shrink-0" aria-hidden />
                {c.quote}
              </p>
            </li>
          ))}
        </ul>
        <p
          data-demo={3}
          className={`-mx-1 mt-4 flex items-center gap-1.5 rounded px-1 text-xs text-good transition-colors duration-300 ${focus === 3 ? "bg-good-soft" : ""}`}
        >
          <CheckCircle2 className="size-3.5" aria-hidden /> You apply on the employer&apos;s own website
        </p>
      </div>
      <figcaption className="mt-3 text-center text-xs text-muted">
        Every score quotes the job ad, so you can check it yourself.
      </figcaption>

      {step != null && (
        <div
          className="pointer-events-none absolute left-0 top-0 z-10 transition-[transform,opacity] duration-700 ease-in-out"
          style={{
            transform: `translate(${cursor?.x ?? 0}px, ${cursor?.y ?? 0}px)`,
            opacity: focus == null ? 0 : 1,
          }}
          aria-hidden
        >
          {focus != null && (
            <span key={step} className="animate-click absolute -left-2 -top-2 size-4 rounded-full bg-accent/50" />
          )}
          <MousePointer2 className="relative -left-1 -top-1 size-5 fill-ink stroke-surface drop-shadow" />
        </div>
      )}
    </figure>
  );
}

/** Where the cursor tip should sit: a little way into the target line, or just below the
 * card's bottom-right corner when there is no target (it fades out there). */
function pointAt(box: HTMLElement | null, target: HTMLElement | null | undefined): Point | null {
  if (!box) return null;
  const b = box.getBoundingClientRect();
  if (!target) return { x: b.width - 40, y: b.height - 10 };
  const t = target.getBoundingClientRect();
  return { x: t.left - b.left + Math.min(t.width * 0.55, 180), y: t.top - b.top + t.height * 0.6 };
}

/** Counts the displayed number towards `target` over about a second. */
function useCountTo(target: number, ms = 1000) {
  const [value, setValue] = useState(target);
  const from = useRef(target);

  useEffect(() => {
    const start = performance.now();
    const origin = from.current;
    let frame = requestAnimationFrame(function tick(now) {
      const t = Math.min(1, (now - start) / ms);
      const next = Math.round(origin + (target - origin) * (1 - (1 - t) ** 3));
      from.current = next;
      setValue(next);
      if (t < 1) frame = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(frame);
  }, [target, ms]);

  return value;
}

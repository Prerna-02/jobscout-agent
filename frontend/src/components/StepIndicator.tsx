import { Check } from "lucide-react";

const STEPS = ["Upload", "Review", "Search", "Results"] as const;
export type Step = (typeof STEPS)[number];

/** Where the user is in the four-step flow. */
export function StepIndicator({ current }: { current: Step }) {
  const at = STEPS.indexOf(current);
  return (
    <ol className="flex items-center gap-2 text-xs sm:text-sm" aria-label="Progress">
      {STEPS.map((step, i) => {
        const done = i < at;
        const active = i === at;
        return (
          <li key={step} className="flex items-center gap-2" aria-current={active ? "step" : undefined}>
            <span
              className={`grid size-6 place-items-center rounded-full font-mono text-xs ${
                done
                  ? "bg-accent text-accent-ink"
                  : active
                    ? "border-2 border-accent text-accent"
                    : "border border-line text-muted"
              }`}
            >
              {done ? <Check className="size-3.5" aria-hidden /> : i + 1}
            </span>
            <span className={active ? "font-medium" : "text-muted"}>{step}</span>
            {i < STEPS.length - 1 && <span className="h-px w-4 bg-line sm:w-8" aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}

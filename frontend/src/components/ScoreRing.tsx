/** A 0–100 score drawn as a ring, coloured by band: green strong, amber fair, red weak. */
export function scoreTone(total: number | null | undefined) {
  if (total == null) return { stroke: "var(--line)", text: "text-muted", label: "Not scored" };
  if (total >= 70) return { stroke: "var(--good)", text: "text-good", label: "Strong match" };
  if (total >= 50) return { stroke: "var(--fair)", text: "text-fair", label: "Worth a look" };
  return { stroke: "var(--weak)", text: "text-weak", label: "Stretch" };
}

/** `shown` draws a different number (e.g. mid count-up) while keeping the colour of `total`. */
export function ScoreRing({
  total,
  shown = total,
  size = 76,
}: {
  total: number | null | undefined;
  shown?: number | null;
  size?: number;
}) {
  const tone = scoreTone(total);
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const filled = shown == null ? 0 : (Math.max(0, Math.min(100, shown)) / 100) * circumference;
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg viewBox="0 0 100 100" className="size-full -rotate-90" role="img" aria-label={total == null ? "Not scored" : `Score ${total} out of 100`}>
        <circle cx="50" cy="50" r={radius} fill="none" stroke="var(--sunken)" strokeWidth="9" />
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke={tone.stroke}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
        />
      </svg>
      <span className={`absolute inset-0 grid place-items-center font-mono text-xl font-medium tabular-nums ${tone.text}`}>
        {shown ?? "–"}
      </span>
    </div>
  );
}

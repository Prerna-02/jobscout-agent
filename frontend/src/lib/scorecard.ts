/**
 * Turning the scorecard editor's state into what the API expects, with the smallest
 * possible request: a preset name, a preset plus weight changes, or fully custom criteria.
 */
import type { Criterion, ResumeRequest, Rubric } from "./api";

export type EditableCriterion = Criterion & { isNew?: boolean };

/** Each criterion's share of the total, as whole percentages that add up to 100. */
export function percentages(criteria: { weight: number }[]): number[] {
  const total = criteria.reduce((sum, c) => sum + c.weight, 0);
  if (total === 0) return criteria.map(() => 0);
  const raw = criteria.map((c) => (c.weight / total) * 100);
  const rounded = raw.map(Math.floor);
  let left = 100 - rounded.reduce((a, b) => a + b, 0);
  // Give the leftover points to the largest remainders, so the shares always total 100.
  const order = raw.map((r, i) => [r - Math.floor(r), i] as const).sort((a, b) => b[0] - a[0]);
  for (const [, i] of order) {
    if (left-- <= 0) break;
    rounded[i] += 1;
  }
  return rounded;
}

/** "Team leadership!" -> "team_leadership" (the API's criterion key format). */
export function keyFromLabel(label: string, taken: Set<string>): string {
  const base =
    label
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .replace(/^(\d)/, "c_$1")
      .slice(0, 24) || "custom";
  let key = base.length >= 2 ? base : `${base}_x`;
  for (let n = 2; taken.has(key); n++) key = `${base}_${n}`;
  return key;
}

export function scorecardRequest(
  presetName: string,
  preset: Rubric,
  edited: EditableCriterion[],
): Pick<ResumeRequest, "preset" | "weights" | "rubric"> {
  const original = new Map(preset.criteria.map((c) => [c.key, c.weight]));
  const sameCriteria =
    edited.length === preset.criteria.length && edited.every((c) => original.has(c.key));
  if (sameCriteria) {
    const weights: Record<string, number> = {};
    for (const c of edited) if (original.get(c.key) !== c.weight) weights[c.key] = c.weight;
    return Object.keys(weights).length ? { preset: presetName, weights } : { preset: presetName };
  }
  return {
    rubric: {
      name: "My scorecard",
      version: 1,
      search_hints: preset.search_hints,
      criteria: edited.map(({ key, label, compare, weight }) => ({ key, label, compare, weight })),
    },
  };
}

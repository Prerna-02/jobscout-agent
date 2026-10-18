"use client";

import { useState } from "react";

import type { Rubric } from "@/lib/api";
import { type EditableCriterion, keyFromLabel, percentages } from "@/lib/scorecard";

type Props = {
  presets: Record<string, Rubric>;
  presetName: string;
  criteria: EditableCriterion[];
  onPresetChange: (name: string) => void;
  onChange: (criteria: EditableCriterion[]) => void;
};

const MAX_CRITERIA = 10;

export function ScorecardEditor({ presets, presetName, criteria, onPresetChange, onChange }: Props) {
  const [label, setLabel] = useState("");
  const [compare, setCompare] = useState("");
  const shares = percentages(criteria);

  function update(index: number, change: Partial<EditableCriterion>) {
    onChange(criteria.map((c, i) => (i === index ? { ...c, ...change } : c)));
  }

  function add() {
    const taken = new Set(criteria.map((c) => c.key));
    onChange([
      ...criteria,
      {
        key: keyFromLabel(label, taken),
        label: label.trim(),
        compare: compare.trim(),
        weight: 10,
        isNew: true,
      },
    ]);
    setLabel("");
    setCompare("");
  }

  const canAdd = label.trim().length >= 2 && compare.trim().length >= 5 && criteria.length < MAX_CRITERIA;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {Object.entries(presets).map(([name, rubric]) => (
          <button
            key={name}
            type="button"
            onClick={() => onPresetChange(name)}
            aria-pressed={presetName === name}
            className={`rounded-full border px-3 py-1 text-sm ${
              presetName === name
                ? "border-accent bg-accent text-accent-ink"
                : "border-line bg-surface hover:border-accent"
            }`}
          >
            {rubric.name}
          </button>
        ))}
        <span className="text-xs text-muted">Pick a starting point, then adjust.</span>
      </div>

      <ul className="space-y-3">
        {criteria.map((c, i) => (
          <li key={c.key} className="rounded-lg border border-line bg-surface p-3">
            <div className="flex items-baseline justify-between gap-3">
              <label htmlFor={`weight-${c.key}`} className="font-medium">
                {c.label}
                {c.isNew && <span className="ml-2 text-xs text-accent">your criterion</span>}
              </label>
              <span className="font-mono text-sm tabular-nums">{shares[i]}%</span>
            </div>
            <p className="mt-0.5 text-xs text-muted">{c.compare}</p>
            <div className="mt-2 flex items-center gap-3">
              <input
                id={`weight-${c.key}`}
                type="range"
                min={0}
                max={100}
                step={5}
                value={c.weight}
                onChange={(e) => update(i, { weight: Number(e.target.value) })}
                className="w-full accent-[var(--accent)]"
              />
              {c.isNew && (
                <button
                  type="button"
                  onClick={() => onChange(criteria.filter((_, j) => j !== i))}
                  className="text-xs text-muted underline hover:text-weak"
                >
                  Remove
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>

      {criteria.length < MAX_CRITERIA && (
        <div className="rounded-lg border border-dashed border-line p-3">
          <p className="text-sm font-medium">Add your own criterion</p>
          <div className="mt-2 grid gap-2 sm:grid-cols-[1fr_2fr_auto]">
            <input
              id="new-criterion-label"
              placeholder="Name, e.g. Languages"
              value={label}
              maxLength={40}
              onChange={(e) => setLabel(e.target.value)}
              className="rounded-lg border border-line bg-surface px-3 py-2 text-sm"
            />
            <input
              id="new-criterion-compare"
              placeholder="What to compare, e.g. languages the job needs vs mine"
              value={compare}
              maxLength={200}
              onChange={(e) => setCompare(e.target.value)}
              className="rounded-lg border border-line bg-surface px-3 py-2 text-sm"
            />
            <button
              type="button"
              disabled={!canAdd}
              onClick={add}
              className="rounded-lg border border-accent px-3 py-2 text-sm text-accent disabled:opacity-40"
            >
              Add
            </button>
          </div>
          <p className="mt-2 text-xs text-muted">
            Criteria about age, gender, religion, nationality or looks are not allowed.
          </p>
        </div>
      )}
    </div>
  );
}

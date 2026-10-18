"use client";

import { AlertTriangle, Plus, RotateCcw, X } from "lucide-react";
import { useState } from "react";

import { addSkills } from "@/lib/skills";


type Props = {
  skills: string[];
  onChange: (skills: string[]) => void;
  flagged: string[]; // listed by the AI but not written in the CV
};

/** Remove skills that don't fit, add missing ones, and decide on the flagged ones. */
export function SkillsEditor({ skills, onChange, flagged }: Props) {
  const [draft, setDraft] = useState("");
  const [removed, setRemoved] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const has = (name: string) => skills.some((s) => s.toLowerCase() === name.toLowerCase());
  const pendingFlagged = flagged.filter((s) => !has(s));

  function remove(name: string) {
    onChange(skills.filter((s) => s !== name));
    setRemoved((r) => [name, ...r.filter((x) => x !== name)]);
  }

  function restore(name: string) {
    if (!has(name)) onChange([...skills, name]);
    setRemoved((r) => r.filter((x) => x !== name));
  }

  function add() {
    const { added, problem } = addSkills(skills, draft);
    setError(problem);
    if (added.length === 0) return;
    onChange([...skills, ...added]);
    const lower = new Set(added.map((s) => s.toLowerCase()));
    setRemoved((r) => r.filter((x) => !lower.has(x.toLowerCase())));
    if (!problem) setDraft("");
  }

  return (
    <div className="space-y-4">
      <ul className="flex flex-wrap gap-1.5" aria-label="Your skills">
        {skills.map((s) => (
          <li key={s} className="group flex items-center gap-1 rounded-full border border-line bg-sunken py-1 pr-1 pl-3 text-sm">
            {s}
            <button
              type="button"
              onClick={() => remove(s)}
              aria-label={`Remove ${s}`}
              className="grid size-5 place-items-center rounded-full text-muted hover:bg-weak-soft hover:text-weak"
            >
              <X className="size-3" aria-hidden />
            </button>
          </li>
        ))}
        {skills.length === 0 && <li className="text-sm text-muted">No skills yet. Add a few below.</li>}
      </ul>

      <div>
        <div className="flex gap-2">
          <input
            id="add-skill"
            value={draft}
            maxLength={600}
            placeholder="Add skills, e.g. Power BI, Git"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                add();
              }
            }}
            className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm"
          />
          <button
            type="button"
            onClick={add}
            className="flex items-center gap-1 rounded-lg border border-accent px-3 text-sm text-accent hover:bg-accent-soft"
          >
            <Plus className="size-4" aria-hidden /> Add
          </button>
        </div>
        {error && <p className="mt-1 text-xs text-weak">{error}</p>}
      </div>

      {removed.length > 0 && (
        <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted">
          Removed:
          {removed.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => restore(s)}
              className="flex items-center gap-1 rounded-full border border-dashed border-line px-2 py-0.5 hover:border-accent hover:text-accent"
              title={`Put ${s} back`}
            >
              <RotateCcw className="size-3" aria-hidden /> {s}
            </button>
          ))}
        </p>
      )}

      {pendingFlagged.length > 0 && (
        <div className="rounded-xl border border-fair/30 bg-fair-soft p-3">
          <p className="flex items-center gap-1.5 text-sm font-medium text-fair">
            <AlertTriangle className="size-4" aria-hidden />
            Not found in your CV. Add only the ones you really have:
          </p>
          <ul className="mt-2 flex flex-wrap gap-1.5">
            {pendingFlagged.map((s) => (
              <li key={s}>
                <button
                  type="button"
                  onClick={() => onChange([...skills, s])}
                  className="flex items-center gap-1 rounded-full border border-fair/50 px-2.5 py-1 text-sm text-fair hover:bg-fair hover:text-surface"
                >
                  <Plus className="size-3" aria-hidden /> {s}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

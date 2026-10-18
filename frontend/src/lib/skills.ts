const SAFE = /^[\w .,&+#/()'-]+$/; // same characters the API accepts
export const MAX_SKILLS = 80;

/** The new skills in `text`, which may be one skill or a list ("Power BI, Python; Git"),
 * plus a message about any that were left out. */
export function addSkills(current: string[], text: string): { added: string[]; problem: string | null } {
  const known = new Set(current.map((s) => s.toLowerCase()));
  const added: string[] = [];
  const problems: string[] = [];
  for (const part of text.split(/[,;\n]/)) {
    const name = part.trim().replace(/\s+/g, " ");
    if (!name) continue;
    if (name.length > 60 || !SAFE.test(name)) problems.push(`"${name}" has unusual characters or is too long`);
    else if (known.has(name.toLowerCase())) problems.push(`${name} is already in your list`);
    else if (current.length + added.length >= MAX_SKILLS) problems.push(`you can keep up to ${MAX_SKILLS} skills`);
    else {
      added.push(name);
      known.add(name.toLowerCase());
    }
  }
  const problem = problems.length ? `Not added: ${[...new Set(problems)].join("; ")}.` : null;
  return { added, problem };
}

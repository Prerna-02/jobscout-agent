import { describe, expect, it } from "vitest";

import type { Rubric } from "./api";
import { keyFromLabel, percentages, scorecardRequest } from "./scorecard";
import { addSkills } from "./skills";
import { createSSEParser, type SSEEvent } from "./sse";

describe("SSE parser", () => {
  it("handles events split across chunks", () => {
    const events: SSEEvent[] = [];
    const parser = createSSEParser((e) => events.push(e));
    parser.push('event: progress\ndata: {"step":"plan_query",');
    parser.push('"message":"Searching"}\n\nevent: res');
    parser.push('ult\ndata: {"matches":[]}\n\n');
    expect(events).toEqual([
      { event: "progress", data: { step: "plan_query", message: "Searching" } },
      { event: "result", data: { matches: [] } },
    ]);
  });

  it("skips malformed events instead of failing", () => {
    const events: SSEEvent[] = [];
    const parser = createSSEParser((e) => events.push(e));
    parser.push("event: progress\ndata: {broken\n\nevent: result\ndata: {}\n\n");
    expect(events).toEqual([{ event: "result", data: {} }]);
  });
});

describe("scorecard", () => {
  const preset: Rubric = {
    name: "Student",
    version: 1,
    search_hints: ["internship"],
    criteria: [
      { key: "skills", label: "Skills", compare: "Skills vs skills", weight: 60 },
      { key: "location", label: "Location", compare: "Place vs place", weight: 40 },
    ],
  };

  it("percentages always add up to 100", () => {
    expect(percentages([{ weight: 1 }, { weight: 1 }, { weight: 1 }])).toEqual([34, 33, 33]);
    expect(percentages([{ weight: 0 }, { weight: 0 }])).toEqual([0, 0]);
  });

  it("sends just the preset when nothing changed", () => {
    expect(scorecardRequest("student", preset, [...preset.criteria])).toEqual({ preset: "student" });
  });

  it("sends only the changed weights", () => {
    const edited = [{ ...preset.criteria[0], weight: 80 }, preset.criteria[1]];
    expect(scorecardRequest("student", preset, edited)).toEqual({
      preset: "student",
      weights: { skills: 80 },
    });
  });

  it("sends full custom criteria when criteria were added", () => {
    const added = { key: "languages", label: "Languages", compare: "Languages vs languages", weight: 20 };
    const request = scorecardRequest("student", preset, [...preset.criteria, added]);
    expect(request.rubric?.criteria.map((c) => c.key)).toEqual(["skills", "location", "languages"]);
    expect(request.rubric?.search_hints).toEqual(["internship"]);
  });

  it("makes valid, unique keys from labels", () => {
    expect(keyFromLabel("Team leadership!", new Set())).toBe("team_leadership");
    expect(keyFromLabel("Skills", new Set(["skills"]))).toBe("skills_2");
    expect(keyFromLabel("3D design", new Set())).toBe("c_3d_design");
  });
});

describe("addSkills", () => {
  it("splits a pasted list into separate skills", () => {
    const { added, problem } = addSkills(["SQL"], "Power BI,Leadership, Problem Solving;Python\n Git");
    expect(added).toEqual(["Power BI", "Leadership", "Problem Solving", "Python", "Git"]);
    expect(problem).toBeNull();
  });

  it("skips duplicates and odd text but keeps the rest", () => {
    const { added, problem } = addSkills(["SQL"], "sql, Excel, excel, <b>x</b>");
    expect(added).toEqual(["Excel"]);
    expect(problem).toContain("sql is already in your list");
    expect(problem).toContain("unusual characters");
  });
});

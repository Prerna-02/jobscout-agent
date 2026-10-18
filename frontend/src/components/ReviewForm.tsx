"use client";

import { ArrowRight, Briefcase, Gauge, MapPin, Sparkles, UserRound, X } from "lucide-react";
import { useState } from "react";

import type { ResumeRequest, StartResponse } from "@/lib/api";
import { type EditableCriterion, scorecardRequest } from "@/lib/scorecard";

import { ScorecardEditor } from "./ScorecardEditor";
import { SkillsEditor } from "./SkillsEditor";

const COUNTRIES: [string, string][] = [
  ["ae", "United Arab Emirates"],
  ["in", "India"],
  ["us", "United States"],
  ["gb", "United Kingdom"],
  ["sa", "Saudi Arabia"],
  ["qa", "Qatar"],
  ["sg", "Singapore"],
  ["ca", "Canada"],
  ["au", "Australia"],
  ["de", "Germany"],
];
const DATES: [ResumeRequest["date_posted"], string][] = [
  ["3days", "Last 3 days"],
  ["week", "Last week"],
  ["month", "Last month"],
  ["all", "Any time"],
];
const field = "w-full rounded-lg border border-line bg-surface px-3 py-2";

type Props = { run: StartResponse; onSubmit: (choices: ResumeRequest) => void };

export function ReviewForm({ run, onSubmit }: Props) {
  const { profile, suggested, presets } = run;
  const allSkills = profile.skills ?? [];
  const flagged = allSkills.filter((s) => !s.in_cv).map((s) => s.name);

  const [skills, setSkills] = useState<string[]>(allSkills.filter((s) => s.in_cv).map((s) => s.name));
  const [titles, setTitles] = useState<string[]>((suggested.target_titles as string[]) ?? []);
  const [titleDraft, setTitleDraft] = useState("");
  const [country, setCountry] = useState<string>((suggested.country as string) ?? "");
  const [otherCountry, setOtherCountry] = useState(false);
  const [city, setCity] = useState<string>((suggested.city as string) ?? "");
  const [remote, setRemote] = useState<boolean>(Boolean(suggested.remote_only));
  const [date, setDate] = useState<ResumeRequest["date_posted"]>("week");
  const [presetName, setPresetName] = useState<string>((suggested.preset as string) ?? "professional");
  const [criteria, setCriteria] = useState<EditableCriterion[]>(presets[presetName].criteria);

  function addTitle() {
    const title = titleDraft.trim();
    if (title && titles.length < 3 && !titles.includes(title)) setTitles([...titles, title]);
    setTitleDraft("");
  }

  function choosePreset(name: string) {
    setPresetName(name);
    setCriteria(presets[name].criteria);
  }

  const countryOk = /^[a-z]{2}$/i.test(country);
  const missing = [
    titles.length === 0 && "a job title",
    !countryOk && "a country",
    !criteria.some((c) => c.weight > 0) && "a scorecard weight above 0",
  ].filter(Boolean) as string[];
  const countryName = COUNTRIES.find(([c]) => c === country)?.[1] ?? country.toUpperCase();

  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (missing.length) return;
    onSubmit({
      target_titles: titles,
      country: country.toLowerCase(),
      city: city.trim() || null,
      remote_only: remote,
      date_posted: date,
      employment_types: [],
      skills,
      ...scorecardRequest(presetName, presets[presetName], criteria),
    });
  }

  return (
    <form onSubmit={submit} className="space-y-6 pb-4">
      <Card icon={UserRound} title="Your profile" note="Built from your CV. Contact details were removed first.">
        <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
          <Fact label="Seniority" value={profile.seniority} />
          <Fact
            label="Experience"
            value={profile.years_experience == null ? "Unknown" : `${profile.years_experience} years`}
          />
          <Fact label="Education" value={(profile.education ?? []).join("; ") || "-"} />
          <Fact label="Based in" value={(profile.locations ?? []).join("; ") || "-"} />
        </dl>
        {profile.summary && (
          <p className="mt-4 rounded-lg bg-sunken px-3 py-2 text-sm text-muted">{profile.summary}</p>
        )}
      </Card>

      <Card
        icon={Sparkles}
        title="Skills"
        note={`${skills.length} skills will be used. Remove any that don't describe you, and add what's missing.`}
      >
        <SkillsEditor skills={skills} onChange={setSkills} flagged={flagged} />
      </Card>

      <Card icon={Briefcase} title="What are you looking for?">
        <div className="space-y-5">
          <div>
            <label htmlFor="title-input" className="mb-1.5 block text-sm font-medium">
              Job titles <span className="font-normal text-muted">(up to 3)</span>
            </label>
            {titles.length > 0 && (
              <ul className="mb-2 flex flex-wrap gap-1.5">
                {titles.map((t) => (
                  <li key={t} className="flex items-center gap-1 rounded-full bg-accent-soft py-1 pr-1 pl-3 text-sm text-accent">
                    {t}
                    <button
                      type="button"
                      aria-label={`Remove ${t}`}
                      onClick={() => setTitles(titles.filter((x) => x !== t))}
                      className="grid size-5 place-items-center rounded-full hover:bg-surface"
                    >
                      <X className="size-3" aria-hidden />
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {titles.length < 3 && (
              <div className="flex gap-2">
                <input
                  id="title-input"
                  value={titleDraft}
                  maxLength={60}
                  placeholder="e.g. Data Analyst, then press Enter"
                  onChange={(e) => setTitleDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addTitle();
                    }
                  }}
                  className={field}
                />
                <button type="button" onClick={addTitle} className="rounded-lg border border-line px-3 text-sm hover:border-accent">
                  Add
                </button>
              </div>
            )}
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="mb-1.5 flex items-center gap-1 font-medium">
                <MapPin className="size-3.5 text-muted" aria-hidden /> Country
              </span>
              <select
                id="country"
                value={otherCountry ? "other" : country}
                onChange={(e) => {
                  const other = e.target.value === "other";
                  setOtherCountry(other);
                  setCountry(other ? "" : e.target.value);
                }}
                className={field}
              >
                <option value="">Choose a country</option>
                {COUNTRIES.map(([code, name]) => (
                  <option key={code} value={code}>
                    {name}
                  </option>
                ))}
                <option value="other">Other (type the 2-letter code)</option>
              </select>
              {otherCountry && (
                <input
                  id="country-code"
                  value={country}
                  maxLength={2}
                  placeholder="2-letter code, e.g. fr"
                  onChange={(e) => setCountry(e.target.value.toLowerCase())}
                  className={`${field} mt-2`}
                />
              )}
            </label>
            <label className="block text-sm">
              <span className="mb-1.5 block font-medium">City (optional)</span>
              <input id="city" value={city} maxLength={40} onChange={(e) => setCity(e.target.value)} className={field} />
            </label>
            <label className="block text-sm">
              <span className="mb-1.5 block font-medium">Posted</span>
              <select
                id="date-posted"
                value={date}
                onChange={(e) => setDate(e.target.value as ResumeRequest["date_posted"])}
                className={field}
              >
                {DATES.map(([value, name]) => (
                  <option key={value} value={value}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex cursor-pointer items-center justify-between gap-3 self-end rounded-lg border border-line bg-surface px-3 py-2 text-sm">
              Remote jobs only
              <input
                id="remote-only"
                type="checkbox"
                checked={remote}
                onChange={(e) => setRemote(e.target.checked)}
                className="size-4 accent-[var(--accent)]"
              />
            </label>
          </div>
        </div>
      </Card>

      <Card
        icon={Gauge}
        title="How should jobs be scored?"
        note="Each job gets 0–10 on every criterion, with a quote from the ad as proof. Weights decide how much each one counts."
      >
        <ScorecardEditor
          presets={presets}
          presetName={presetName}
          criteria={criteria}
          onPresetChange={choosePreset}
          onChange={setCriteria}
        />
      </Card>

      <div className="sticky bottom-4 z-10">
        <div className="flex flex-col gap-3 rounded-2xl border border-line bg-surface/90 p-3 shadow-xl shadow-black/10 backdrop-blur sm:flex-row sm:items-center">
          <p className="min-w-0 flex-1 px-1 text-sm text-muted">
            {missing.length ? (
              <>Still needed: {missing.join(", ")}.</>
            ) : (
              <>
                <span className="font-medium text-ink">{titles.join(" / ")}</span>
                {" · "}
                {city ? `${city}, ` : ""}
                {countryName}
                {remote ? " · remote" : ""}
                {" · "}
                {presets[presetName].name} scorecard
              </>
            )}
          </p>
          <button
            type="submit"
            disabled={missing.length > 0}
            className="flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent to-accent-2 px-5 py-2.5 font-medium text-accent-ink shadow-lg shadow-accent/20 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
          >
            Find and score jobs <ArrowRight className="size-4" aria-hidden />
          </button>
        </div>
      </div>
    </form>
  );
}

function Card({
  icon: Icon,
  title,
  note,
  children,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="mb-4 flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent">
          <Icon className="size-4.5" aria-hidden />
        </span>
        <div>
          <h2 className="font-display text-xl font-semibold">{title}</h2>
          {note && <p className="text-sm text-muted">{note}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs uppercase tracking-wide text-muted">{label}</dt>
      <dd className="mt-0.5 break-words">{value}</dd>
    </div>
  );
}

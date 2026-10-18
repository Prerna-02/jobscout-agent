"use client";

import { FileText, Loader2, Lock, UploadCloud } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState, useSyncExternalStore } from "react";

import { ApiError, startRun } from "@/lib/api";
import { inviteCode, saveInviteCode } from "@/lib/storage";

const MAX_MB = 4;
const ACCEPT = ".pdf,.docx";

export function UploadForm() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The saved code is read from the browser after hydration (the server has no storage).
  const savedInvite = useSyncExternalStore(noSubscription, inviteCode, () => "");
  const [typedInvite, setInvite] = useState<string | null>(null);
  const invite = typedInvite ?? savedInvite;
  const [needsInvite, setNeedsInvite] = useState(false);

  function choose(candidate: File | undefined) {
    setError(null);
    if (!candidate) return;
    if (!/\.(pdf|docx)$/i.test(candidate.name)) {
      setError("Please choose a PDF or Word (.docx) file.");
      return;
    }
    if (candidate.size > MAX_MB * 1024 * 1024) {
      setError(`That file is larger than ${MAX_MB} MB.`);
      return;
    }
    setFile(candidate);
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    saveInviteCode(invite);
    try {
      const run = await startRun(file);
      router.push(`/runs/${run.run_id}`);
    } catch (err) {
      const apiError = err instanceof ApiError ? err : null;
      if (apiError?.status === 401) setNeedsInvite(true);
      setError(apiError?.message ?? "Something went wrong. Please try again.");
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <div
        role="button"
        tabIndex={0}
        aria-label="Choose your CV file"
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          choose(e.dataTransfer.files[0]);
        }}
        className={`group flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          dragging ? "border-accent bg-accent-soft" : "border-line bg-surface hover:border-accent hover:bg-accent-soft/40"
        }`}
      >
        <input
          ref={input}
          id="cv-file"
          type="file"
          accept={ACCEPT}
          className="sr-only"
          onChange={(e) => choose(e.target.files?.[0])}
        />
        {file ? (
          <>
            <FileText className="mb-3 size-8 text-accent" aria-hidden />
            <p className="font-medium">{file.name}</p>
            <p className="mt-1 text-sm text-muted">
              {(file.size / 1024).toFixed(0)} KB · click to choose another file
            </p>
          </>
        ) : (
          <>
            <span className="mb-3 grid size-12 place-items-center rounded-full bg-accent-soft text-accent transition-transform group-hover:-translate-y-0.5">
              <UploadCloud className="size-6" aria-hidden />
            </span>
            <p className="font-medium">Drop your CV here, or click to choose</p>
            <p className="mt-1 text-sm text-muted">PDF or Word (.docx), up to {MAX_MB} MB</p>
          </>
        )}
      </div>

      {(needsInvite || invite) && (
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Invite code</span>
          <input
            id="invite-code"
            value={invite}
            onChange={(e) => setInvite(e.target.value)}
            className="w-full rounded-lg border border-line bg-surface px-3 py-2"
            autoComplete="off"
          />
        </label>
      )}

      {error && (
        <p role="alert" className="rounded-lg bg-weak-soft px-3 py-2 text-sm text-weak">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={!file || busy}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent to-accent-2 px-4 py-3 font-medium text-accent-ink shadow-lg shadow-accent/20 transition-opacity hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
      >
        {busy && <Loader2 className="size-4 animate-spin" aria-hidden />}
        {busy ? "Reading your CV… (about 15 seconds)" : "Read my CV"}
      </button>
      <p className="flex items-start justify-center gap-1.5 text-xs text-muted">
        <Lock className="mt-0.5 size-3 shrink-0" aria-hidden />
        Your name, email, phone and address are removed before any AI model sees your CV. The file
        itself is never stored.
      </p>
    </form>
  );
}

function noSubscription() {
  return () => {};
}

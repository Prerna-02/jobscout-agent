import { ShieldCheck } from "lucide-react";

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-line">
      <div className="mx-auto max-w-5xl px-4 py-6 text-sm text-muted">
        <p className="flex items-center gap-2">
          <ShieldCheck className="size-4 shrink-0 text-accent" aria-hidden />
          Contact details are removed before any AI sees your CV. Never pay to apply for a job.
        </p>
      </div>
    </footer>
  );
}

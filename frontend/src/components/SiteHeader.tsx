import { ScanSearch } from "lucide-react";
import Link from "next/link";
import { Suspense } from "react";

import { NavLinks } from "./NavLinks";

/** Static header. Only the links read the URL (to highlight the current page), so only
 * they wait for it, inside <Suspense>, as Cache Components requires. */
export function SiteHeader() {
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-surface/80 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
        <Link href="/" className="flex items-center gap-2" aria-label="JobScout home">
          <span className="grid size-8 place-items-center rounded-lg bg-gradient-to-br from-accent to-accent-2 text-accent-ink">
            <ScanSearch className="size-4.5" aria-hidden />
          </span>
          <span className="font-display text-xl font-bold tracking-tight">
            Job<span className="text-gradient">Scout</span>
          </span>
        </Link>
        <Suspense fallback={<NavLinks current={null} />}>
          <NavLinks />
        </Suspense>
      </div>
    </header>
  );
}

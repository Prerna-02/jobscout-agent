"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Find jobs" },
  { href: "/about", label: "About" },
];

/** Header links. Pass `current={null}` for the static placeholder (nothing highlighted). */
export function NavLinks({ current }: { current?: string | null }) {
  return current === null ? <Links path={null} /> : <LiveLinks />;
}

function LiveLinks() {
  return <Links path={usePathname()} />;
}

function Links({ path }: { path: string | null }) {
  return (
    <nav className="flex items-center gap-1 text-sm">
      {LINKS.map(({ href, label }) => {
        const active = path !== null && (href === "/" ? path === "/" : path.startsWith(href));
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={`rounded-lg px-3 py-1.5 transition-colors ${
              active ? "bg-accent-soft font-medium text-accent" : "text-muted hover:text-ink"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}

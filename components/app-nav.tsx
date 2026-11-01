"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BookOpen, Code, Database, FileSearch, Grid3x3, House, MessageSquareWarning } from "lucide-react";

import { DATASET_URL, REPO_URL } from "@/lib/study";
import { cn } from "@/lib/ui";

import { ThemeToggle } from "./theme-toggle";

const GROUPS = [
  { label: "Result", links: [{ href: "/", label: "Overview", icon: House }] },
  {
    label: "Evidence",
    links: [
      { href: "/task-prior", label: "The task comes first", icon: Grid3x3 },
      { href: "/retrieval", label: "Retrieval within a task", icon: FileSearch },
      { href: "/pushback", label: "Pushback and the judge", icon: MessageSquareWarning },
    ],
  },
  { label: "Rigor", links: [{ href: "/method", label: "Method and limits", icon: BookOpen }] },
];

export function AppNav() {
  const pathname = usePathname().replace(/\/$/, "") || "/";

  return (
    <nav className="sticky top-0 z-10 flex flex-col border-b border-border bg-surface-1 md:h-screen md:w-[220px] md:shrink-0 md:border-r md:border-b-0">
      <div className="flex items-start gap-2 px-4 py-3 md:py-4">
        <Link href="/" className="min-w-0">
          <div className="text-[13.5px] leading-tight font-semibold tracking-tight">Agentic Failure Attribution</div>
          <div className="mt-0.5 text-[11px] leading-snug text-text-muted">What agent traces do and don&apos;t tell you</div>
        </Link>
        <span className="ml-auto">
          <ThemeToggle />
        </span>
      </div>

      <div className="flex gap-1 overflow-x-auto px-2 pb-2 md:block md:flex-1 md:overflow-visible md:pb-0">
        {GROUPS.map((g) => (
          <div key={g.label} className="contents md:block">
            <div className="label-caps hidden px-2 pt-4 pb-1 md:block">{g.label}</div>
            <ul className="contents md:flex md:flex-col md:gap-0.5">
              {g.links.map(({ href, label, icon: Icon }) => {
                const active = pathname === href;
                return (
                  <li key={href} className="shrink-0">
                    <Link
                      href={href}
                      aria-current={active ? "page" : undefined}
                      className={cn(
                        "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] whitespace-nowrap transition-colors",
                        active ? "bg-surface-3 text-text-primary" : "text-text-secondary hover:bg-surface-2 hover:text-text-primary",
                      )}
                    >
                      <Icon size={15} strokeWidth={1.75} className="shrink-0" />
                      {label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>

      <div className="hidden space-y-1.5 border-t border-border px-4 py-3 text-[12px] md:block">
        <a href={REPO_URL} className="flex items-center gap-2 text-text-secondary hover:text-text-primary">
          <Code size={14} strokeWidth={1.75} /> Code on GitHub
        </a>
        <a href={DATASET_URL} className="flex items-center gap-2 text-text-secondary hover:text-text-primary">
          <Database size={14} strokeWidth={1.75} /> Data on Hugging Face
        </a>
      </div>
    </nav>
  );
}

"use client";

import { Moon, Sun } from "lucide-react";

import { THEME_KEY, type Theme } from "@/lib/theme";

/** Both icons are rendered and CSS shows the right one, so the static HTML never guesses the theme. */
export function ThemeToggle() {
  function toggle() {
    const root = document.documentElement;
    const resolved: Theme =
      (root.dataset.theme as Theme | undefined) ??
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next: Theme = resolved === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // storage blocked: the choice lasts for this page view only
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle light and dark theme"
      title="Toggle light and dark theme"
      className="flex size-6 items-center justify-center rounded-md border border-border text-text-secondary transition-colors hover:bg-surface-2 hover:text-text-primary"
    >
      <span className="when-dark">
        <Sun size={13} strokeWidth={1.75} />
      </span>
      <span className="when-light">
        <Moon size={13} strokeWidth={1.75} />
      </span>
    </button>
  );
}

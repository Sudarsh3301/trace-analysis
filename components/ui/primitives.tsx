import type { ReactNode } from "react";

import { cn } from "@/lib/ui";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="flex items-start justify-between gap-6 border-b border-border px-6 py-4">
      <div className="min-w-0">
        <h1 className="text-[20px] leading-snug font-semibold tracking-tight md:text-[24px]">{title}</h1>
        {description && (
          <p className="mt-1 max-w-2xl text-[12.5px] leading-relaxed text-text-secondary">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "default" | "muted";
}) {
  return (
    <div className="panel px-3.5 py-3">
      <div className="label-caps">{label}</div>
      <div
        className={cn(
          "num mt-1.5 text-[22px] leading-none font-medium tracking-tight",
          tone === "muted" && "text-text-muted",
        )}
      >
        {value}
      </div>
      {hint && <div className="mt-1.5 text-[11px] leading-snug text-text-muted">{hint}</div>}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warning" | "critical" | "accent";
  className?: string;
}) {
  const tones = {
    neutral: "border-border bg-surface-2 text-text-secondary",
    good: "border-transparent bg-[color-mix(in_oklab,var(--color-status-good)_20%,transparent)] text-[var(--color-status-good)]",
    warning:
      "border-transparent bg-[color-mix(in_oklab,var(--color-status-warning)_20%,transparent)] text-[var(--color-status-warning)]",
    critical:
      "border-transparent bg-[color-mix(in_oklab,var(--color-status-critical)_20%,transparent)] text-[var(--color-status-critical)]",
    accent:
      "border-transparent bg-[color-mix(in_oklab,var(--color-accent)_20%,transparent)] text-[var(--color-accent)]",
  } as const;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10.5px] font-medium tracking-wide whitespace-nowrap",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Button({
  children,
  variant = "secondary",
  size = "md",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
}) {
  const variants = {
    primary:
      "bg-[var(--color-accent)] text-white hover:brightness-110 border-transparent disabled:opacity-40",
    secondary:
      "bg-surface-2 text-text-primary border-border hover:bg-surface-3 disabled:opacity-40",
    ghost:
      "bg-transparent text-text-secondary border-transparent hover:bg-surface-2 hover:text-text-primary disabled:opacity-40",
  } as const;

  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-[background-color,filter,opacity] disabled:cursor-not-allowed",
        size === "sm" ? "px-2 py-1 text-[11.5px]" : "px-2.5 py-1.5 text-[12.5px]",
        variants[variant],
        className,
      )}
    >
      {children}
    </button>
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="panel flex flex-col items-center justify-center gap-2 px-6 py-14 text-center">
      <p className="text-[14px] font-medium">{title}</p>
      <p className="max-w-md text-[12.5px] leading-relaxed text-text-secondary">{description}</p>
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

/**
 * Renders untrusted text (agent content, tool output) as text.
 *
 * React escapes by default; the point of this component is that it is the only
 * sanctioned way to show ingested content, so a future `dangerouslySetInnerHTML`
 * has nowhere natural to appear.
 */
export function RawText({ value, className }: { value: string; className?: string }) {
  return (
    <pre
      className={cn(
        "font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap break-words text-text-secondary",
        className,
      )}
    >
      {value}
    </pre>
  );
}

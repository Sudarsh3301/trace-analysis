import type { ReactNode } from "react";

import { PageHeader } from "./primitives";

/** A page: header, then content. The document scrolls; the sidebar stays put on wide screens. */
export function Page({ title, description, children }: { title: string; description?: ReactNode; children: ReactNode }) {
  return (
    <>
      <PageHeader title={title} description={description} />
      <div className="mx-auto max-w-[1100px] space-y-5 px-4 py-5 md:px-6">{children}</div>
    </>
  );
}

/** A titled panel: one chart or one argument. */
export function Section({
  id,
  title,
  description,
  children,
}: {
  id?: string;
  title: string;
  description?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="panel">
      <header className="border-b border-border px-4 py-3">
        <h2 className="text-[14px] font-semibold tracking-tight">{title}</h2>
        {description && <p className="mt-0.5 text-[12px] leading-relaxed text-text-secondary">{description}</p>}
      </header>
      <div className="px-4 py-4">{children}</div>
    </section>
  );
}

/** Small print under a chart: how to read it, and what it does not show. */
export function Note({ children }: { children: ReactNode }) {
  return <p className="text-[11.5px] leading-relaxed text-text-muted">{children}</p>;
}

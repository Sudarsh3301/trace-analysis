import type { Metadata } from "next";

import { AppNav } from "@/components/app-nav";
import { THEME_KEY } from "@/lib/theme";

import "./globals.css";

export const metadata: Metadata = {
  title: "Agentic Failure Attribution",
  description:
    "What agent traces do and don't tell you: trace signals tested against outcomes on 7,760 τ-bench conversations.",
};

// Runs before first paint, so a saved choice never flashes the wrong theme. Without one, the OS setting applies (CSS).
const THEME_SCRIPT = `try{var t=localStorage.getItem("${THEME_KEY}");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-screen">
        <div className="md:flex">
          <AppNav />
          <main className="min-w-0 flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}

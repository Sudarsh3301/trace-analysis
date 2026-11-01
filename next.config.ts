import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A static site: `next build` writes plain HTML, JS and JSON to out/. No server, database or key at runtime.
  output: "export",
  // Set by the GitHub Pages workflow, where the site lives under /<repo>/. Empty for local builds.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? "",
  // Pins the workspace root so Turbopack does not walk up to the home directory looking for a lockfile.
  turbopack: { root: dirname(fileURLToPath(import.meta.url)) },
};

export default nextConfig;

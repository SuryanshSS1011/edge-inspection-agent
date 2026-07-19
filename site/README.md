# Tollgate project site

The Tollgate project homepage. A static Vite + React page that reads the real eval numbers
from `src/data.ts` (copied from the repo's `eval/*.md` output) and renders the thesis, the
interactive escalation-band widget, results, ablation, the offline story, and get-started.

## Develop

```bash
cd site
npm install
npm run dev        # http://localhost:5173
```

## Deploy to GitHub Pages

The site is served at `https://<user>.github.io/edge-inspection-agent/`, so `vite.config.ts`
sets `base` to `/edge-inspection-agent/` for production builds.

```bash
npm run deploy     # builds, then pushes dist/ to the gh-pages branch
```

Then in the GitHub repo: Settings -> Pages -> Source: **Deploy from a branch**, branch
**gh-pages** / root. `predeploy` copies `index.html` to `404.html` (SPA fallback) and adds
`.nojekyll` so Vite's hashed assets are served as-is.

## Updating the numbers

All figures live in `src/data.ts` and are copied from the repo's real eval output
(`eval/results_table_real.md`, `results_multi.md`, `backbone_ablation.md`). Re-run the eval
scripts, then update `src/data.ts` to match. No number on the page is invented.

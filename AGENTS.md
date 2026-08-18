# Repository guidance

## Product contract

Skyline Type Studio is an anonymous, browser-only poster editor. Preserve these
invariants in every change:

- Process photos locally; network requests may fetch model files, never user
  images.
- Render typography with HTML canvas and masks. This is a compositing tool, not
  an image-generation product.
- Keep the editor usable without an account, server, database, or secret.
- Preserve full-resolution PNG export even when analysis uses a smaller canvas.

## Architecture

`app/page.tsx` owns segmentation, depth-layer construction, canvas rendering,
and the editor UI. `app/globals.css` owns the visual system. `static-app/` is a
thin Vite entry that reuses those files for GitHub Pages; avoid duplicating the
editor there. The Vinext build remains the source for the existing Sites
deployment.

## Change workflow

Run `npm run lint`, `npm test`, and `npm run build:pages` after code changes.
Treat a successful build as an intermediate check: inspect mask boundaries at
text intersections and verify the downloadable PNG at source resolution when
rendering behavior changes. Keep generated `dist/` and `pages-dist/` artifacts
out of commits.

GitHub Pages deploys `pages-dist/` from `main` through
`.github/workflows/deploy-pages.yml`. Keep relative asset paths working under a
repository subpath.

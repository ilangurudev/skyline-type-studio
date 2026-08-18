# Skyline Type Studio

Create cinematic, layered poster typography from any photograph. Skyline Type
Studio segments a photo into semantic and depth-aware layers, then lets you put
multiline text in front of or behind the layers you choose.

Everything runs locally in the browser. Photos are not uploaded to a server,
and there is no account or sign-in.

## Run locally

Requires Node.js 22 or newer.

```bash
npm install
npm run dev
```

Open the local URL printed by the development server. The first segmentation
can take a little longer because the browser downloads and caches the AI models.

## Control it with code

The local CLI runs the same browser analysis and canvas renderer without opening
the editor UI. Install its managed browser once:

```bash
npm run studio:install-browser
```

Analyze a photograph once and create an editable work directory:

```bash
npm run studio -- init --input /path/to/photo.jpg --work work/my-poster
npm run studio -- inspect --work work/my-poster
```

Edit `work/my-poster/recipe.json` with code, then request compact verification
renders. Previews are 768 px on the long edge and WebP quality 0.82; `--overlay`
adds the detected depth colors for mask review.

```bash
npm run studio -- preview --work work/my-poster
npm run studio -- preview --work work/my-poster --overlay
```

The preview response includes a `renderId`. Once that exact preview is approved,
use it to unlock a source-resolution PNG and a reusable project file:

```bash
npm run studio -- export \
  --work work/my-poster \
  --approved <renderId> \
  --output outputs/poster.png \
  --project outputs/poster.skyline.cfg
```

The `.skyline.cfg` archive contains the recipe and cached masks, but not the
source photograph. Import it in the visual editor and choose the matching photo
to restore the composition without rerunning analysis. CLI results are JSON on
stdout; progress and diagnostics are written to stderr.

## Build and verify

```bash
npm run lint
npm test
npm run build:pages
```

`npm run build:pages` creates the static GitHub Pages site in `pages-dist/`.
Pushes to `main` deploy it using `.github/workflows/deploy-pages.yml`.

## How it works

- Semantic segmentation identifies sky, terrain, vegetation, architecture,
  water, and people.
- Monocular depth estimation separates large regions into near, middle, and far
  layers.
- Canvas compositing renders the selected scene layers over or under the text.
- The finished poster downloads as a full-resolution PNG.

The browser loads quantized SegFormer and Depth Anything V2 models from Hugging
Face on demand. No image-generation model is used.

## License

This project has not yet been assigned an open-source license. All rights are
reserved unless a license is added later.

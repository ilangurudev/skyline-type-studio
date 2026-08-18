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

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the Skyline Type Studio editor", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Skyline Type Studio<\/title>/i);
  assert.match(html, /Choose a photograph/i);
  assert.match(html, /Text layers/i);
  assert.match(html, /Add new text layer/i);
  assert.match(html, /Select a layer to edit its type and position/i);
  assert.match(html, /3D extrusion/i);
  assert.match(html, /Depth order/i);
  assert.match(html, /Every image and text layer, ordered from deepest to closest/i);
  assert.match(html, /Show colored layer overlay/i);
  assert.match(html, /Models download once/i);
  assert.match(html, /Download PNG \+ project/i);
  assert.match(html, /Animation timeline/i);
  assert.match(html, /Modern Reel/i);
  assert.match(html, /Slow Cinema/i);
  assert.match(html, /Editorial Flash/i);
  assert.match(html, /Opening screen/i);
  assert.match(html, /Download animated WebM/i);
  assert.match(html, /Import \.skyline\.cfg project/i);
  assert.doesNotMatch(html, /Horizon guide/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/i);
});

test("animates the base photo, depth planes, and text on one timeline", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const renderer = await readFile(new URL("../studio/render.ts", import.meta.url), "utf8");
  assert.match(source, /timeline\.baseAnimation/);
  assert.match(source, /timeline\.sceneAnimations/);
  assert.match(source, /layer\.animation/);
  assert.match(source, /MediaRecorder/);
  assert.match(source, /captureStream\(30\)/);
  assert.match(renderer, /timeline\.backgroundColor/);
  assert.match(renderer, /maskedImage\(image, layer/);
  assert.match(renderer, /animationFrame\(textLayer\.animation/);
  assert.match(renderer, /animationFinished\(timeline\.baseAnimation/);
  assert.match(renderer, /if \(sceneSettled\) ctx\.drawImage\(image/);
  assert.match(renderer, /frame\.blur/);
  assert.match(source, /applyModernReel/);
  assert.match(source, /applySlowCinema/);
  assert.match(source, /applyEditorialFlash/);
  assert.match(source, /backgroundColor: "#ffffff"/);
});

test("wires semantic masks to depth-aware layer splitting", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(source, /onnx-community\/depth-anything-v2-small/);
  assert.match(source, /createDepthLayers\(groupedMasks, depthMap\)/);
  assert.match(source, /Mountain|Terrain/);
  assert.match(source, /Depth-enhanced analysis/);
});

test("renders and depth-masks every text layer independently", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const renderer = await readFile(new URL("../studio/render.ts", import.meta.url), "utf8");
  assert.match(source, /TextAlign, TextLayer/);
  assert.match(source, /renderPoster\(\{/);
  assert.match(renderer, /for \(const textLayer of textLayers\)/);
  assert.match(renderer, /frontLayers = semanticLayers\.filter/);
  assert.match(renderer, /createMaskCanvas\(layer\).*destination-out/);
  assert.match(renderer, /globalCompositeOperation = "destination-out"/);
  assert.match(source, /addTextLayer/);
  assert.match(renderer, /extrusionLength/);
  assert.match(renderer, /textLayer\.extrusionColor/);
  assert.match(source, /createDepthStack/);
  assert.match(source, /Complete layer depth order/);
  assert.match(source, /draggable onDragStart/);
  assert.match(source, /drag-handle/);
  assert.match(source, /Deepest.*Closest/);
});

test("new text layers use the shared 15 percent default and expanded font catalog", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const types = await readFile(new URL("../studio/types.ts", import.meta.url), "utf8");
  assert.match(source, /createTextLayer/);
  assert.match(source, /FONT_OPTIONS/);
  assert.doesNotMatch(source, /const FONT_OPTIONS =/);
  assert.match(types, /fontSize: 15/);
  assert.match(types, /\["Avenir Next"/);
  assert.match(types, /\["Bodoni 72"/);
  assert.match(types, /\["Menlo"/);
  assert.ok((types.match(/^ {2}\["/gm) ?? []).length >= 30);
});

test("new projects default to a complete three-second animation", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const types = await readFile(new URL("../studio/types.ts", import.meta.url), "utf8");
  assert.match(types, /DEFAULT_TIMELINE_DURATION = 3000/);
  assert.match(types, /duration: DEFAULT_TIMELINE_DURATION/);
  assert.match(source, /useState\(DEFAULT_TIMELINE_DURATION\)/);
  assert.match(types, /Math\.min\(2100, 1800 \+ \(index - 1\) \* 120\)/);
});

test("ships an agent-native preview and approved-export contract", async () => {
  const cli = await readFile(new URL("../tools/studio-cli.mjs", import.meta.url), "utf8");
  const schema = JSON.parse(await readFile(new URL("../studio/recipe.schema.json", import.meta.url), "utf8"));
  assert.match(cli, /command === "init"/);
  assert.match(cli, /command === "inspect"/);
  assert.match(cli, /command === "preview"/);
  assert.match(cli, /command === "export"/);
  assert.match(cli, /command === "animate"/);
  assert.match(cli, /maxDimension: 768/);
  assert.match(cli, /options\.time.*Number\(options\.time\) \* 1000/);
  assert.match(cli, /--time must be a number of seconds/);
  assert.match(cli, /Approval is stale/);
  assert.match(cli, /Animated export dimensions do not match/);
  assert.equal(schema.properties.schemaVersion.const, 1);
  assert.equal(schema.properties.textLayers.minItems, 1);
  assert.deepEqual(schema.$defs.animation.properties.effect.enum, ["fade", "rise", "drift", "zoom", "reel"]);
  assert.deepEqual(schema.properties.timeline.properties.backgroundColor.enum, ["#000000", "#ffffff"]);
});

test("project archives retain cached masks without embedding the photograph", async () => {
  const project = await readFile(new URL("../studio/project.ts", import.meta.url), "utf8");
  assert.match(project, /"project\.json"/);
  assert.match(project, /serializeAnalysis/);
  assert.match(project, /sha256File/);
  assert.doesNotMatch(project, /sourceImage|imageBytes|originalPhoto/);
});

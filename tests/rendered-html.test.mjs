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
  assert.match(html, /<title>Skyline Reel Studio<\/title>/i);
  assert.match(html, /Skyline Reel Studio/i);
  assert.match(html, /Add photographs/i);
  assert.match(html, /Reel scenes/i);
  assert.match(html, /Text layers/i);
  assert.match(html, /Add new text layer/i);
  assert.match(html, /Select a layer to edit its type and position/i);
  assert.match(html, /3D extrusion/i);
  assert.match(html, /Depth order/i);
  assert.match(html, /Every image and text layer, ordered from deepest to closest/i);
  assert.match(html, /Show colored layer overlay/i);
  assert.match(html, /Photos and soundtracks stay in this browser/i);
  assert.match(html, /Download PNG \+ project/i);
  assert.match(html, /Scene animation/i);
  assert.match(html, /Insta Edit/i);
  assert.match(html, /Slow Cinema/i);
  assert.match(html, /Editorial Flash/i);
  assert.match(html, /Opening screen/i);
  assert.match(html, /Download animated WebM/i);
  assert.match(html, /Download full reel WebM/i);
  assert.match(html, /Soundtrack/i);
  assert.match(html, /Sync layer entrances to the beat/i);
  assert.match(html, /Play full reel/i);
  assert.match(html, /Vertical · 9:16/i);
  assert.match(html, /Photo · 2:3 \(4×6\)/i);
  assert.match(html, /Import \.skyline\.cfg project/i);
  assert.doesNotMatch(html, /Horizon guide/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/i);
});

test("composes multiple scenes with local beat detection and soundtrack export", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const reel = await readFile(new URL("../studio/reel.ts", import.meta.url), "utf8");
  const renderer = await readFile(new URL("../studio/reel-render.ts", import.meta.url), "utf8");
  assert.match(source, /multiple hidden onChange=\{handleFile\}/);
  assert.match(source, /scenesRef/);
  assert.match(source, /handleSoundtrack/);
  assert.match(source, /decodeAudioData/);
  assert.match(source, /createMediaStreamDestination/);
  assert.match(source, /Select soundtrack section/);
  assert.match(source, /Soundtrack section start/);
  assert.match(source, /soundtrackOffset = soundtrackStart \/ 1000/);
  assert.match(source, /audioSource\?\.start\(0, soundtrackOffset/);
  assert.match(source, /video\/webm;codecs=vp9,opus/);
  assert.match(source, /downloadReel/);
  assert.match(reel, /export function detectBeats/);
  assert.match(reel, /export function buildReelTimeline/);
  assert.match(reel, /export function syncSceneToBeats/);
  assert.match(reel, /nearestBeat/);
  assert.match(renderer, /renderReelFrame/);
  assert.match(renderer, /drawFrame/);
  assert.match(renderer, /outputDimensions/);
  assert.match(renderer, /transitionProgress/);
  assert.match(renderer, /renderPoster/);
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
  assert.match(renderer, /if \(sceneSettled\) \{/);
  assert.match(renderer, /if \(cover\) drawCover\(ctx, image/);
  assert.match(renderer, /frame\.blur/);
  assert.match(source, /applyInstaEdit/);
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
  assert.match(renderer, /fittedMask\(layer.*destination-out/);
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

test("new scenes start with an empty text layer", async () => {
  const types = await readFile(new URL("../studio/types.ts", import.meta.url), "utf8");
  assert.match(types, /text: index === 1 \? "" : "NEW TEXT"/);
  assert.doesNotMatch(types, /ONE DESERT/);
});

test("new projects default to a complete three-second animation", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const types = await readFile(new URL("../studio/types.ts", import.meta.url), "utf8");
  assert.match(types, /DEFAULT_TIMELINE_DURATION = 3000/);
  assert.match(types, /duration: DEFAULT_TIMELINE_DURATION/);
  assert.match(source, /useState\(DEFAULT_TIMELINE_DURATION\)/);
  assert.match(types, /Math\.min\(2100, 1800 \+ \(index - 1\) \* 120\)/);
});

test("every photo-cut preset defaults to three seconds", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(source, /function createInstaEditTimeline/);
  assert.match(source, /duration: DEFAULT_TIMELINE_DURATION/);
  assert.equal((source.match(/const duration = DEFAULT_TIMELINE_DURATION;/g) ?? []).length, 2);
  assert.match(source, /Social · 3 sec/);
  assert.match(source, /Film title · 3 sec/);
  assert.match(source, /Editorial · 3 sec/);
  assert.doesNotMatch(source, /Social · 5 sec|Film title · 8 sec/);
});

test("uploads auto-analyze photos sequentially with Insta Edit as the default", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(source, /const analyzeScenesSequentially = async/);
  assert.match(source, /for \(const \[index, scene\] of pending\.entries\(\)\)/);
  assert.match(source, /await analyzeScene\(scene\)/);
  assert.match(source, /await analyzeScenesSequentially\(added\)/);
  assert.match(source, /timeline: createInstaEditTimeline\(\)/);
  assert.match(source, /scene\.timeline = createInstaEditTimeline\(layerIds\)/);
  assert.match(source, /applyInstaEditTextAnimations\(\[createTextLayer/);
  assert.match(source, /queued for analysis/);
  assert.doesNotMatch(source, /select to analyze/);
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

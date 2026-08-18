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
  assert.match(html, /Poster text/i);
  assert.match(html, /Depth layers/i);
  assert.match(html, /Semantic objects are split into near, middle, and far planes/i);
  assert.match(html, /Show colored layer overlay/i);
  assert.match(html, /Models download once/i);
  assert.match(html, /Download full-resolution PNG/i);
  assert.doesNotMatch(html, /Horizon guide/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/i);
});

test("wires semantic masks to depth-aware layer splitting", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(source, /onnx-community\/depth-anything-v2-small/);
  assert.match(source, /createDepthLayers\(groupedMasks, depthMap\)/);
  assert.match(source, /Mountain|Terrain/);
  assert.match(source, /Depth-enhanced analysis/);
});

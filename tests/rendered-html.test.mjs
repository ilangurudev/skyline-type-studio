import assert from "node:assert/strict";
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
  assert.match(html, /Choose which detected objects cover the text/i);
  assert.match(html, /Show colored layer overlay/i);
  assert.match(html, /Download full-resolution PNG/i);
  assert.doesNotMatch(html, /Horizon guide/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/i);
});

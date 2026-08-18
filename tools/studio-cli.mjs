#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import process from "node:process";
import { zipSync, strToU8 } from "fflate";
import { chromium } from "playwright";
import { createServer } from "vite";

const projectRoot = resolve(import.meta.dirname, "..");

function fail(message, code = "STUDIO_ERROR") {
  process.stderr.write(`${message}\n`);
  process.stdout.write(`${JSON.stringify({ ok: false, code, message })}\n`);
  process.exitCode = 1;
}

function parseArgs(args) {
  const [command, ...rest] = args;
  const options = {};
  for (let index = 0; index < rest.length; index += 1) {
    const token = rest[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    if (key === "overlay") options[key] = true;
    else {
      const value = rest[++index];
      if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}.`);
      options[key] = value;
    }
  }
  return { command, options };
}

function required(options, name) {
  if (!options[name]) throw new Error(`--${name} is required.`);
  return options[name];
}

async function sha256Path(path) {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

function renderId(source, analysis, recipe) {
  return createHash("sha256").update(source.sha256).update(JSON.stringify(analysis)).update(JSON.stringify(recipe)).digest("hex").slice(0, 24);
}

async function readWork(workOption) {
  const work = resolve(required({ work: workOption }, "work"));
  const [source, analysis, recipe] = await Promise.all([
    readFile(join(work, "source.json"), "utf8").then(JSON.parse),
    readFile(join(work, "analysis.json"), "utf8").then(JSON.parse),
    readFile(join(work, "recipe.json"), "utf8").then(JSON.parse),
  ]);
  const currentHash = await sha256Path(source.path);
  if (currentHash !== source.sha256) throw new Error("The source photograph no longer matches this work directory.");
  return { work, source, analysis, recipe };
}

async function withRunner(sourcePath, action) {
  const server = await createServer({ root: join(projectRoot, "static-app"), resolve: { alias: { "@": projectRoot } }, server: { host: "127.0.0.1", port: 0 }, logLevel: "error" });
  let browser;
  try {
    await server.listen();
    const url = server.resolvedUrls?.local?.[0];
    if (!url) throw new Error("The local automation runner did not start.");
    try {
      browser = await chromium.launch({ headless: true });
    } catch (error) {
      if (String(error).includes("Executable doesn't exist")) throw new Error("Playwright Chromium is not installed. Run: npm run studio:install-browser");
      throw error;
    }
    const page = await browser.newPage();
    await page.goto(new URL("automation.html", url).href);
    await page.waitForSelector("html[data-ready='true']");
    await page.setInputFiles("#automation-source", sourcePath);
    return await action(page);
  } finally {
    await browser?.close();
    await server.close();
  }
}

async function saveRender(page, output, analysis, recipe, options) {
  await mkdir(dirname(output), { recursive: true });
  const downloadPromise = page.waitForEvent("download");
  const dimensionsPromise = page.evaluate(([cached, currentRecipe, renderOptions]) => window.skylineAutomation.render(cached, currentRecipe, renderOptions), [analysis, recipe, { ...options, filename: basename(output) }]);
  const download = await downloadPromise;
  const dimensions = await dimensionsPromise;
  await download.saveAs(output);
  return dimensions;
}

async function saveAnimation(page, output, analysis, recipe, options) {
  await mkdir(dirname(output), { recursive: true });
  const downloadPromise = page.waitForEvent("download");
  const resultPromise = page.evaluate(([cached, currentRecipe, animationOptions]) => window.skylineAutomation.animate(cached, currentRecipe, animationOptions), [analysis, recipe, { ...options, filename: basename(output) }]);
  const download = await downloadPromise;
  const result = await resultPromise;
  await download.saveAs(output);
  return result;
}

async function init(options) {
  const input = resolve(required(options, "input"));
  const work = resolve(required(options, "work"));
  const file = await stat(input);
  if (!file.isFile()) throw new Error("--input must identify a photograph file.");
  process.stderr.write("Analyzing the photograph locally…\n");
  const result = await withRunner(input, (page) => page.evaluate(() => window.skylineAutomation.analyze()));
  const source = { path: input, name: basename(input), size: file.size, width: result.width, height: result.height, sha256: await sha256Path(input) };
  await mkdir(work, { recursive: true });
  await Promise.all([
    writeFile(join(work, "source.json"), `${JSON.stringify(source, null, 2)}\n`),
    writeFile(join(work, "analysis.json"), `${JSON.stringify(result.analysis)}\n`),
    writeFile(join(work, "recipe.json"), `${JSON.stringify(result.recipe, null, 2)}\n`),
  ]);
  process.stdout.write(`${JSON.stringify({ ok: true, command: "init", work, source: { ...source, path: undefined }, layers: result.analysis.layers.map(({ id, label, coverage, depthScore }) => ({ id, label, coverage, depthScore })) })}\n`);
}

async function inspect(options) {
  const state = await readWork(required(options, "work"));
  process.stdout.write(`${JSON.stringify({ ok: true, command: "inspect", renderId: renderId(state.source, state.analysis, state.recipe), source: { ...state.source, path: undefined }, quality: state.analysis.quality, layers: state.analysis.layers.map(({ id, label, coverage, depthScore }) => ({ id, label, coverage, depthScore })), recipe: state.recipe })}\n`);
}

async function preview(options) {
  const state = await readWork(required(options, "work"));
  const output = resolve(options.output || join(state.work, options.overlay ? "preview-overlay.webp" : "preview.webp"));
  const id = renderId(state.source, state.analysis, state.recipe);
  const time = options.time === undefined ? undefined : Number(options.time) * 1000;
  if (time !== undefined && (!Number.isFinite(time) || time < 0 || time > 60000)) throw new Error("--time must be a number of seconds from 0 to 60.");
  const dimensions = await withRunner(state.source.path, (page) => saveRender(page, output, state.analysis, state.recipe, { format: "webp", maxDimension: 768, overlay: Boolean(options.overlay), time }));
  const report = { ok: true, command: "preview", renderId: id, output, format: "webp", quality: 0.82, ...dimensions, sourceWidth: state.source.width, sourceHeight: state.source.height, overlay: Boolean(options.overlay), time: time === undefined ? null : time };
  await writeFile(join(state.work, "preview-report.json"), `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(report)}\n`);
}

async function exportProject(options) {
  const state = await readWork(required(options, "work"));
  const approved = required(options, "approved");
  const currentId = renderId(state.source, state.analysis, state.recipe);
  if (approved !== currentId) throw new Error(`Approval is stale. Preview the current recipe and approve render ${currentId}.`);
  const output = resolve(required(options, "output"));
  const projectPath = resolve(required(options, "project"));
  process.stderr.write("Rendering the approved poster at source resolution…\n");
  const dimensions = await withRunner(state.source.path, (page) => saveRender(page, output, state.analysis, state.recipe, { format: "png", overlay: false }));
  if (dimensions.width !== state.source.width || dimensions.height !== state.source.height) throw new Error("Full export dimensions do not match the source photograph.");
  const project = { schemaVersion: 1, createdAt: new Date().toISOString(), source: { name: state.source.name, size: state.source.size, width: state.source.width, height: state.source.height, sha256: state.source.sha256 }, analysis: state.analysis, recipe: state.recipe };
  await mkdir(dirname(projectPath), { recursive: true });
  await writeFile(projectPath, zipSync({ "project.json": strToU8(JSON.stringify(project)) }, { level: 6 }));
  process.stdout.write(`${JSON.stringify({ ok: true, command: "export", renderId: currentId, output, project: projectPath, ...dimensions })}\n`);
}

async function animate(options) {
  const state = await readWork(required(options, "work"));
  const approved = required(options, "approved");
  const currentId = renderId(state.source, state.analysis, state.recipe);
  if (approved !== currentId) throw new Error(`Approval is stale. Preview the current recipe and approve render ${currentId}.`);
  if (!state.recipe.timeline) throw new Error("The recipe does not contain an animation timeline.");
  const output = resolve(required(options, "output"));
  const fps = options.fps === undefined ? 30 : Number(options.fps);
  if (!Number.isInteger(fps) || fps < 12 || fps > 60) throw new Error("--fps must be an integer from 12 to 60.");
  process.stderr.write("Rendering the approved animation locally at source resolution…\n");
  const result = await withRunner(state.source.path, (page) => saveAnimation(page, output, state.analysis, state.recipe, { fps }));
  if (result.width !== state.source.width || result.height !== state.source.height) throw new Error("Animated export dimensions do not match the source photograph.");
  process.stdout.write(`${JSON.stringify({ ok: true, command: "animate", renderId: currentId, output, ...result })}\n`);
}

async function main() {
  const { command, options } = parseArgs(process.argv.slice(2));
  if (command === "init") return init(options);
  if (command === "inspect") return inspect(options);
  if (command === "preview") return preview(options);
  if (command === "export") return exportProject(options);
  if (command === "animate") return animate(options);
  throw new Error("Usage: studio <init|inspect|preview|export|animate> [options]");
}

main().catch((error) => fail(error instanceof Error ? error.message : String(error)));

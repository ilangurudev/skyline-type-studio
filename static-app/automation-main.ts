import { analyzeImage } from "../studio/analysis";
import { deserializeAnalysis, serializeAnalysis } from "../studio/project";
import { renderPoster } from "../studio/render";
import { createRecipe, validateRecipe, type CachedAnalysis, type StudioRecipeV1 } from "../studio/types";

const input = document.querySelector<HTMLInputElement>("#automation-source")!;
let image: HTMLImageElement | null = null;

async function loadSource() {
  const file = input.files?.[0];
  if (!file) throw new Error("No source photograph was provided to the automation runner.");
  if (image) return image;
  const url = URL.createObjectURL(file);
  try {
    image = new Image();
    await new Promise<void>((resolve, reject) => {
      image!.onload = () => resolve();
      image!.onerror = () => reject(new Error("The source photograph could not be decoded."));
      image!.src = url;
    });
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function downloadCanvas(canvas: HTMLCanvasElement, filename: string, mimeType: string, quality?: number) {
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new Error("The browser could not encode the render.")), mimeType, quality));
  downloadBlob(blob, filename);
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

window.skylineAutomation = {
  async analyze() {
    const source = await loadSource();
    const result = await analyzeImage(source);
    const analysis = serializeAnalysis(result.quality, result.skyMask, result.layers);
    return {
      width: source.naturalWidth,
      height: source.naturalHeight,
      analysis,
      recipe: createRecipe(result.layers.map((layer) => layer.id)),
    };
  },
  async render(analysis: CachedAnalysis, recipe: StudioRecipeV1, options: { filename: string; maxDimension?: number; overlay?: boolean; format: "webp" | "png"; time?: number }) {
    validateRecipe(recipe);
    const unresolved = recipe.textLayers.flatMap((layer) => layer.frontLayerIds).filter((id) => !analysis.layers.some((candidate) => candidate.id === id));
    if (unresolved.length) throw new Error(`Recipe references unknown depth layers: ${[...new Set(unresolved)].join(", ")}`);
    const source = await loadSource();
    const hydrated = deserializeAnalysis(analysis);
    const canvas = document.createElement("canvas");
    const dimensions = renderPoster({
      target: canvas,
      image: source,
      textLayers: recipe.textLayers,
      activeTextLayerId: recipe.activeTextLayerId,
      semanticLayers: hydrated.layers,
      skyMask: hydrated.skyMask,
      maskOverlay: options.overlay,
      maxDimension: options.maxDimension,
      time: options.time,
      timeline: recipe.timeline,
    });
    await downloadCanvas(canvas, options.filename, options.format === "webp" ? "image/webp" : "image/png", options.format === "webp" ? 0.82 : undefined);
    return dimensions;
  },
  async animate(analysis: CachedAnalysis, recipe: StudioRecipeV1, options: { filename: string; fps?: number }) {
    validateRecipe(recipe);
    if (!recipe.timeline) throw new Error("The recipe does not contain an animation timeline.");
    const source = await loadSource();
    const hydrated = deserializeAnalysis(analysis);
    const canvas = document.createElement("canvas");
    const renderAt = (time: number) => renderPoster({
      target: canvas,
      image: source,
      textLayers: recipe.textLayers,
      activeTextLayerId: recipe.activeTextLayerId,
      semanticLayers: hydrated.layers,
      skyMask: hydrated.skyMask,
      time,
      timeline: recipe.timeline,
    });
    const dimensions = renderAt(0);
    const fps = Math.max(12, Math.min(60, options.fps ?? 30));
    const stream = canvas.captureStream(fps);
    const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 12_000_000 } : undefined);
    const finished = new Promise<Blob>((resolve, reject) => {
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = () => reject(new Error("The browser could not record the animation."));
      recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
    });
    recorder.start();
    const started = performance.now();
    await new Promise<void>((resolve) => {
      const tick = (now: number) => {
        const time = Math.min(recipe.timeline!.duration, now - started);
        renderAt(time);
        if (time < recipe.timeline!.duration) requestAnimationFrame(tick);
        else setTimeout(() => { recorder.stop(); resolve(); }, 50);
      };
      requestAnimationFrame(tick);
    });
    const blob = await finished;
    downloadBlob(blob, options.filename);
    return { ...dimensions, duration: recipe.timeline.duration, fps, mimeType: blob.type, size: blob.size };
  },
};

document.documentElement.dataset.ready = "true";

declare global {
  interface Window {
    skylineAutomation: {
      analyze(): Promise<{ width: number; height: number; analysis: CachedAnalysis; recipe: StudioRecipeV1 }>;
      render(analysis: CachedAnalysis, recipe: StudioRecipeV1, options: { filename: string; maxDimension?: number; overlay?: boolean; format: "webp" | "png"; time?: number }): Promise<{ width: number; height: number }>;
      animate(analysis: CachedAnalysis, recipe: StudioRecipeV1, options: { filename: string; fps?: number }): Promise<{ width: number; height: number; duration: number; fps: number; mimeType: string; size: number }>;
    };
  }
}

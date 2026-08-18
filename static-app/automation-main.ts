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
  async render(analysis: CachedAnalysis, recipe: StudioRecipeV1, options: { filename: string; maxDimension?: number; overlay?: boolean; format: "webp" | "png" }) {
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
    });
    await downloadCanvas(canvas, options.filename, options.format === "webp" ? "image/webp" : "image/png", options.format === "webp" ? 0.82 : undefined);
    return dimensions;
  },
};

document.documentElement.dataset.ready = "true";

declare global {
  interface Window {
    skylineAutomation: {
      analyze(): Promise<{ width: number; height: number; analysis: CachedAnalysis; recipe: StudioRecipeV1 }>;
      render(analysis: CachedAnalysis, recipe: StudioRecipeV1, options: { filename: string; maxDimension?: number; overlay?: boolean; format: "webp" | "png" }): Promise<{ width: number; height: number }>;
    };
  }
}

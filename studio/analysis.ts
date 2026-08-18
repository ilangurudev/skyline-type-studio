import type { BaseSemanticMask, BinaryMask, SemanticLayer } from "./types";

type Segment = { label?: string; mask: { width: number; height: number; data: ArrayLike<number> } };
type Segmenter = (input: HTMLCanvasElement) => Promise<Segment[]>;
type DepthOutput = { predicted_depth?: { dims?: number[]; data?: ArrayLike<number> }; depth?: { width: number; height: number; data: ArrayLike<number> } };
type DepthEstimator = (input: HTMLCanvasElement) => Promise<DepthOutput>;

const MODEL_ID = "Xenova/segformer-b0-finetuned-ade-512-512";
const DEPTH_MODEL_ID = "onnx-community/depth-anything-v2-small";
let segmenterPromise: Promise<Segmenter> | null = null;
let depthEstimatorPromise: Promise<DepthEstimator> | null = null;

const LAYER_COLORS: Array<[number, number, number]> = [
  [217, 255, 72], [255, 112, 84], [255, 194, 66], [162, 117, 255],
  [72, 224, 187], [255, 111, 193], [97, 155, 255], [231, 231, 90],
];

async function getSegmenter() {
  if (!segmenterPromise) {
    segmenterPromise = import("@huggingface/transformers").then(async ({ pipeline }) => {
      const model = await pipeline("image-segmentation", MODEL_ID, { dtype: "q8", device: "wasm" });
      return model as unknown as Segmenter;
    });
  }
  return segmenterPromise;
}

async function getDepthEstimator() {
  if (!depthEstimatorPromise) {
    depthEstimatorPromise = import("@huggingface/transformers").then(async ({ pipeline }) => {
      const model = await pipeline("depth-estimation", DEPTH_MODEL_ID, { dtype: "q8", device: "wasm" });
      return model as unknown as DepthEstimator;
    });
  }
  return depthEstimatorPromise;
}

function cleanSkyMask(input: ArrayLike<number>, width: number, height: number) {
  const sky = Uint8ClampedArray.from(input, (value) => (value > 127 ? 255 : 0));
  const visited = new Uint8Array(width * height);
  const queue = new Int32Array(width * height);
  const maxFloatingArea = Math.round(width * height * 0.035);
  for (let start = 0; start < sky.length; start += 1) {
    if (sky[start] || visited[start]) continue;
    let head = 0, tail = 0, minY = height, maxY = 0, touchesBottom = false;
    const component: number[] = [];
    queue[tail++] = start;
    visited[start] = 1;
    while (head < tail) {
      const index = queue[head++];
      component.push(index);
      const x = index % width;
      const y = Math.floor(index / width);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      if (y >= height - 2) touchesBottom = true;
      for (const next of [index - 1, index + 1, index - width, index + width]) {
        if (next < 0 || next >= sky.length || visited[next] || sky[next]) continue;
        if (Math.abs((next % width) - x) > 1) continue;
        visited[next] = 1; queue[tail++] = next;
      }
    }
    if (!touchesBottom && component.length <= maxFloatingArea && maxY - minY <= height * 0.34) {
      for (const index of component) sky[index] = 255;
    }
  }
  return sky;
}

function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function semanticGroup(label: string) {
  const value = label.toLowerCase();
  if (["mountain", "earth", "rock", "sand", "grass", "field", "road", "hill", "land"].includes(value)) return "Terrain";
  if (["tree", "plant", "palm", "flower", "bush"].includes(value)) return "Vegetation";
  if (["building", "skyscraper", "house", "wall", "tower", "bridge", "fence"].includes(value)) return "Architecture";
  if (["sea", "water", "river", "lake", "waterfall"].includes(value)) return "Water";
  if (["person", "people", "man", "woman"].includes(value)) return "People";
  return titleCase(label);
}

function groupSemanticMasks(baseMasks: BaseSemanticMask[]) {
  const groups = new Map<string, BaseSemanticMask>();
  for (const mask of baseMasks) {
    const label = semanticGroup(mask.label);
    const existing = groups.get(label);
    if (!existing) groups.set(label, { ...mask, label, data: new Uint8ClampedArray(mask.data) });
    else for (let index = 0; index < existing.data.length; index += 1) if (mask.data[index]) existing.data[index] = 255;
  }
  let grouped = [...groups.values()].map((mask, index) => summarizeMask({ ...mask, sourceIndex: index }));
  const terrain = grouped.find((mask) => mask.label === "Terrain");
  const water = grouped.find((mask) => mask.label === "Water");
  if (terrain && water) {
    const waterPixels = water.data.reduce((sum, value) => sum + (value ? 1 : 0), 0);
    const terrainPixels = terrain.data.reduce((sum, value) => sum + (value ? 1 : 0), 0);
    if (waterPixels / water.data.length < 0.018 && terrainPixels / terrain.data.length > 0.12) {
      for (let index = 0; index < terrain.data.length; index += 1) if (water.data[index]) terrain.data[index] = 255;
      grouped = grouped.filter((mask) => mask !== water);
    }
  }
  return grouped.map((mask, index) => summarizeMask({ ...mask, sourceIndex: index }));
}

function summarizeMask(mask: BaseSemanticMask) {
  let count = 0, yTotal = 0;
  for (let index = 0; index < mask.data.length; index += 1) {
    if (!mask.data[index]) continue;
    count += 1; yTotal += Math.floor(index / mask.width);
  }
  return { ...mask, averageY: count ? yTotal / count : 0 };
}

function resizeDepthMap(output: DepthOutput | null, width: number, height: number) {
  const tensor = output?.predicted_depth;
  const raw = tensor?.data ?? output?.depth?.data;
  if (!raw?.length) return null;
  const sourceHeight = tensor?.dims?.at(-2) ?? output?.depth?.height ?? height;
  const sourceWidth = tensor?.dims?.at(-1) ?? output?.depth?.width ?? width;
  const resized = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.round((y / Math.max(1, height - 1)) * (sourceHeight - 1)));
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.round((x / Math.max(1, width - 1)) * (sourceWidth - 1)));
      resized[y * width + x] = Number(raw[sourceY * sourceWidth + sourceX] ?? 0);
    }
  }
  return resized;
}

function smoothDepthMap(input: Float32Array, width: number, height: number) {
  const horizontal = new Float32Array(input.length);
  const output = new Float32Array(input.length);
  const radius = 2;
  for (let y = 0; y < height; y += 1) {
    let sum = 0;
    for (let x = -radius; x <= radius; x += 1) sum += input[y * width + Math.max(0, Math.min(width - 1, x))];
    for (let x = 0; x < width; x += 1) {
      horizontal[y * width + x] = sum / 5;
      sum -= input[y * width + Math.max(0, x - radius)];
      sum += input[y * width + Math.min(width - 1, x + radius + 1)];
    }
  }
  for (let x = 0; x < width; x += 1) {
    let sum = 0;
    for (let y = -radius; y <= radius; y += 1) sum += horizontal[Math.max(0, Math.min(height - 1, y)) * width + x];
    for (let y = 0; y < height; y += 1) {
      output[y * width + x] = sum / 5;
      sum -= horizontal[Math.max(0, y - radius) * width + x];
      sum += horizontal[Math.min(height - 1, y + radius + 1) * width + x];
    }
  }
  return output;
}

function percentile(sorted: number[], amount: number) {
  return sorted[Math.max(0, Math.min(sorted.length - 1, Math.round((sorted.length - 1) * amount)))] ?? 0;
}

function buildDepthClusters(depth: Float32Array | null, foreground: Uint8Array) {
  if (!depth) return null;
  const sample: number[] = [];
  const stride = Math.max(1, Math.floor(depth.length / 50000));
  for (let index = 0; index < depth.length; index += stride) if (foreground[index] && Number.isFinite(depth[index])) sample.push(depth[index]);
  if (sample.length < 100) return null;
  sample.sort((a, b) => a - b);
  const low = percentile(sample, 0.04), high = percentile(sample, 0.96);
  if (high - low < 1e-5) return null;
  const normalize = (value: number) => Math.max(0, Math.min(1, (value - low) / (high - low)));
  const normalizedSample = sample.map(normalize);
  let centers = [percentile(normalizedSample, 0.2), percentile(normalizedSample, 0.5), percentile(normalizedSample, 0.8)];
  for (let iteration = 0; iteration < 8; iteration += 1) {
    const sums = [0, 0, 0], counts = [0, 0, 0];
    for (const value of normalizedSample) {
      let closest = 0;
      if (Math.abs(value - centers[1]) < Math.abs(value - centers[closest])) closest = 1;
      if (Math.abs(value - centers[2]) < Math.abs(value - centers[closest])) closest = 2;
      sums[closest] += value; counts[closest] += 1;
    }
    centers = centers.map((center, index) => counts[index] ? sums[index] / counts[index] : center).sort((a, b) => a - b);
  }
  if (centers[2] - centers[0] < 0.16) return null;
  const normalized = Float32Array.from(depth, normalize);
  const bands = new Uint8Array(depth.length);
  for (let index = 0; index < normalized.length; index += 1) {
    let closest = 0;
    if (Math.abs(normalized[index] - centers[1]) < Math.abs(normalized[index] - centers[closest])) closest = 1;
    if (Math.abs(normalized[index] - centers[2]) < Math.abs(normalized[index] - centers[closest])) closest = 2;
    bands[index] = closest;
  }
  return { normalized, bands, centers };
}

function createDepthLayers(baseMasks: BaseSemanticMask[], depth: Float32Array | null) {
  if (!baseMasks.length) return { layers: [] as SemanticLayer[], depthEnhanced: false };
  const { width, height } = baseMasks[0];
  const pixelTotal = width * height;
  const foreground = new Uint8Array(pixelTotal);
  for (const mask of baseMasks) for (let index = 0; index < pixelTotal; index += 1) if (mask.data[index]) foreground[index] = 1;
  const clusters = buildDepthClusters(depth ? smoothDepthMap(depth, width, height) : null, foreground);
  const layers: SemanticLayer[] = [];
  let colorIndex = 0;
  for (const base of baseMasks) {
    const baseCount = base.data.reduce((sum, value) => sum + (value ? 1 : 0), 0);
    const baseCoverage = baseCount / pixelTotal;
    const counts = [0, 0, 0];
    if (clusters) for (let index = 0; index < pixelTotal; index += 1) if (base.data[index]) counts[clusters.bands[index]] += 1;
    const presentBands = counts.map((count, band) => ({ count, band })).filter(({ count }) => count / pixelTotal >= 0.0045);
    if (clusters && baseCoverage >= 0.025 && presentBands.length >= 2) {
      const activeBands = presentBands.map(({ band }) => band);
      for (const band of [...activeBands].sort((a, b) => b - a)) {
        let assignedCount = 0;
        const data = Uint8ClampedArray.from(base.data, (value, index) => {
          if (!value) return 0;
          const sourceBand = clusters.bands[index];
          const assignedBand = activeBands.reduce((closest, candidate) => Math.abs(candidate - sourceBand) < Math.abs(closest - sourceBand) ? candidate : closest, activeBands[0]);
          if (assignedBand !== band) return 0;
          assignedCount += 1; return 255;
        });
        const names = ["Far", "Middle", "Near"];
        layers.push({ id: `${base.label.toLowerCase().replaceAll(" ", "-")}-${names[band].toLowerCase()}-${base.sourceIndex}`, label: `${titleCase(base.label)} · ${names[band]}`, width, height, data, color: LAYER_COLORS[colorIndex++ % LAYER_COLORS.length], coverage: assignedCount / pixelTotal, depthScore: clusters.centers[band] });
      }
    } else {
      let depthTotal = 0;
      if (clusters) for (let index = 0; index < pixelTotal; index += 1) if (base.data[index]) depthTotal += clusters.normalized[index];
      layers.push({ id: `${base.label.toLowerCase().replaceAll(" ", "-")}-${base.sourceIndex}`, label: titleCase(base.label), width, height, data: base.data, color: LAYER_COLORS[colorIndex++ % LAYER_COLORS.length], coverage: baseCoverage, depthScore: clusters && baseCount ? depthTotal / baseCount : base.averageY / Math.max(1, height) });
    }
  }
  return { layers: layers.filter((layer) => layer.coverage >= 0.00075).sort((a, b) => b.depthScore - a.depthScore), depthEnhanced: Boolean(clusters) };
}

export async function analyzeImage(image: HTMLImageElement, onStatus?: (status: "loading-model" | "analyzing") => void) {
  onStatus?.("loading-model");
  const [segmenter, depthEstimator] = await Promise.all([
    getSegmenter(),
    getDepthEstimator().catch((error) => { console.warn("Depth model unavailable; continuing with semantic layers.", error); depthEstimatorPromise = null; return null; }),
  ]);
  onStatus?.("analyzing");
  const scale = Math.min(1, 1024 / Math.max(image.naturalWidth, image.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  canvas.getContext("2d")!.drawImage(image, 0, 0, canvas.width, canvas.height);
  const segments = await segmenter(canvas);
  const depthOutput = depthEstimator ? await depthEstimator(canvas).catch((error) => { console.warn("Depth analysis failed; continuing with semantic layers.", error); return null; }) : null;
  const sky = segments.find((segment) => segment.label?.toLowerCase() === "sky");
  const cleanSky = sky ? cleanSkyMask(sky.mask.data, sky.mask.width, sky.mask.height) : null;
  const skyMask: BinaryMask | null = sky && cleanSky ? { width: sky.mask.width, height: sky.mask.height, data: cleanSky } : null;
  const baseMasks = segments.filter((segment) => segment.label?.toLowerCase() !== "sky").map((segment, index) => {
    const { width, height } = segment.mask;
    const data = Uint8ClampedArray.from(segment.mask.data, (value, pixelIndex) => cleanSky?.[pixelIndex] ? 0 : value > 127 ? 255 : 0);
    return summarizeMask({ label: segment.label || `Layer ${index + 1}`, sourceIndex: index, width, height, data, averageY: 0 });
  });
  if (!baseMasks.length) throw new Error("No distinct image layers were detected in this photograph.");
  const grouped = groupSemanticMasks(baseMasks);
  const { layers, depthEnhanced } = createDepthLayers(grouped, resizeDepthMap(depthOutput, grouped[0].width, grouped[0].height));
  if (!layers.length) throw new Error("No distinct depth layers were detected in this photograph.");
  return { skyMask, layers, quality: (depthEnhanced ? "depth" : "semantic") as "depth" | "semantic" };
}

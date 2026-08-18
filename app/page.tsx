"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";

type TextAlign = "left" | "center" | "right";
type BinaryMask = { width: number; height: number; data: Uint8ClampedArray };
type BaseSemanticMask = BinaryMask & { label: string; sourceIndex: number; averageY: number };
type SemanticLayer = BinaryMask & { id: string; label: string; color: [number, number, number]; coverage: number; depthScore: number };
type MaskStatus = "idle" | "loading-model" | "analyzing" | "ready" | "error";
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

const FONT_OPTIONS = [
  ["Impact", "Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif"],
  ["Arial Black", "'Arial Black', Arial, sans-serif"],
  ["Helvetica", "Helvetica, Arial, sans-serif"],
  ["Georgia", "Georgia, serif"],
  ["Times", "'Times New Roman', Times, serif"],
  ["Courier", "'Courier New', monospace"],
] as const;

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
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      if (y >= height - 2) touchesBottom = true;
      for (const next of [index - 1, index + 1, index - width, index + width]) {
        if (next < 0 || next >= sky.length || visited[next] || sky[next]) continue;
        if (Math.abs((next % width) - x) > 1) continue;
        visited[next] = 1;
        queue[tail++] = next;
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
    if (!existing) {
      groups.set(label, { ...mask, label, data: new Uint8ClampedArray(mask.data) });
      continue;
    }
    for (let index = 0; index < existing.data.length; index += 1) if (mask.data[index]) existing.data[index] = 255;
  }
  let grouped = [...groups.values()].map((mask, index) => {
    let count = 0;
    let yTotal = 0;
    for (let pixelIndex = 0; pixelIndex < mask.data.length; pixelIndex += 1) {
      if (!mask.data[pixelIndex]) continue;
      count += 1;
      yTotal += Math.floor(pixelIndex / mask.width);
    }
    return { ...mask, sourceIndex: index, averageY: count ? yTotal / count : 0 };
  });

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
  return grouped.map((mask, index) => {
    let count = 0;
    let yTotal = 0;
    for (let pixelIndex = 0; pixelIndex < mask.data.length; pixelIndex += 1) {
      if (!mask.data[pixelIndex]) continue;
      count += 1;
      yTotal += Math.floor(pixelIndex / mask.width);
    }
    return { ...mask, sourceIndex: index, averageY: count ? yTotal / count : 0 };
  });
}

function resizeDepthMap(output: DepthOutput | null, width: number, height: number) {
  const tensor = output?.predicted_depth;
  const raw = tensor?.data ?? output?.depth?.data;
  if (!raw?.length) return null;
  const dims = tensor?.dims;
  const sourceHeight = dims?.at(-2) ?? output?.depth?.height ?? height;
  const sourceWidth = dims?.at(-1) ?? output?.depth?.width ?? width;
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
      horizontal[y * width + x] = sum / (radius * 2 + 1);
      sum -= input[y * width + Math.max(0, x - radius)];
      sum += input[y * width + Math.min(width - 1, x + radius + 1)];
    }
  }
  for (let x = 0; x < width; x += 1) {
    let sum = 0;
    for (let y = -radius; y <= radius; y += 1) sum += horizontal[Math.max(0, Math.min(height - 1, y)) * width + x];
    for (let y = 0; y < height; y += 1) {
      output[y * width + x] = sum / (radius * 2 + 1);
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
  for (let index = 0; index < depth.length; index += stride) {
    if (foreground[index] && Number.isFinite(depth[index])) sample.push(depth[index]);
  }
  if (sample.length < 100) return null;
  sample.sort((a, b) => a - b);
  const low = percentile(sample, 0.04);
  const high = percentile(sample, 0.96);
  if (high - low < 1e-5) return null;
  const normalize = (value: number) => Math.max(0, Math.min(1, (value - low) / (high - low)));
  const normalizedSample = sample.map(normalize);
  let centers = [percentile(normalizedSample, 0.2), percentile(normalizedSample, 0.5), percentile(normalizedSample, 0.8)];
  for (let iteration = 0; iteration < 8; iteration += 1) {
    const sums = [0, 0, 0];
    const counts = [0, 0, 0];
    for (const value of normalizedSample) {
      let closest = 0;
      if (Math.abs(value - centers[1]) < Math.abs(value - centers[closest])) closest = 1;
      if (Math.abs(value - centers[2]) < Math.abs(value - centers[closest])) closest = 2;
      sums[closest] += value;
      counts[closest] += 1;
    }
    centers = centers.map((center, index) => counts[index] ? sums[index] / counts[index] : center).sort((a, b) => a - b);
  }
  if (centers[2] - centers[0] < 0.16) return null;
  const normalized = Float32Array.from(depth, normalize);
  const bands = new Uint8Array(depth.length);
  for (let index = 0; index < normalized.length; index += 1) {
    const value = normalized[index];
    let closest = 0;
    if (Math.abs(value - centers[1]) < Math.abs(value - centers[closest])) closest = 1;
    if (Math.abs(value - centers[2]) < Math.abs(value - centers[closest])) closest = 2;
    bands[index] = closest;
  }
  return { normalized, bands, centers };
}

function createDepthLayers(baseMasks: BaseSemanticMask[], depth: Float32Array | null) {
  if (!baseMasks.length) return { layers: [] as SemanticLayer[], depthEnhanced: false };
  const { width, height } = baseMasks[0];
  const pixelTotal = width * height;
  const foreground = new Uint8Array(pixelTotal);
  for (const mask of baseMasks) {
    for (let index = 0; index < pixelTotal; index += 1) if (mask.data[index]) foreground[index] = 1;
  }
  const clusters = buildDepthClusters(depth ? smoothDepthMap(depth, width, height) : null, foreground);
  const layers: SemanticLayer[] = [];
  let colorIndex = 0;

  for (const base of baseMasks) {
    const baseCount = base.data.reduce((sum, value) => sum + (value ? 1 : 0), 0);
    const baseCoverage = baseCount / pixelTotal;
    const counts = [0, 0, 0];
    if (clusters) {
      for (let index = 0; index < pixelTotal; index += 1) if (base.data[index]) counts[clusters.bands[index]] += 1;
    }
    const presentBands = counts.map((count, band) => ({ count, band })).filter(({ count }) => count / pixelTotal >= 0.0045);
    const shouldSplit = Boolean(clusters && baseCoverage >= 0.025 && presentBands.length >= 2);

    if (shouldSplit && clusters) {
      const activeBands = presentBands.map(({ band }) => band);
      for (const band of [...activeBands].sort((a, b) => b - a)) {
        let assignedCount = 0;
        const data = Uint8ClampedArray.from(base.data, (value, index) => {
          if (!value) return 0;
          const sourceBand = clusters.bands[index];
          const assignedBand = activeBands.reduce((closest, candidate) => Math.abs(candidate - sourceBand) < Math.abs(closest - sourceBand) ? candidate : closest, activeBands[0]);
          if (assignedBand !== band) return 0;
          assignedCount += 1;
          return 255;
        });
        const names = ["Far", "Middle", "Near"];
        layers.push({
          id: `${base.label.toLowerCase().replaceAll(" ", "-")}-${names[band].toLowerCase()}-${base.sourceIndex}`,
          label: `${titleCase(base.label)} · ${names[band]}`,
          width,
          height,
          data,
          color: LAYER_COLORS[colorIndex++ % LAYER_COLORS.length],
          coverage: assignedCount / pixelTotal,
          depthScore: clusters.centers[band],
        });
      }
    } else {
      let depthTotal = 0;
      if (clusters) {
        for (let index = 0; index < pixelTotal; index += 1) if (base.data[index]) depthTotal += clusters.normalized[index];
      }
      layers.push({
        id: `${base.label.toLowerCase().replaceAll(" ", "-")}-${base.sourceIndex}`,
        label: titleCase(base.label),
        width,
        height,
        data: base.data,
        color: LAYER_COLORS[colorIndex++ % LAYER_COLORS.length],
        coverage: baseCoverage,
        depthScore: clusters && baseCount ? depthTotal / baseCount : base.averageY / Math.max(1, height),
      });
    }
  }

  return {
    layers: layers.filter((layer) => layer.coverage >= 0.00075).sort((a, b) => b.depthScore - a.depthScore),
    depthEnhanced: Boolean(clusters),
  };
}

function createMaskCanvas(mask: BinaryMask) {
  const canvas = document.createElement("canvas");
  canvas.width = mask.width;
  canvas.height = mask.height;
  const ctx = canvas.getContext("2d")!;
  const pixels = ctx.createImageData(mask.width, mask.height);
  for (let index = 0; index < mask.data.length; index += 1) {
    const offset = index * 4;
    pixels.data[offset] = pixels.data[offset + 1] = pixels.data[offset + 2] = 255;
    pixels.data[offset + 3] = mask.data[index];
  }
  ctx.putImageData(pixels, 0, 0);
  return canvas;
}

function mergeMasks(layers: SemanticLayer[], layerIds: string[]): BinaryMask | null {
  const selected = layers.filter((layer) => layerIds.includes(layer.id));
  if (!selected.length) return null;
  const { width, height } = selected[0];
  const data = new Uint8ClampedArray(width * height);
  for (const layer of selected) {
    for (let index = 0; index < data.length; index += 1) {
      if (layer.data[index] > data[index]) data[index] = layer.data[index];
    }
  }
  return { width, height, data };
}

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const analysisSequence = useRef(0);
  const [imageName, setImageName] = useState("");
  const [dimensions, setDimensions] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [maskStatus, setMaskStatus] = useState<MaskStatus>("idle");
  const [analysisQuality, setAnalysisQuality] = useState<"idle" | "semantic" | "depth">("idle");
  const [maskError, setMaskError] = useState("");
  const [skyMask, setSkyMask] = useState<BinaryMask | null>(null);
  const [semanticLayers, setSemanticLayers] = useState<SemanticLayer[]>([]);
  const [frontLayerIds, setFrontLayerIds] = useState<string[]>([]);
  const [text, setText] = useState("ONE DESERT\nAFTER\nANOTHER");
  const [font, setFont] = useState<string>(FONT_OPTIONS[0][1]);
  const [fontSize, setFontSize] = useState(14);
  const [lineGap, setLineGap] = useState(18);
  const [xPosition, setXPosition] = useState(50);
  const [yPosition, setYPosition] = useState(38);
  const [alignment, setAlignment] = useState<TextAlign>("center");
  const [textColor, setTextColor] = useState("#f6edd7");
  const [shadowColor, setShadowColor] = useState("#3a2a22");
  const [shadow, setShadow] = useState(true);
  const [showMask, setShowMask] = useState(false);

  const drawPoster = useCallback((target: HTMLCanvasElement, maskOverlay = false) => {
    const image = imageRef.current;
    if (!image) return;
    const scale = Math.min(1, 7000 / Math.max(image.naturalWidth, image.naturalHeight));
    const width = Math.round(image.naturalWidth * scale);
    const height = Math.round(image.naturalHeight * scale);
    target.width = width;
    target.height = height;
    const ctx = target.getContext("2d")!;
    ctx.drawImage(image, 0, 0, width, height);

    const typeLayer = document.createElement("canvas");
    typeLayer.width = width;
    typeLayer.height = height;
    const typeCtx = typeLayer.getContext("2d")!;
    const sizePx = (fontSize / 100) * height;
    const lines = text.split("\n");
    const lineHeight = sizePx * (1 + lineGap / 100);
    const totalHeight = Math.max(sizePx, (lines.length - 1) * lineHeight + sizePx);
    const startBaseline = (yPosition / 100) * height - totalHeight / 2 + sizePx * 0.82;
    const x = (xPosition / 100) * width;
    typeCtx.font = `900 ${sizePx}px ${font}`;
    typeCtx.textAlign = alignment;
    typeCtx.textBaseline = "alphabetic";
    typeCtx.lineJoin = "round";
    lines.forEach((line, index) => {
      const baseline = startBaseline + index * lineHeight;
      if (shadow) {
        typeCtx.fillStyle = shadowColor;
        const offset = Math.max(3, sizePx * 0.045);
        typeCtx.fillText(line, x + offset, baseline + offset);
      }
      typeCtx.fillStyle = textColor;
      typeCtx.fillText(line, x, baseline);
    });
    ctx.drawImage(typeLayer, 0, 0);

    const frontMask = mergeMasks(semanticLayers, frontLayerIds);
    if (frontMask) {
      const foreground = document.createElement("canvas");
      foreground.width = width;
      foreground.height = height;
      const foregroundCtx = foreground.getContext("2d")!;
      foregroundCtx.drawImage(image, 0, 0, width, height);
      foregroundCtx.globalCompositeOperation = "destination-in";
      foregroundCtx.imageSmoothingEnabled = true;
      foregroundCtx.drawImage(createMaskCanvas(frontMask), 0, 0, width, height);
      ctx.drawImage(foreground, 0, 0);
    }

    if (maskOverlay && (skyMask || semanticLayers.length)) {
      const overlay = document.createElement("canvas");
      const reference = skyMask ?? semanticLayers[0];
      overlay.width = reference.width;
      overlay.height = reference.height;
      const overlayCtx = overlay.getContext("2d")!;
      const pixels = overlayCtx.createImageData(reference.width, reference.height);
      for (let index = 0; index < reference.width * reference.height; index += 1) {
        const offset = index * 4;
        if (skyMask?.data[index]) {
          pixels.data[offset] = 73;
          pixels.data[offset + 1] = 175;
          pixels.data[offset + 2] = 255;
          pixels.data[offset + 3] = 64;
        }
        for (const layer of semanticLayers) {
          if (!layer.data[index]) continue;
          pixels.data[offset] = layer.color[0];
          pixels.data[offset + 1] = layer.color[1];
          pixels.data[offset + 2] = layer.color[2];
          pixels.data[offset + 3] = frontLayerIds.includes(layer.id) ? 104 : 42;
          break;
        }
      }
      overlayCtx.putImageData(pixels, 0, 0);
      ctx.drawImage(overlay, 0, 0, width, height);
    }
  }, [alignment, font, fontSize, frontLayerIds, lineGap, semanticLayers, shadow, shadowColor, skyMask, text, textColor, xPosition, yPosition]);

  const analyzeImage = useCallback(async (image: HTMLImageElement) => {
    const sequence = ++analysisSequence.current;
    setSkyMask(null);
    setSemanticLayers([]);
    setFrontLayerIds([]);
    setAnalysisQuality("idle");
    setMaskError("");
    setMaskStatus("loading-model");
    try {
      const [segmenter, depthEstimator] = await Promise.all([
        getSegmenter(),
        getDepthEstimator().catch((error) => {
          console.warn("Depth model unavailable; continuing with semantic layers.", error);
          depthEstimatorPromise = null;
          return null;
        }),
      ]);
      if (sequence !== analysisSequence.current) return;
      setMaskStatus("analyzing");
      const scale = Math.min(1, 1024 / Math.max(image.naturalWidth, image.naturalHeight));
      const analysis = document.createElement("canvas");
      analysis.width = Math.max(1, Math.round(image.naturalWidth * scale));
      analysis.height = Math.max(1, Math.round(image.naturalHeight * scale));
      analysis.getContext("2d")!.drawImage(image, 0, 0, analysis.width, analysis.height);
      const segments = await segmenter(analysis);
      if (sequence !== analysisSequence.current) return;
      const depthOutput = depthEstimator ? await depthEstimator(analysis).catch((error) => {
        console.warn("Depth analysis failed; continuing with semantic layers.", error);
        return null;
      }) : null;
      if (sequence !== analysisSequence.current) return;
      const sky = segments.find((segment) => segment.label?.toLowerCase() === "sky");
      const cleanSky = sky ? cleanSkyMask(sky.mask.data, sky.mask.width, sky.mask.height) : null;
      if (sky && cleanSky) setSkyMask({ width: sky.mask.width, height: sky.mask.height, data: cleanSky });

      const baseMasks = segments
        .filter((segment) => segment.label?.toLowerCase() !== "sky")
        .map((segment, index) => {
          const { width, height } = segment.mask;
          const data = Uint8ClampedArray.from(segment.mask.data, (value, pixelIndex) => {
            if (cleanSky?.[pixelIndex]) return 0;
            return value > 127 ? 255 : 0;
          });
          let pixelCount = 0;
          let yTotal = 0;
          for (let pixelIndex = 0; pixelIndex < data.length; pixelIndex += 1) {
            if (!data[pixelIndex]) continue;
            pixelCount += 1;
            yTotal += Math.floor(pixelIndex / width);
          }
          const label = segment.label || `Layer ${index + 1}`;
          return {
            label,
            sourceIndex: index,
            width,
            height,
            data,
            averageY: pixelCount ? yTotal / pixelCount : 0,
          } satisfies BaseSemanticMask;
        });

      if (!baseMasks.length) throw new Error("No distinct image layers were detected in this photograph.");
      const groupedMasks = groupSemanticMasks(baseMasks);
      const depthMap = resizeDepthMap(depthOutput, groupedMasks[0].width, groupedMasks[0].height);
      const { layers, depthEnhanced } = createDepthLayers(groupedMasks, depthMap);
      if (!layers.length) throw new Error("No distinct depth layers were detected in this photograph.");
      setSemanticLayers(layers);
      setFrontLayerIds(layers.map((layer) => layer.id));
      setAnalysisQuality(depthEnhanced ? "depth" : "semantic");
      setMaskStatus("ready");
    } catch (error) {
      console.error(error);
      setMaskError(error instanceof Error ? error.message : "The vision models could not run in this browser.");
      setMaskStatus("error");
    }
  }, []);

  const setLayerInFront = (layerId: string, inFront: boolean) => {
    setFrontLayerIds((current) => inFront ? [...new Set([...current, layerId])] : current.filter((id) => id !== layerId));
  };

  useEffect(() => { if (canvasRef.current) drawPoster(canvasRef.current, showMask); }, [drawPoster, showMask]);

  const loadFile = (file?: File) => {
    if (!file || !file.type.startsWith("image/")) return;
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      setImageName(file.name);
      setDimensions(`${image.naturalWidth.toLocaleString()} × ${image.naturalHeight.toLocaleString()}`);
      URL.revokeObjectURL(objectUrl);
      void analyzeImage(image);
    };
    image.onerror = () => { URL.revokeObjectURL(objectUrl); setMaskError("That image could not be opened."); setMaskStatus("error"); };
    image.src = objectUrl;
  };
  const handleFile = (event: ChangeEvent<HTMLInputElement>) => loadFile(event.target.files?.[0]);
  const handleDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setIsDragging(false); loadFile(event.dataTransfer.files?.[0]); };
  const downloadPoster = () => {
    if (!imageRef.current) return;
    const exportCanvas = document.createElement("canvas");
    drawPoster(exportCanvas, false);
    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${imageName.replace(/\.[^.]+$/, "") || "poster"}-skyline-poster.png`;
      anchor.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  };

  const busy = maskStatus === "loading-model" || maskStatus === "analyzing";
  const statusText = maskStatus === "loading-model" ? "Loading segmentation + depth…" : maskStatus === "analyzing" ? "Tracing depth planes…" : maskStatus === "ready" ? `${semanticLayers.length} depth layer${semanticLayers.length === 1 ? "" : "s"} ready` : maskStatus === "error" ? "Layers unavailable" : "Waiting for image";

  return <main className="studio-shell">
    <header className="masthead"><div><p className="eyebrow">Browser-based poster maker</p><h1>Skyline Type Studio</h1></div><p className="privacy-note">Models download once. Your photograph stays in this browser.</p></header>
    <section className="workspace">
      <aside className="control-panel" aria-label="Poster controls">
        <section className="control-section">
          <div className="panel-heading"><span className="step">01</span><div><p className="label">Source image</p><p className="hint">Mountains, coasts, cities, and trees all work.</p></div></div>
          <button className="upload-button" onClick={() => fileInput.current?.click()}>{imageName ? "Replace photograph" : "Choose a photograph"}</button>
          <input ref={fileInput} type="file" accept="image/*" hidden onChange={handleFile} />
          {imageName && <p className="file-meta"><span>{imageName}</span><span>{dimensions}</span></p>}
        </section>
        <section className="control-section">
          <div className="panel-heading"><span className="step">02</span><div><p className="label">Poster text</p><p className="hint">Line breaks are preserved.</p></div></div>
          <textarea value={text} onChange={(event) => setText(event.target.value)} rows={4} aria-label="Poster text" />
          <div className="field-grid">
            <label><span>Typeface</span><select value={font} onChange={(event) => setFont(event.target.value)}>{FONT_OPTIONS.map(([name, value]) => <option value={value} key={name}>{name}</option>)}</select></label>
            <label><span>Alignment</span><select value={alignment} onChange={(event) => setAlignment(event.target.value as TextAlign)}><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option></select></label>
          </div>
          <Range label="Font size" value={fontSize} min={4} max={30} suffix="%" onChange={setFontSize} />
          <Range label="Line spacing" value={lineGap} min={-20} max={80} suffix="%" onChange={setLineGap} />
          <Range label="Horizontal position" value={xPosition} min={0} max={100} suffix="%" onChange={setXPosition} />
          <Range label="Vertical position" value={yPosition} min={0} max={100} suffix="%" onChange={setYPosition} />
          <div className="color-grid">
            <label><span>Type</span><input type="color" value={textColor} onChange={(event) => setTextColor(event.target.value)} /></label>
            <label><span>Shadow</span><input type="color" value={shadowColor} onChange={(event) => setShadowColor(event.target.value)} /></label>
          </div>
          <Toggle checked={shadow} onChange={setShadow} label="Hard offset shadow" />
        </section>
        <section className="control-section mask-section">
          <div className="panel-heading"><span className="step">03</span><div><p className="label">Depth layers</p><p className="hint">Semantic objects are split into near, middle, and far planes.</p></div></div>
          <div className={`mask-readout ${maskStatus}`}><span className="status-dot" />{statusText}</div>
          {maskStatus === "ready" && <p className={`quality-badge ${analysisQuality}`}>{analysisQuality === "depth" ? "Depth-enhanced analysis" : "Semantic fallback"}</p>}
          {maskError && <p className="mask-error">{maskError}</p>}
          {semanticLayers.length > 0 && <>
            <div className="depth-presets" aria-label="Depth presets">
              <button type="button" onClick={() => setFrontLayerIds(semanticLayers.map((layer) => layer.id))}>All in front</button>
              <button type="button" onClick={() => setFrontLayerIds([])}>Text on top</button>
            </div>
            <div className="depth-stack" aria-label="Detected image layers">
              <div className="stack-cap"><span>Closer</span><span>Object depth</span></div>
              {semanticLayers.filter((layer) => frontLayerIds.includes(layer.id)).map((layer) => <DepthLayerRow key={layer.id} layer={layer} inFront onChange={setLayerInFront} />)}
              <div className="text-layer-marker"><span>T</span><b>Your text layer</b></div>
              {semanticLayers.filter((layer) => !frontLayerIds.includes(layer.id)).map((layer) => <DepthLayerRow key={layer.id} layer={layer} inFront={false} onChange={setLayerInFront} />)}
              <div className="base-layer"><span className="layer-swatch sky-base" /><span><b>Original photo</b><small>Base layer</small></span></div>
            </div>
          </>}
          <Toggle checked={showMask} onChange={setShowMask} label="Show colored layer overlay" />
        </section>
        <button className="download-button" disabled={!imageName || busy} onClick={downloadPoster}>{busy ? statusText : "Download full-resolution PNG"}</button>
      </aside>
      <section className="preview-panel" aria-label="Poster preview">
        <div className="preview-topline"><div><span className={`status-dot ${semanticLayers.length ? "ready" : ""}`} />Live canvas</div><span>{imageName ? statusText : "Waiting for image"}</span></div>
        {!imageName ? <div className={`drop-zone ${isDragging ? "dragging" : ""}`} onClick={() => fileInput.current?.click()} onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInput.current?.click(); }}><span className="drop-mark">＋</span><strong>Drop a photograph here</strong><small>or click to browse · JPEG, PNG, WebP</small></div> : <div className="canvas-stage"><canvas ref={canvasRef} aria-label="Live poster preview" />{busy && <div className="calculating-chip">{statusText}</div>}</div>}
        <footer className="preview-footer"><span><i className="key-swatch sky" />Sky / base</span><span><i className="key-swatch land" />Selected layers</span><span className="footer-tip">Set each object behind or in front of your text to build a custom depth stack.</span></footer>
      </section>
    </section>
  </main>;
}

function Range({ label, value, min, max, suffix, onChange }: { label: string; value: number; min: number; max: number; suffix: string; onChange: (value: number) => void }) {
  const id = `range-${label.toLowerCase().replaceAll(" ", "-")}`;
  return <div className="range-field"><span><label htmlFor={id}><b>{label}</b></label><output htmlFor={id}>{value}{suffix}</output></span><input id={id} type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} /></div>;
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (value: boolean) => void; label: string }) {
  return <label className="toggle-row"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span className="toggle-track"><i /></span><b>{label}</b></label>;
}

function DepthLayerRow({ layer, inFront, onChange }: { layer: SemanticLayer; inFront: boolean; onChange: (layerId: string, inFront: boolean) => void }) {
  return <div className={`depth-layer ${inFront ? "is-front" : "is-behind"}`}>
    <span className="layer-swatch" style={{ backgroundColor: `rgb(${layer.color.join(",")})` }} />
    <span className="layer-name">{layer.label}<small>{Math.max(0.1, layer.coverage * 100).toFixed(1)}%</small></span>
    <span className="depth-choice" role="group" aria-label={`${layer.label} text depth`}>
      <button type="button" className={!inFront ? "active" : ""} aria-pressed={!inFront} onClick={() => onChange(layer.id, false)}>Behind</button>
      <button type="button" className={inFront ? "active" : ""} aria-pressed={inFront} onClick={() => onChange(layer.id, true)}>In front</button>
    </span>
  </div>;
}

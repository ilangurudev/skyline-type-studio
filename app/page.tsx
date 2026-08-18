"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import { createProjectArchive, deserializeAnalysis, readProjectArchive, serializeAnalysis, sha256File } from "@/studio/project";
import { renderPoster } from "@/studio/render";
import type { SkylineProjectV1, SourceFingerprint } from "@/studio/types";

type TextAlign = "left" | "center" | "right";
type TextLayer = {
  id: string;
  name: string;
  text: string;
  font: string;
  fontSize: number;
  lineGap: number;
  xPosition: number;
  yPosition: number;
  alignment: TextAlign;
  textColor: string;
  shadowColor: string;
  shadow: boolean;
  extrusion: boolean;
  extrusionColor: string;
  extrusionDepth: number;
  extrusionAngle: number;
  frontLayerIds: string[];
};
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

function createTextLayer(id: string, index: number, frontLayerIds: string[] = []): TextLayer {
  return {
    id,
    name: `Text ${index}`,
    text: index === 1 ? "ONE DESERT\nAFTER\nANOTHER" : "NEW TEXT",
    font: FONT_OPTIONS[0][1],
    fontSize: index === 1 ? 14 : 10,
    lineGap: 18,
    xPosition: 50,
    yPosition: index === 1 ? 38 : 50,
    alignment: "center",
    textColor: "#f6edd7",
    shadowColor: "#3a2a22",
    shadow: true,
    extrusion: false,
    extrusionColor: "#8a3f2b",
    extrusionDepth: 10,
    extrusionAngle: 45,
    frontLayerIds,
  };
}

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

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const projectInput = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const sourceFingerprintRef = useRef<SourceFingerprint | null>(null);
  const pendingProjectRef = useRef<SkylineProjectV1 | null>(null);
  const analysisSequence = useRef(0);
  const nextTextLayerId = useRef(2);
  const [imageName, setImageName] = useState("");
  const [dimensions, setDimensions] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [maskStatus, setMaskStatus] = useState<MaskStatus>("idle");
  const [analysisQuality, setAnalysisQuality] = useState<"idle" | "semantic" | "depth">("idle");
  const [maskError, setMaskError] = useState("");
  const [skyMask, setSkyMask] = useState<BinaryMask | null>(null);
  const [semanticLayers, setSemanticLayers] = useState<SemanticLayer[]>([]);
  const [textLayers, setTextLayers] = useState<TextLayer[]>(() => [createTextLayer("text-1", 1)]);
  const [activeTextLayerId, setActiveTextLayerId] = useState("text-1");
  const [showMask, setShowMask] = useState(false);
  const activeTextLayer = textLayers.find((layer) => layer.id === activeTextLayerId) ?? textLayers[0];

  const updateActiveTextLayer = (updates: Partial<TextLayer>) => {
    setTextLayers((current) => current.map((layer) => layer.id === activeTextLayerId ? { ...layer, ...updates } : layer));
  };

  const addTextLayer = () => {
    const index = nextTextLayerId.current++;
    const layer = createTextLayer(`text-${index}`, index);
    setTextLayers((current) => [...current, layer]);
    setActiveTextLayerId(layer.id);
  };

  const removeActiveTextLayer = () => {
    if (textLayers.length === 1) return;
    const activeIndex = textLayers.findIndex((layer) => layer.id === activeTextLayerId);
    const nextActive = textLayers[Math.max(0, activeIndex - 1)] ?? textLayers[0];
    setTextLayers((current) => current.filter((layer) => layer.id !== activeTextLayerId));
    setActiveTextLayerId(nextActive.id);
  };

  const moveActiveTextLayer = (direction: -1 | 1) => {
    setTextLayers((current) => {
      const from = current.findIndex((layer) => layer.id === activeTextLayerId);
      const to = from + direction;
      if (from < 0 || to < 0 || to >= current.length) return current;
      const reordered = [...current];
      [reordered[from], reordered[to]] = [reordered[to], reordered[from]];
      return reordered;
    });
  };

  const drawPoster = useCallback((target: HTMLCanvasElement, maskOverlay = false, maxDimension?: number) => {
    const image = imageRef.current;
    if (!image) return;
    return renderPoster({ target, image, textLayers, semanticLayers, skyMask, activeTextLayerId, maskOverlay, maxDimension });
  }, [activeTextLayerId, semanticLayers, skyMask, textLayers]);

  const analyzeImage = useCallback(async (image: HTMLImageElement) => {
    const sequence = ++analysisSequence.current;
    setSkyMask(null);
    setSemanticLayers([]);
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
      setTextLayers((current) => current.map((textLayer) => ({ ...textLayer, frontLayerIds: layers.map((layer) => layer.id) })));
      setAnalysisQuality(depthEnhanced ? "depth" : "semantic");
      setMaskStatus("ready");
    } catch (error) {
      console.error(error);
      setMaskError(error instanceof Error ? error.message : "The vision models could not run in this browser.");
      setMaskStatus("error");
    }
  }, []);

  const setLayerInFront = (layerId: string, inFront: boolean) => {
    if (!activeTextLayer) return;
    const frontLayerIds = inFront
      ? [...new Set([...activeTextLayer.frontLayerIds, layerId])]
      : activeTextLayer.frontLayerIds.filter((id) => id !== layerId);
    updateActiveTextLayer({ frontLayerIds });
  };

  useEffect(() => { if (canvasRef.current) drawPoster(canvasRef.current, showMask); }, [drawPoster, showMask]);

  const loadFile = async (file?: File) => {
    if (!file || !file.type.startsWith("image/")) return;
    const pendingProject = pendingProjectRef.current;
    setMaskError("");
    let sha256: string;
    try {
      sha256 = await sha256File(file);
    } catch {
      setMaskError("The browser could not fingerprint this photograph.");
      setMaskStatus("error");
      return;
    }
    if (pendingProject && sha256 !== pendingProject.source.sha256) {
      setMaskError(`This project expects ${pendingProject.source.name}. Choose the original matching photograph.`);
      setMaskStatus("error");
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      if (pendingProject && (image.naturalWidth !== pendingProject.source.width || image.naturalHeight !== pendingProject.source.height)) {
        URL.revokeObjectURL(objectUrl);
        setMaskError("The selected photograph dimensions do not match this project.");
        setMaskStatus("error");
        return;
      }
      imageRef.current = image;
      setImageName(file.name);
      setDimensions(`${image.naturalWidth.toLocaleString()} × ${image.naturalHeight.toLocaleString()}`);
      sourceFingerprintRef.current = { name: file.name, size: file.size, width: image.naturalWidth, height: image.naturalHeight, sha256 };
      URL.revokeObjectURL(objectUrl);
      if (pendingProject) {
        const restored = deserializeAnalysis(pendingProject.analysis);
        setSkyMask(restored.skyMask);
        setSemanticLayers(restored.layers);
        setAnalysisQuality(restored.quality);
        setTextLayers(pendingProject.recipe.textLayers);
        setActiveTextLayerId(pendingProject.recipe.activeTextLayerId);
        nextTextLayerId.current = Math.max(2, pendingProject.recipe.textLayers.length + 1);
        setMaskStatus("ready");
        pendingProjectRef.current = null;
      } else {
        void analyzeImage(image);
      }
    };
    image.onerror = () => { URL.revokeObjectURL(objectUrl); setMaskError("That image could not be opened."); setMaskStatus("error"); };
    image.src = objectUrl;
  };
  const handleFile = (event: ChangeEvent<HTMLInputElement>) => { void loadFile(event.target.files?.[0]); };
  const handleDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setIsDragging(false); void loadFile(event.dataTransfer.files?.[0]); };
  const handleProject = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      pendingProjectRef.current = readProjectArchive(new Uint8Array(await file.arrayBuffer()));
      setMaskError("");
      fileInput.current?.click();
    } catch (error) {
      pendingProjectRef.current = null;
      setMaskError(error instanceof Error ? error.message : "That Skyline project could not be opened.");
      setMaskStatus("error");
    }
  };
  const triggerDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const downloadPoster = () => {
    if (!imageRef.current) return;
    const exportCanvas = document.createElement("canvas");
    drawPoster(exportCanvas, false);
    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const baseName = imageName.replace(/\.[^.]+$/, "") || "poster";
      triggerDownload(blob, `${baseName}-skyline-poster.png`);
      const source = sourceFingerprintRef.current;
      if (source && semanticLayers.length && analysisQuality !== "idle") {
        const project: SkylineProjectV1 = {
          schemaVersion: 1,
          createdAt: new Date().toISOString(),
          source,
          analysis: serializeAnalysis(analysisQuality, skyMask, semanticLayers),
          recipe: { schemaVersion: 1, activeTextLayerId, textLayers },
        };
        triggerDownload(new Blob([createProjectArchive(project)], { type: "application/octet-stream" }), `${baseName}.skyline.cfg`);
      }
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
          <button type="button" className="project-button" onClick={() => projectInput.current?.click()}>Import .skyline.cfg project</button>
          <input ref={projectInput} type="file" accept=".cfg,.skyline.cfg,application/octet-stream" hidden onChange={(event) => { void handleProject(event); }} />
          {imageName && <p className="file-meta"><span>{imageName}</span><span>{dimensions}</span></p>}
        </section>
        <section className="control-section">
          <div className="panel-heading"><span className="step">02</span><div><p className="label">Text layers</p><p className="hint">Each layer keeps its own type, position, and depth.</p></div></div>
          <div className="text-layer-list" aria-label="Text layers">
            {textLayers.map((layer, index) => <button type="button" key={layer.id} className={layer.id === activeTextLayerId ? "active" : ""} onClick={() => setActiveTextLayerId(layer.id)} aria-pressed={layer.id === activeTextLayerId}><span>T{index + 1}</span><b>{layer.name}</b><small>{layer.text.split("\n")[0] || "Empty layer"}</small></button>)}
          </div>
          <p className="layer-order-hint">Lower text layers render in front.</p>
          <button type="button" className="add-layer-button" onClick={addTextLayer}>＋ Add new text layer</button>
          {activeTextLayer && <div className="text-layer-editor">
            <div className="layer-editor-bar">
              <label><span>Layer name</span><input value={activeTextLayer.name} onChange={(event) => updateActiveTextLayer({ name: event.target.value })} aria-label="Text layer name" /></label>
              <div className="layer-actions" aria-label="Text layer actions">
                <button type="button" onClick={() => moveActiveTextLayer(-1)} disabled={textLayers[0]?.id === activeTextLayer.id} aria-label="Move text layer backward">Back</button>
                <button type="button" onClick={() => moveActiveTextLayer(1)} disabled={textLayers.at(-1)?.id === activeTextLayer.id} aria-label="Move text layer forward">Front</button>
                <button type="button" className="remove-layer" onClick={removeActiveTextLayer} disabled={textLayers.length === 1} aria-label="Delete text layer">Delete</button>
              </div>
            </div>
            <textarea value={activeTextLayer.text} onChange={(event) => updateActiveTextLayer({ text: event.target.value })} rows={4} aria-label="Poster text" />
            <div className="field-grid">
              <label><span>Typeface</span><select value={activeTextLayer.font} onChange={(event) => updateActiveTextLayer({ font: event.target.value })}>{FONT_OPTIONS.map(([name, value]) => <option value={value} key={name}>{name}</option>)}</select></label>
              <label><span>Alignment</span><select value={activeTextLayer.alignment} onChange={(event) => updateActiveTextLayer({ alignment: event.target.value as TextAlign })}><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option></select></label>
            </div>
            <Range idPrefix={activeTextLayer.id} label="Font size" value={activeTextLayer.fontSize} min={4} max={30} suffix="%" onChange={(fontSize) => updateActiveTextLayer({ fontSize })} />
            <Range idPrefix={activeTextLayer.id} label="Line spacing" value={activeTextLayer.lineGap} min={-20} max={80} suffix="%" onChange={(lineGap) => updateActiveTextLayer({ lineGap })} />
            <Range idPrefix={activeTextLayer.id} label="Horizontal position" value={activeTextLayer.xPosition} min={0} max={100} suffix="%" onChange={(xPosition) => updateActiveTextLayer({ xPosition })} />
            <Range idPrefix={activeTextLayer.id} label="Vertical position" value={activeTextLayer.yPosition} min={0} max={100} suffix="%" onChange={(yPosition) => updateActiveTextLayer({ yPosition })} />
            <div className="color-grid">
              <label><span>Type</span><input type="color" value={activeTextLayer.textColor} onChange={(event) => updateActiveTextLayer({ textColor: event.target.value })} /></label>
              <label><span>Shadow</span><input type="color" value={activeTextLayer.shadowColor} onChange={(event) => updateActiveTextLayer({ shadowColor: event.target.value })} /></label>
            </div>
            <Toggle checked={activeTextLayer.shadow} onChange={(shadow) => updateActiveTextLayer({ shadow })} label="Hard offset shadow" />
            <div className={`effect-card ${activeTextLayer.extrusion ? "active" : ""}`}>
              <Toggle checked={activeTextLayer.extrusion} onChange={(extrusion) => updateActiveTextLayer({ extrusion })} label="3D extrusion" />
              <p>Build a solid side wall behind the type for more visual weight.</p>
              {activeTextLayer.extrusion && <div className="effect-controls">
                <label className="effect-color"><span>3D side color</span><input type="color" value={activeTextLayer.extrusionColor} onChange={(event) => updateActiveTextLayer({ extrusionColor: event.target.value })} /></label>
                <Range idPrefix={activeTextLayer.id} label="3D depth" value={activeTextLayer.extrusionDepth} min={1} max={24} suffix="%" onChange={(extrusionDepth) => updateActiveTextLayer({ extrusionDepth })} />
                <Range idPrefix={activeTextLayer.id} label="3D direction" value={activeTextLayer.extrusionAngle} min={-180} max={180} suffix="°" onChange={(extrusionAngle) => updateActiveTextLayer({ extrusionAngle })} />
              </div>}
            </div>
          </div>}
        </section>
        <section className="control-section mask-section">
          <div className="panel-heading"><span className="step">03</span><div><p className="label">Depth layers</p><p className="hint">Semantic objects are split into near, middle, and far planes.</p></div></div>
          <div className={`mask-readout ${maskStatus}`}><span className="status-dot" />{statusText}</div>
          {maskStatus === "ready" && <p className={`quality-badge ${analysisQuality}`}>{analysisQuality === "depth" ? "Depth-enhanced analysis" : "Semantic fallback"}</p>}
          {maskError && <p className="mask-error">{maskError}</p>}
          {semanticLayers.length > 0 && <>
            <div className="depth-presets" aria-label="Depth presets">
              <button type="button" onClick={() => updateActiveTextLayer({ frontLayerIds: semanticLayers.map((layer) => layer.id) })}>All in front</button>
              <button type="button" onClick={() => updateActiveTextLayer({ frontLayerIds: [] })}>Text on top</button>
            </div>
            <div className="depth-stack" aria-label="Detected image layers">
              <div className="stack-cap"><span>Closer</span><span>Object depth</span></div>
              {semanticLayers.filter((layer) => activeTextLayer?.frontLayerIds.includes(layer.id)).map((layer) => <DepthLayerRow key={layer.id} layer={layer} inFront onChange={setLayerInFront} />)}
              <div className="text-layer-marker"><span>T</span><b>{activeTextLayer?.name ?? "Selected text"}</b></div>
              {semanticLayers.filter((layer) => !activeTextLayer?.frontLayerIds.includes(layer.id)).map((layer) => <DepthLayerRow key={layer.id} layer={layer} inFront={false} onChange={setLayerInFront} />)}
              <div className="base-layer"><span className="layer-swatch sky-base" /><span><b>Original photo</b><small>Base layer</small></span></div>
            </div>
          </>}
          <Toggle checked={showMask} onChange={setShowMask} label="Show colored layer overlay" />
        </section>
        <button className="download-button" disabled={!imageName || busy} onClick={downloadPoster}>{busy ? statusText : "Download PNG + project"}</button>
      </aside>
      <section className="preview-panel" aria-label="Poster preview">
        <div className="preview-topline"><div><span className={`status-dot ${semanticLayers.length ? "ready" : ""}`} />Live canvas</div><span>{imageName ? statusText : "Waiting for image"}</span></div>
        {!imageName ? <div className={`drop-zone ${isDragging ? "dragging" : ""}`} onClick={() => fileInput.current?.click()} onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInput.current?.click(); }}><span className="drop-mark">＋</span><strong>Drop a photograph here</strong><small>or click to browse · JPEG, PNG, WebP</small></div> : <div className="canvas-stage"><canvas ref={canvasRef} aria-label="Live poster preview" />{busy && <div className="calculating-chip">{statusText}</div>}</div>}
        <footer className="preview-footer"><span><i className="key-swatch sky" />Sky / base</span><span><i className="key-swatch land" />Selected layers</span><span className="footer-tip">Set each object behind or in front of the selected text layer.</span></footer>
      </section>
    </section>
  </main>;
}

function Range({ idPrefix = "poster", label, value, min, max, suffix, onChange }: { idPrefix?: string; label: string; value: number; min: number; max: number; suffix: string; onChange: (value: number) => void }) {
  const id = `range-${idPrefix}-${label.toLowerCase().replaceAll(" ", "-")}`;
  return <div className="range-field"><span><label htmlFor={id}><b>{label}</b></label><output htmlFor={id}><input aria-label={`${label} value`} type="number" min={min} max={max} value={value} onChange={(event) => onChange(Math.min(max, Math.max(min, Number(event.target.value))))} />{suffix}</output></span><input id={id} type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} /></div>;
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

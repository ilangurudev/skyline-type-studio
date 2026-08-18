"use client";

/* eslint-disable jsx-a11y/media-has-caption -- this editor handles user-provided music, not authored speech */

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import { createProjectArchive, deserializeAnalysis, readProjectArchive, serializeAnalysis, sha256File } from "@/studio/project";
import { renderPoster } from "@/studio/render";
import { renderReelFrame } from "@/studio/reel-render";
import { buildReelTimeline, DEFAULT_REEL_SETTINGS, detectBeats, estimateTempo, selectSoundtrackBeats } from "@/studio/reel";
import { createLayerAnimation, createTextLayer, createTimelineSettings, DEFAULT_TIMELINE_DURATION, FONT_OPTIONS } from "@/studio/types";
import type { ReelSettings } from "@/studio/reel";
import type { AnalysisQuality, AnimationEffect, LayerAnimation, SkylineProjectV1, SourceFingerprint, TextAlign, TextLayer, TimelineSettings } from "@/studio/types";

type BinaryMask = { width: number; height: number; data: Uint8ClampedArray };
type BaseSemanticMask = BinaryMask & { label: string; sourceIndex: number; averageY: number };
type SemanticLayer = BinaryMask & { id: string; label: string; color: [number, number, number]; coverage: number; depthScore: number };
type DepthStackItem = { kind: "semantic"; layer: SemanticLayer } | { kind: "text"; layer: TextLayer; textIndex: number };
type MaskStatus = "idle" | "loading-model" | "analyzing" | "ready" | "error";
type Segment = { label?: string; mask: { width: number; height: number; data: ArrayLike<number> } };
type Segmenter = (input: HTMLCanvasElement) => Promise<Segment[]>;
type DepthOutput = { predicted_depth?: { dims?: number[]; data?: ArrayLike<number> }; depth?: { width: number; height: number; data: ArrayLike<number> } };
type DepthEstimator = (input: HTMLCanvasElement) => Promise<DepthOutput>;

type ReelSceneRuntime = {
  id: string;
  image: HTMLImageElement;
  imageName: string;
  dimensions: string;
  source: SourceFingerprint;
  skyMask: BinaryMask | null;
  semanticLayers: SemanticLayer[];
  textLayers: TextLayer[];
  activeTextLayerId: string;
  timeline: TimelineSettings;
  analysisQuality: AnalysisQuality | "idle";
  maskStatus: MaskStatus;
  maskError: string;
};

type ReelSceneSummary = Pick<ReelSceneRuntime, "id" | "imageName" | "dimensions" | "maskStatus"> & { duration: number };

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

function createDepthStack(semanticLayers: SemanticLayer[], textLayers: TextLayer[]): DepthStackItem[] {
  const deepestFirst = [...semanticLayers].reverse();
  const textBySlot = new Map<number, Array<{ layer: TextLayer; textIndex: number }>>();
  for (const [textIndex, layer] of textLayers.entries()) {
    const foregroundCount = semanticLayers.filter((semanticLayer) => layer.frontLayerIds.includes(semanticLayer.id)).length;
    const slot = Math.max(0, Math.min(semanticLayers.length, semanticLayers.length - foregroundCount));
    textBySlot.set(slot, [...(textBySlot.get(slot) ?? []), { layer, textIndex }]);
  }

  const stack: DepthStackItem[] = [];
  for (let slot = 0; slot <= deepestFirst.length; slot += 1) {
    for (const text of textBySlot.get(slot) ?? []) stack.push({ kind: "text", ...text });
    if (slot < deepestFirst.length) stack.push({ kind: "semantic", layer: deepestFirst[slot] });
  }
  return stack;
}

function createInstaEditTimeline(layerIds: string[] = []): TimelineSettings {
  const depthOrder = [...layerIds].reverse();
  return {
    duration: DEFAULT_TIMELINE_DURATION,
    backgroundColor: "#000000",
    baseAnimation: { enabled: true, effect: "reel", delay: 50, duration: 700 },
    sceneAnimations: Object.fromEntries(depthOrder.map((id, index) => [id, {
      enabled: true,
      effect: index % 3 === 1 ? "drift" : "reel",
      delay: 380 + index * Math.max(90, Math.min(170, 800 / Math.max(1, depthOrder.length))),
      duration: 650,
    }])),
  };
}

function applyInstaEditTextAnimations(layers: TextLayer[]) {
  return layers.map((layer, index) => ({
    ...layer,
    animation: { enabled: true, effect: "reel" as const, delay: 1500 + index * Math.min(110, 500 / Math.max(1, layers.length - 1)), duration: 600 },
  }));
}

function formatTimestamp(milliseconds: number) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const projectInput = useRef<HTMLInputElement>(null);
  const soundtrackInput = useRef<HTMLInputElement>(null);
  const audioElement = useRef<HTMLAudioElement>(null);
  const audioBufferRef = useRef<AudioBuffer | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const sourceFingerprintRef = useRef<SourceFingerprint | null>(null);
  const pendingProjectRef = useRef<SkylineProjectV1 | null>(null);
  const scenesRef = useRef<ReelSceneRuntime[]>([]);
  const activeSceneIdRef = useRef<string | null>(null);
  const analysisQueueRef = useRef(false);
  const nextSceneId = useRef(1);
  const nextTextLayerId = useRef(2);
  const playbackFrame = useRef<number | null>(null);
  const reelPlaybackFrame = useRef<number | null>(null);
  const [sceneList, setSceneList] = useState<ReelSceneSummary[]>([]);
  const [activeSceneId, setActiveSceneId] = useState<string | null>(null);
  const [imageName, setImageName] = useState("");
  const [dimensions, setDimensions] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [maskStatus, setMaskStatus] = useState<MaskStatus>("idle");
  const [analysisQueueRunning, setAnalysisQueueRunning] = useState(false);
  const [analysisQuality, setAnalysisQuality] = useState<"idle" | "semantic" | "depth">("idle");
  const [maskError, setMaskError] = useState("");
  const [skyMask, setSkyMask] = useState<BinaryMask | null>(null);
  const [semanticLayers, setSemanticLayers] = useState<SemanticLayer[]>([]);
  const [textLayers, setTextLayers] = useState<TextLayer[]>(() => applyInstaEditTextAnimations([createTextLayer("text-1", 1)]));
  const [activeTextLayerId, setActiveTextLayerId] = useState("text-1");
  const [draggedTextLayerId, setDraggedTextLayerId] = useState<string | null>(null);
  const [showMask, setShowMask] = useState(false);
  const [timeline, setTimeline] = useState<TimelineSettings>(() => createInstaEditTimeline());
  const [playhead, setPlayhead] = useState(DEFAULT_TIMELINE_DURATION);
  const [isPlaying, setIsPlaying] = useState(false);
  const [previewMode, setPreviewMode] = useState<"scene" | "reel">("scene");
  const [reelSettings, setReelSettings] = useState<ReelSettings>(DEFAULT_REEL_SETTINGS);
  const [reelPlayhead, setReelPlayhead] = useState(0);
  const [isReelPlaying, setIsReelPlaying] = useState(false);
  const [soundtrackName, setSoundtrackName] = useState("");
  const [soundtrackUrl, setSoundtrackUrl] = useState("");
  const [soundtrackDuration, setSoundtrackDuration] = useState(0);
  const [soundtrackSectionStart, setSoundtrackSectionStart] = useState(0);
  const [beatTimes, setBeatTimes] = useState<number[]>([]);
  const [tempo, setTempo] = useState<number | null>(null);
  const [exportStatus, setExportStatus] = useState("");
  const activeTextLayer = textLayers.find((layer) => layer.id === activeTextLayerId) ?? textLayers[0];

  const refreshSceneList = () => {
    setSceneList(scenesRef.current.map((scene) => ({
      id: scene.id,
      imageName: scene.imageName,
      dimensions: scene.dimensions,
      maskStatus: scene.maskStatus,
      duration: scene.timeline.duration,
    })));
  };

  const captureActiveScene = () => {
    const id = activeSceneIdRef.current;
    if (!id || !imageRef.current || !sourceFingerprintRef.current) return;
    const scene = scenesRef.current.find((candidate) => candidate.id === id);
    if (!scene) return;
    Object.assign(scene, {
      image: imageRef.current,
      imageName,
      dimensions,
      source: sourceFingerprintRef.current,
      skyMask,
      semanticLayers,
      textLayers,
      activeTextLayerId,
      timeline,
      analysisQuality,
      maskStatus,
      maskError,
    });
  };

  const activateScene = (sceneId: string) => {
    if (maskStatus === "loading-model" || maskStatus === "analyzing") return;
    captureActiveScene();
    refreshSceneList();
    const scene = scenesRef.current.find((candidate) => candidate.id === sceneId);
    if (!scene) return;
    activeSceneIdRef.current = scene.id;
    setActiveSceneId(scene.id);
    imageRef.current = scene.image;
    sourceFingerprintRef.current = scene.source;
    setImageName(scene.imageName);
    setDimensions(scene.dimensions);
    setSkyMask(scene.skyMask);
    setSemanticLayers(scene.semanticLayers);
    setTextLayers(scene.textLayers);
    setActiveTextLayerId(scene.activeTextLayerId);
    setTimeline(scene.timeline);
    setPlayhead(scene.timeline.duration);
    setAnalysisQuality(scene.analysisQuality);
    setMaskStatus(scene.maskStatus);
    setMaskError(scene.maskError);
    nextTextLayerId.current = Math.max(2, scene.textLayers.length + 1);
    setPreviewMode("scene");
  };

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

  const drawPoster = useCallback((target: HTMLCanvasElement, maskOverlay = false, maxDimension?: number, time?: number) => {
    const image = imageRef.current;
    if (!image) return;
    return renderPoster({ target, image, textLayers, semanticLayers, skyMask, activeTextLayerId, maskOverlay, maxDimension, time, timeline });
  }, [activeTextLayerId, semanticLayers, skyMask, textLayers, timeline]);

  const baseReelDuration = sceneList.reduce((sum, scene) => sum + (scene.id === activeSceneId ? timeline.duration : scene.duration), 0);
  const soundtrackWindowDuration = Math.max(DEFAULT_TIMELINE_DURATION, baseReelDuration);
  const soundtrackMaxStart = Math.max(0, soundtrackDuration - soundtrackWindowDuration);
  const soundtrackStart = Math.min(soundtrackSectionStart, soundtrackMaxStart);
  const soundtrackBeatTimes = selectSoundtrackBeats(beatTimes, soundtrackStart);
  const reelSchedule = buildReelTimeline(sceneList.map((scene) => ({ id: scene.id, duration: scene.id === activeSceneId ? timeline.duration : scene.duration })), soundtrackBeatTimes, reelSettings.beatSync);
  const soundtrackSectionDuration = Math.max(DEFAULT_TIMELINE_DURATION, reelSchedule.duration);
  const soundtrackSectionEnd = Math.min(soundtrackDuration, soundtrackStart + soundtrackSectionDuration);
  const soundtrackSectionBeatCount = soundtrackBeatTimes.filter((beat) => beat <= soundtrackSectionDuration).length;

  const drawReel = useCallback((target: HTMLCanvasElement, time: number, maxDimension?: number) => {
    const scenes = scenesRef.current.map((scene) => scene.id === activeSceneId ? {
      ...scene,
      image: imageRef.current ?? scene.image,
      skyMask,
      semanticLayers,
      textLayers,
      timeline,
    } : scene);
    return renderReelFrame({ target, scenes, settings: reelSettings, beatTimes: soundtrackBeatTimes, time, maxDimension });
  }, [activeSceneId, reelSettings, semanticLayers, skyMask, soundtrackBeatTimes, textLayers, timeline]);

  const analyzeScene = async (scene: ReelSceneRuntime) => {
    const isActive = () => activeSceneIdRef.current === scene.id;
    scene.skyMask = null;
    scene.semanticLayers = [];
    scene.analysisQuality = "idle";
    scene.maskError = "";
    scene.maskStatus = "loading-model";
    refreshSceneList();
    if (isActive()) {
      setSkyMask(null);
      setSemanticLayers([]);
      setAnalysisQuality("idle");
      setMaskError("");
      setMaskStatus("loading-model");
    }
    try {
      const [segmenter, depthEstimator] = await Promise.all([
        getSegmenter(),
        getDepthEstimator().catch((error) => {
          console.warn("Depth model unavailable; continuing with semantic layers.", error);
          depthEstimatorPromise = null;
          return null;
        }),
      ]);
      scene.maskStatus = "analyzing";
      refreshSceneList();
      if (isActive()) setMaskStatus("analyzing");
      const scale = Math.min(1, 1024 / Math.max(scene.image.naturalWidth, scene.image.naturalHeight));
      const analysis = document.createElement("canvas");
      analysis.width = Math.max(1, Math.round(scene.image.naturalWidth * scale));
      analysis.height = Math.max(1, Math.round(scene.image.naturalHeight * scale));
      analysis.getContext("2d")!.drawImage(scene.image, 0, 0, analysis.width, analysis.height);
      const segments = await segmenter(analysis);
      const depthOutput = depthEstimator ? await depthEstimator(analysis).catch((error) => {
        console.warn("Depth analysis failed; continuing with semantic layers.", error);
        return null;
      }) : null;
      const sky = segments.find((segment) => segment.label?.toLowerCase() === "sky");
      const cleanSky = sky ? cleanSkyMask(sky.mask.data, sky.mask.width, sky.mask.height) : null;
      const nextSkyMask = sky && cleanSky ? { width: sky.mask.width, height: sky.mask.height, data: cleanSky } : null;

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
      const layerIds = layers.map((layer) => layer.id);
      scene.skyMask = nextSkyMask;
      scene.semanticLayers = layers;
      scene.textLayers = applyInstaEditTextAnimations(scene.textLayers.map((textLayer) => ({ ...textLayer, frontLayerIds: layerIds })));
      scene.timeline = createInstaEditTimeline(layerIds);
      scene.analysisQuality = depthEnhanced ? "depth" : "semantic";
      scene.maskStatus = "ready";
      scene.maskError = "";
      refreshSceneList();
      if (isActive()) {
        setSkyMask(scene.skyMask);
        setSemanticLayers(scene.semanticLayers);
        setTextLayers(scene.textLayers);
        setTimeline(scene.timeline);
        setAnalysisQuality(scene.analysisQuality);
        setMaskStatus("ready");
        setMaskError("");
      }
      return true;
    } catch (error) {
      console.error(error);
      scene.maskError = error instanceof Error ? error.message : "The vision models could not run in this browser.";
      scene.maskStatus = "error";
      refreshSceneList();
      if (isActive()) {
        setMaskError(scene.maskError);
        setMaskStatus("error");
      }
      return false;
    }
  };

  const analyzeScenesSequentially = async (scenes: ReelSceneRuntime[]) => {
    if (analysisQueueRef.current) return;
    const pending = scenes.filter((scene) => scene.maskStatus === "idle");
    if (!pending.length) return;
    analysisQueueRef.current = true;
    setAnalysisQueueRunning(true);
    let completed = 0;
    let failed = 0;
    try {
      for (const [index, scene] of pending.entries()) {
        setExportStatus(`Analyzing photo ${index + 1} of ${pending.length}: ${scene.imageName}`);
        if (await analyzeScene(scene)) completed += 1;
        else failed += 1;
      }
      setExportStatus(failed ? `${completed} photo${completed === 1 ? "" : "s"} analyzed · ${failed} could not be segmented.` : `${completed} photo${completed === 1 ? "" : "s"} analyzed — reel layers are ready.`);
    } finally {
      analysisQueueRef.current = false;
      setAnalysisQueueRunning(false);
    }
  };

  const depthStack = createDepthStack(semanticLayers, textLayers);

  const moveTextLayerToDepthIndex = (textLayerId: string, targetIndex: number) => {
    setTextLayers((current) => {
      const currentStack = createDepthStack(semanticLayers, current);
      const currentIndex = currentStack.findIndex((item) => item.kind === "text" && item.layer.id === textLayerId);
      const remaining = currentStack.filter((item) => item.kind !== "text" || item.layer.id !== textLayerId);
      const adjustedTarget = currentIndex >= 0 && targetIndex > currentIndex ? targetIndex - 1 : targetIndex;
      const insertionIndex = Math.max(0, Math.min(remaining.length, adjustedTarget));
      const movedLayer = current.find((layer) => layer.id === textLayerId);
      if (!movedLayer) return current;
      const nextStack: DepthStackItem[] = [...remaining];
      nextStack.splice(insertionIndex, 0, { kind: "text", layer: movedLayer, textIndex: 0 });
      const movedIndex = nextStack.findIndex((item) => item.kind === "text" && item.layer.id === textLayerId);
      const frontLayerIds = nextStack
        .slice(movedIndex + 1)
        .filter((item): item is Extract<DepthStackItem, { kind: "semantic" }> => item.kind === "semantic")
        .map((item) => item.layer.id)
        .reverse();
      const byId = new Map(current.map((layer) => [layer.id, layer]));
      return nextStack
        .filter((item): item is Extract<DepthStackItem, { kind: "text" }> => item.kind === "text")
        .map((item) => {
          const layer = byId.get(item.layer.id)!;
          return layer.id === textLayerId ? { ...layer, frontLayerIds } : layer;
        });
    });
  };

  const nudgeTextLayer = (textLayerId: string, direction: -1 | 1) => {
    const currentIndex = depthStack.findIndex((item) => item.kind === "text" && item.layer.id === textLayerId);
    if (currentIndex < 0) return;
    moveTextLayerToDepthIndex(textLayerId, direction < 0 ? currentIndex - 1 : currentIndex + 2);
  };

  useEffect(() => {
    const id = activeSceneIdRef.current;
    const scene = scenesRef.current.find((candidate) => candidate.id === id);
    if (!scene || !imageRef.current || !sourceFingerprintRef.current) return;
    Object.assign(scene, { image: imageRef.current, source: sourceFingerprintRef.current, imageName, dimensions, skyMask, semanticLayers, textLayers, activeTextLayerId, timeline, analysisQuality, maskStatus, maskError });
  }, [activeTextLayerId, analysisQuality, dimensions, imageName, maskError, maskStatus, semanticLayers, skyMask, textLayers, timeline]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (previewMode === "reel") drawReel(canvasRef.current, reelPlayhead, 900);
    else drawPoster(canvasRef.current, showMask, undefined, playhead);
  }, [drawPoster, drawReel, playhead, previewMode, reelPlayhead, showMask]);

  useEffect(() => () => {
    if (playbackFrame.current !== null) cancelAnimationFrame(playbackFrame.current);
    if (reelPlaybackFrame.current !== null) cancelAnimationFrame(reelPlaybackFrame.current);
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
  }, []);

  const stopPlayback = () => {
    if (playbackFrame.current !== null) cancelAnimationFrame(playbackFrame.current);
    playbackFrame.current = null;
    setIsPlaying(false);
  };

  const play = () => {
    stopReelPlayback();
    setPreviewMode("scene");
    stopPlayback();
    const startAt = playhead >= timeline.duration ? 0 : playhead;
    setIsPlaying(true);
    let startedAt: number | null = null;
    const tick = (now: number) => {
      if (startedAt === null) startedAt = now - startAt;
      const next = Math.min(timeline.duration, now - startedAt);
      setPlayhead(next);
      if (next < timeline.duration) playbackFrame.current = requestAnimationFrame(tick);
      else stopPlayback();
    };
    playbackFrame.current = requestAnimationFrame(tick);
  };

  const stopReelPlayback = () => {
    if (reelPlaybackFrame.current !== null) cancelAnimationFrame(reelPlaybackFrame.current);
    reelPlaybackFrame.current = null;
    audioElement.current?.pause();
    setIsReelPlaying(false);
  };

  const playReel = () => {
    if (!sceneList.length || reelSchedule.duration <= 0) return;
    stopPlayback();
    stopReelPlayback();
    setPreviewMode("reel");
    const startAt = reelPlayhead >= reelSchedule.duration ? 0 : reelPlayhead;
    if (audioElement.current && soundtrackName) {
      audioElement.current.currentTime = Math.min((soundtrackStart + startAt) / 1000, audioElement.current.duration || 0);
      void audioElement.current.play().catch(() => setExportStatus("Press Play again to allow soundtrack playback."));
    }
    setIsReelPlaying(true);
    let startedAt: number | null = null;
    const tick = (now: number) => {
      if (startedAt === null) startedAt = now - startAt;
      const next = Math.min(reelSchedule.duration, now - startedAt);
      setReelPlayhead(next);
      if (next < reelSchedule.duration) reelPlaybackFrame.current = requestAnimationFrame(tick);
      else stopReelPlayback();
    };
    reelPlaybackFrame.current = requestAnimationFrame(tick);
  };

  const applyInstaEdit = () => {
    stopPlayback();
    setTimeline(createInstaEditTimeline(semanticLayers.map((layer) => layer.id)));
    setTextLayers(applyInstaEditTextAnimations);
    setPlayhead(0);
    setExportStatus("Insta Edit applied — press Play to preview");
  };

  const applySlowCinema = () => {
    stopPlayback();
    const duration = DEFAULT_TIMELINE_DURATION;
    const depthOrder = [...semanticLayers].reverse();
    const gap = Math.max(90, Math.min(170, 850 / Math.max(1, depthOrder.length)));
    setTimeline({
      duration,
      backgroundColor: "#000000",
      baseAnimation: { enabled: true, effect: "zoom", delay: 70, duration: 800 },
      sceneAnimations: Object.fromEntries(depthOrder.map((layer, index) => [layer.id, {
        enabled: true,
        effect: index % 2 === 0 ? "fade" : "drift",
        delay: 500 + index * gap,
        duration: 700,
      }])),
    });
    setTextLayers((current) => current.map((layer, index) => ({
      ...layer,
      animation: { enabled: true, effect: "fade", delay: 1700 + index * Math.min(100, 400 / Math.max(1, current.length - 1)), duration: 700 },
    })));
    setPlayhead(0);
    setExportStatus("Slow Cinema applied — press Play to preview");
  };

  const applyEditorialFlash = () => {
    stopPlayback();
    const duration = DEFAULT_TIMELINE_DURATION;
    const depthOrder = [...semanticLayers].reverse();
    const gap = Math.max(90, Math.min(180, 620 / Math.max(1, depthOrder.length)));
    setTimeline({
      duration,
      backgroundColor: "#ffffff",
      baseAnimation: { enabled: true, effect: "drift", delay: 0, duration: 520 },
      sceneAnimations: Object.fromEntries(depthOrder.map((layer, index) => [layer.id, {
        enabled: true,
        effect: index % 2 === 0 ? "rise" : "drift",
        delay: 330 + index * gap,
        duration: 520,
      }])),
    });
    setTextLayers((current) => current.map((layer, index) => ({
      ...layer,
      animation: {
        enabled: true,
        effect: index % 2 === 0 ? "rise" : "reel",
        delay: 1280 + index * 130,
        duration: 520,
      },
    })));
    setPlayhead(0);
    setExportStatus("Editorial Flash applied — press Play to preview");
  };

  const decodeImage = (file: File) => new Promise<HTMLImageElement>((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => { URL.revokeObjectURL(objectUrl); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(objectUrl); reject(new Error(`${file.name} could not be opened.`)); };
    image.src = objectUrl;
  });

  const loadFiles = async (files: File[]) => {
    if (analysisQueueRef.current || maskStatus === "loading-model" || maskStatus === "analyzing") return;
    const images = files.filter((file) => file.type.startsWith("image/"));
    if (!images.length) return;
    const pendingProject = pendingProjectRef.current;
    setMaskError("");
    setExportStatus(`Preparing ${images.length} photograph${images.length === 1 ? "" : "s"}…`);
    try {
      const added: ReelSceneRuntime[] = [];
      for (const [index, file] of images.entries()) {
        const sha256 = await sha256File(file);
        if (pendingProject && index === 0 && sha256 !== pendingProject.source.sha256) {
          throw new Error(`This project expects ${pendingProject.source.name}. Choose the original matching photograph.`);
        }
        const image = await decodeImage(file);
        if (pendingProject && index === 0 && (image.naturalWidth !== pendingProject.source.width || image.naturalHeight !== pendingProject.source.height)) {
          throw new Error("The selected photograph dimensions do not match this project.");
        }
        const source = { name: file.name, size: file.size, width: image.naturalWidth, height: image.naturalHeight, sha256 };
        const dimensionsLabel = `${image.naturalWidth.toLocaleString()} × ${image.naturalHeight.toLocaleString()}`;
        const defaultText = applyInstaEditTextAnimations([createTextLayer("text-1", 1)]);
        const scene: ReelSceneRuntime = {
          id: `scene-${nextSceneId.current++}`,
          image,
          imageName: file.name,
          dimensions: dimensionsLabel,
          source,
          skyMask: null,
          semanticLayers: [],
          textLayers: defaultText,
          activeTextLayerId: defaultText[0].id,
          timeline: createInstaEditTimeline(),
          analysisQuality: "idle",
          maskStatus: "idle",
          maskError: "",
        };
        if (pendingProject && index === 0) {
          const restored = deserializeAnalysis(pendingProject.analysis);
          scene.skyMask = restored.skyMask;
          scene.semanticLayers = restored.layers;
          scene.analysisQuality = restored.quality;
          scene.textLayers = pendingProject.recipe.textLayers;
          scene.activeTextLayerId = pendingProject.recipe.activeTextLayerId;
          scene.timeline = pendingProject.recipe.timeline ?? createTimelineSettings(restored.layers.map((layer) => layer.id));
          scene.maskStatus = "ready";
        }
        added.push(scene);
      }
      captureActiveScene();
      scenesRef.current = [...scenesRef.current, ...added];
      pendingProjectRef.current = null;
      refreshSceneList();
      activateScene(added[0].id);
      setExportStatus(`${added.length} scene${added.length === 1 ? "" : "s"} added — beginning automatic analysis.`);
      await analyzeScenesSequentially(added);
    } catch (error) {
      pendingProjectRef.current = null;
      setMaskError(error instanceof Error ? error.message : "The photographs could not be prepared.");
      setMaskStatus("error");
    }
  };
  const handleFile = (event: ChangeEvent<HTMLInputElement>) => {
    const files = [...(event.target.files ?? [])];
    event.target.value = "";
    void loadFiles(files);
  };
  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    void loadFiles([...event.dataTransfer.files]);
  };
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

  const moveScene = (sceneId: string, direction: -1 | 1) => {
    captureActiveScene();
    const currentIndex = scenesRef.current.findIndex((scene) => scene.id === sceneId);
    const targetIndex = currentIndex + direction;
    if (currentIndex < 0 || targetIndex < 0 || targetIndex >= scenesRef.current.length) return;
    const next = [...scenesRef.current];
    [next[currentIndex], next[targetIndex]] = [next[targetIndex], next[currentIndex]];
    scenesRef.current = next;
    refreshSceneList();
  };

  const removeScene = (sceneId: string) => {
    if (maskStatus === "loading-model" || maskStatus === "analyzing") return;
    captureActiveScene();
    const currentIndex = scenesRef.current.findIndex((scene) => scene.id === sceneId);
    scenesRef.current = scenesRef.current.filter((scene) => scene.id !== sceneId);
    refreshSceneList();
    if (sceneId !== activeSceneIdRef.current) return;
    const next = scenesRef.current[Math.min(currentIndex, scenesRef.current.length - 1)];
    activeSceneIdRef.current = next?.id ?? null;
    setActiveSceneId(next?.id ?? null);
    if (next) {
      imageRef.current = next.image;
      sourceFingerprintRef.current = next.source;
      setImageName(next.imageName);
      setDimensions(next.dimensions);
      setSkyMask(next.skyMask);
      setSemanticLayers(next.semanticLayers);
      setTextLayers(next.textLayers);
      setActiveTextLayerId(next.activeTextLayerId);
      setTimeline(next.timeline);
      setPlayhead(next.timeline.duration);
      setAnalysisQuality(next.analysisQuality);
      setMaskStatus(next.maskStatus);
      setMaskError(next.maskError);
      if (next.maskStatus === "idle") void analyzeScenesSequentially([next]);
    } else {
      imageRef.current = null;
      sourceFingerprintRef.current = null;
      setImageName("");
      setDimensions("");
      setSkyMask(null);
      setSemanticLayers([]);
      const emptyText = applyInstaEditTextAnimations([createTextLayer("text-1", 1)]);
      setTextLayers(emptyText);
      setActiveTextLayerId(emptyText[0].id);
      setTimeline(createInstaEditTimeline());
      setPlayhead(DEFAULT_TIMELINE_DURATION);
      setAnalysisQuality("idle");
      setMaskStatus("idle");
      setMaskError("");
      setPreviewMode("scene");
    }
  };

  const handleSoundtrack = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      setExportStatus("Analyzing soundtrack beats locally…");
      const context = new AudioContext();
      const buffer = await context.decodeAudioData(await file.arrayBuffer());
      const mono = new Float32Array(buffer.length);
      for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
        const data = buffer.getChannelData(channel);
        for (let index = 0; index < data.length; index += 1) mono[index] += data[index] / buffer.numberOfChannels;
      }
      const detected = detectBeats(mono, buffer.sampleRate);
      audioBufferRef.current = buffer;
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      const url = URL.createObjectURL(file);
      audioUrlRef.current = url;
      setSoundtrackUrl(url);
      setSoundtrackName(file.name);
      setSoundtrackDuration(buffer.duration * 1000);
      setSoundtrackSectionStart(0);
      setBeatTimes(detected);
      setTempo(estimateTempo(detected));
      setReelSettings((current) => ({ ...current, beatSync: detected.length > 1 }));
      await context.close();
      setExportStatus(detected.length > 1 ? `${detected.length} beats detected — beat sync enabled.` : "Soundtrack loaded; no reliable beat grid was detected.");
    } catch (error) {
      audioBufferRef.current = null;
      setBeatTimes([]);
      setTempo(null);
      setExportStatus(error instanceof Error ? error.message : "The soundtrack could not be decoded.");
    }
  };

  const clearSoundtrack = () => {
    stopReelPlayback();
    audioBufferRef.current = null;
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    audioUrlRef.current = null;
    setSoundtrackUrl("");
    setSoundtrackName("");
    setSoundtrackDuration(0);
    setSoundtrackSectionStart(0);
    setBeatTimes([]);
    setTempo(null);
    setReelSettings((current) => ({ ...current, beatSync: false }));
  };

  const chooseSoundtrackSection = (start: number) => {
    stopReelPlayback();
    const clamped = Math.max(0, Math.min(soundtrackMaxStart, start));
    setSoundtrackSectionStart(clamped);
    if (audioElement.current) audioElement.current.currentTime = clamped / 1000;
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
          recipe: { schemaVersion: 1, activeTextLayerId, textLayers, timeline },
        };
        triggerDownload(new Blob([createProjectArchive(project)], { type: "application/octet-stream" }), `${baseName}.skyline.cfg`);
      }
    }, "image/png");
  };

  const downloadAnimation = async () => {
    if (!imageRef.current || !canvasRef.current || typeof MediaRecorder === "undefined") return;
    stopPlayback();
    setExportStatus("Rendering animation…");
    const exportCanvas = document.createElement("canvas");
    drawPoster(exportCanvas, false, undefined, 0);
    const stream = exportCanvas.captureStream(30);
    const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 12_000_000 } : undefined);
    const finished = new Promise<Blob>((resolve, reject) => {
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = () => reject(new Error("The browser could not record this animation."));
      recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
    });
    recorder.start();
    const started = performance.now();
    await new Promise<void>((resolve) => {
      const renderFrame = (now: number) => {
        const time = Math.min(timeline.duration, now - started);
        drawPoster(exportCanvas, false, undefined, time);
        setPlayhead(time);
        if (time < timeline.duration) requestAnimationFrame(renderFrame);
        else setTimeout(() => { recorder.stop(); resolve(); }, 50);
      };
      requestAnimationFrame(renderFrame);
    });
    const blob = await finished;
    const baseName = imageName.replace(/\.[^.]+$/, "") || "poster";
    triggerDownload(blob, `${baseName}-skyline-animation.webm`);
    setExportStatus("WebM saved at source resolution");
  };

  const downloadReel = async () => {
    if (!scenesRef.current.length || typeof MediaRecorder === "undefined") return;
    captureActiveScene();
    stopPlayback();
    stopReelPlayback();
    setPreviewMode("reel");
    setExportStatus("Rendering reel in real time… 0%");
    const exportCanvas = document.createElement("canvas");
    const initial = drawReel(exportCanvas, 0);
    if (!initial || initial.duration <= 0) return;
    const videoStream = exportCanvas.captureStream(30);
    let audioContext: AudioContext | null = null;
    let audioSource: AudioBufferSourceNode | null = null;
    let stream: MediaStream = videoStream;
    if (audioBufferRef.current) {
      audioContext = new AudioContext();
      const destination = audioContext.createMediaStreamDestination();
      audioSource = audioContext.createBufferSource();
      audioSource.buffer = audioBufferRef.current;
      audioSource.connect(destination);
      stream = new MediaStream([...videoStream.getVideoTracks(), ...destination.stream.getAudioTracks()]);
      await audioContext.resume();
    }
    const mimeType = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm;codecs=vp9",
      "video/webm",
    ].find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 16_000_000, audioBitsPerSecond: 192_000 } : undefined);
    const finished = new Promise<Blob>((resolve, reject) => {
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = () => reject(new Error("The browser could not record this reel."));
      recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
    });
    recorder.start(1000);
    const soundtrackOffset = soundtrackStart / 1000;
    const soundtrackAvailable = Math.max(0, (audioBufferRef.current?.duration ?? initial.duration / 1000) - soundtrackOffset);
    audioSource?.start(0, soundtrackOffset, Math.min(initial.duration / 1000, soundtrackAvailable));
    const started = performance.now();
    let lastProgress = -1;
    await new Promise<void>((resolve) => {
      const renderFrame = (now: number) => {
        const time = Math.min(initial.duration, now - started);
        drawReel(exportCanvas, time);
        setReelPlayhead(time);
        const progress = Math.floor((time / initial.duration) * 100);
        if (progress >= lastProgress + 5) {
          lastProgress = progress;
          setExportStatus(`Rendering reel in real time… ${progress}%`);
        }
        if (time < initial.duration) reelPlaybackFrame.current = requestAnimationFrame(renderFrame);
        else setTimeout(() => { recorder.stop(); resolve(); }, 120);
      };
      reelPlaybackFrame.current = requestAnimationFrame(renderFrame);
    });
    reelPlaybackFrame.current = null;
    const blob = await finished;
    try { audioSource?.stop(); } catch { /* source may already have ended */ }
    await audioContext?.close();
    triggerDownload(blob, "skyline-reel.webm");
    setExportStatus(`Reel saved · ${(initial.duration / 1000).toFixed(1)}s · ${initial.width} × ${initial.height}${audioBufferRef.current ? " · soundtrack included" : ""}`);
  };

  const updateSceneAnimation = (layerId: string, updates: Partial<LayerAnimation>) => {
    setTimeline((current) => ({ ...current, sceneAnimations: { ...current.sceneAnimations, [layerId]: { ...(current.sceneAnimations[layerId] ?? createLayerAnimation()), ...updates } } }));
  };

  const busy = analysisQueueRunning || maskStatus === "loading-model" || maskStatus === "analyzing";
  const reelReady = sceneList.length > 0 && sceneList.every((scene) => (scene.id === activeSceneId ? maskStatus : scene.maskStatus) === "ready");
  const statusText = maskStatus === "loading-model" ? "Loading segmentation + depth…" : maskStatus === "analyzing" ? "Tracing depth planes…" : maskStatus === "ready" ? `${semanticLayers.length} depth layer${semanticLayers.length === 1 ? "" : "s"} ready` : maskStatus === "error" ? "Layers unavailable" : "Waiting for image";

  return <main className="studio-shell">
    <header className="masthead"><div><p className="eyebrow">Browser-based reel maker</p><h1>Skyline Reel Studio</h1></div><p className="privacy-note">Photos and soundtracks stay in this browser. Only model files are fetched.</p></header>
    <section className="workspace">
      <aside className="control-panel" aria-label="Poster controls">
        <section className="control-section">
          <div className="panel-heading"><span className="step">01</span><div><p className="label">Reel scenes</p><p className="hint">Add any number of photographs; depth analysis runs through them one by one.</p></div></div>
          <button className="upload-button" onClick={() => fileInput.current?.click()}>＋ Add photographs</button>
          <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={handleFile} />
          {sceneList.length > 0 && <div className="scene-list" aria-label="Reel scenes">
            {sceneList.map((scene, index) => <div className={`scene-card ${scene.id === activeSceneId ? "active" : ""}`} key={scene.id}>
              <button type="button" className="scene-select" onClick={() => activateScene(scene.id)} disabled={busy} aria-pressed={scene.id === activeSceneId}>
                <span>{String(index + 1).padStart(2, "0")}</span><b>{scene.imageName}</b><small>{((scene.id === activeSceneId ? timeline.duration : scene.duration) / 1000).toFixed(1)}s · {(scene.id === activeSceneId ? maskStatus : scene.maskStatus) === "ready" ? "layers ready" : (scene.id === activeSceneId ? maskStatus : scene.maskStatus) === "idle" ? "queued for analysis" : (scene.id === activeSceneId ? maskStatus : scene.maskStatus) === "loading-model" ? "loading models" : (scene.id === activeSceneId ? maskStatus : scene.maskStatus) === "analyzing" ? "analyzing layers" : "analysis failed"}</small>
              </button>
              <div className="scene-actions">
                <button type="button" onClick={() => moveScene(scene.id, -1)} disabled={index === 0 || busy} aria-label={`Move ${scene.imageName} earlier`}>↑</button>
                <button type="button" onClick={() => moveScene(scene.id, 1)} disabled={index === sceneList.length - 1 || busy} aria-label={`Move ${scene.imageName} later`}>↓</button>
                <button type="button" onClick={() => removeScene(scene.id)} disabled={busy} aria-label={`Remove ${scene.imageName}`}>×</button>
              </div>
            </div>)}
          </div>}
          <div className="reel-global-controls">
            <label><span>Reel frame</span><select value={reelSettings.aspect} onChange={(event) => setReelSettings((current) => ({ ...current, aspect: event.target.value as ReelSettings["aspect"] }))}><option value="9:16">Vertical · 9:16</option><option value="2:3">Photo · 2:3 (4×6)</option><option value="4:5">Portrait · 4:5</option><option value="1:1">Square · 1:1</option><option value="16:9">Landscape · 16:9</option></select></label>
            <label><span>Between scenes</span><select value={reelSettings.transition} onChange={(event) => setReelSettings((current) => ({ ...current, transition: event.target.value as ReelSettings["transition"] }))}><option value="crossfade">Crossfade</option><option value="cut">Hard cut</option></select></label>
          </div>
          <div className="soundtrack-card">
            <div><b>Soundtrack</b><small>{soundtrackName ? `${soundtrackName} · ${(soundtrackDuration / 1000).toFixed(1)}s${tempo ? ` · ~${tempo} BPM` : ""}` : "Optional MP3, WAV, M4A, or browser-decodable audio"}</small></div>
            <button type="button" onClick={() => soundtrackInput.current?.click()}>{soundtrackName ? "Replace" : "Upload audio"}</button>
            {soundtrackName && <button type="button" onClick={clearSoundtrack} aria-label="Remove soundtrack">Remove</button>}
            <input ref={soundtrackInput} type="file" accept="audio/*,.mp3" hidden onChange={handleSoundtrack} />
          </div>
          {soundtrackUrl && <div className="soundtrack-section-selector">
            <div><b>Select soundtrack section</b><output>{formatTimestamp(soundtrackStart)}–{formatTimestamp(soundtrackSectionEnd)}</output></div>
            <input aria-label="Soundtrack section start" type="range" min={0} max={Math.max(1, soundtrackMaxStart)} step={100} value={soundtrackStart} disabled={soundtrackMaxStart <= 0} onChange={(event) => chooseSoundtrackSection(Number(event.target.value))} />
            <p><span>Track start</span><span>Selected window follows the reel length</span><span>{formatTimestamp(soundtrackDuration)}</span></p>
          </div>}
          {soundtrackUrl && <audio ref={audioElement} className="soundtrack-player" src={soundtrackUrl} preload="auto" controls onPlay={(event) => { const current = event.currentTarget.currentTime * 1000; if (current < soundtrackStart || current >= soundtrackSectionEnd) event.currentTarget.currentTime = soundtrackStart / 1000; }} onTimeUpdate={(event) => { if (!isReelPlaying && event.currentTarget.currentTime * 1000 >= soundtrackSectionEnd) event.currentTarget.pause(); }} />}
          <Toggle checked={reelSettings.beatSync} onChange={(beatSync) => setReelSettings((current) => ({ ...current, beatSync }))} label={soundtrackSectionBeatCount ? `Sync layer entrances to ${soundtrackSectionBeatCount} beats in this section` : "Sync layer entrances to the beat"} disabled={!soundtrackSectionBeatCount} />
          <div className="reel-player">
            <button type="button" onClick={isReelPlaying ? stopReelPlayback : playReel} disabled={!reelReady}>{isReelPlaying ? "Pause reel" : sceneList.length > 0 && reelPlayhead >= reelSchedule.duration ? "Replay reel" : "Play full reel"}</button>
            <input aria-label="Reel playhead" type="range" min={0} max={Math.max(1, reelSchedule.duration)} step={10} value={Math.min(reelPlayhead, reelSchedule.duration)} onChange={(event) => { stopReelPlayback(); setPreviewMode("reel"); setReelPlayhead(Number(event.target.value)); }} />
            <output>{(reelPlayhead / 1000).toFixed(1)} / {(reelSchedule.duration / 1000).toFixed(1)}s</output>
          </div>
        </section>
        <section className="control-section">
          <div className="panel-heading"><span className="step">02</span><div><p className="label">Active photograph</p><p className="hint">Mountains, coasts, cities, and trees all work.</p></div></div>
          <button type="button" className="project-button" onClick={() => projectInput.current?.click()}>Import .skyline.cfg project</button>
          <input ref={projectInput} type="file" accept=".cfg,.skyline.cfg,application/octet-stream" hidden onChange={(event) => { void handleProject(event); }} />
          {imageName && <p className="file-meta"><span>{imageName}</span><span>{dimensions}</span></p>}
        </section>
        <section className="control-section">
          <div className="panel-heading"><span className="step">03</span><div><p className="label">Text layers</p><p className="hint">Select a layer to edit its type and position.</p></div></div>
          <div className="text-layer-list" aria-label="Text layers">
            {textLayers.map((layer, index) => <button type="button" key={layer.id} className={layer.id === activeTextLayerId ? "active" : ""} onClick={() => setActiveTextLayerId(layer.id)} aria-pressed={layer.id === activeTextLayerId}><span>T{index + 1}</span><b>{layer.name}</b><small>{layer.text.split("\n")[0] || "Empty layer"}</small></button>)}
          </div>
          <p className="layer-order-hint">Set the complete depth order in step 4.</p>
          <button type="button" className="add-layer-button" onClick={addTextLayer}>＋ Add new text layer</button>
          {activeTextLayer && <div className="text-layer-editor">
            <div className="layer-editor-bar">
              <label><span>Layer name</span><input value={activeTextLayer.name} onChange={(event) => updateActiveTextLayer({ name: event.target.value })} aria-label="Text layer name" /></label>
              <div className="layer-actions" aria-label="Text layer actions">
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
          <div className="panel-heading"><span className="step">04</span><div><p className="label">Depth order</p><p className="hint">Every image and text layer, ordered from deepest to closest.</p></div></div>
          <div className={`mask-readout ${maskStatus}`}><span className="status-dot" />{statusText}</div>
          {maskStatus === "ready" && <p className={`quality-badge ${analysisQuality}`}>{analysisQuality === "depth" ? "Depth-enhanced analysis" : "Semantic fallback"}</p>}
          {maskError && <p className="mask-error">{maskError}</p>}
          {semanticLayers.length > 0 && <>
            <div className="depth-presets" aria-label="Depth presets">
              <button type="button" onClick={() => setTextLayers((current) => current.map((layer) => ({ ...layer, frontLayerIds: semanticLayers.map((semanticLayer) => semanticLayer.id) })))}>Text behind objects</button>
              <button type="button" onClick={() => setTextLayers((current) => current.map((layer) => ({ ...layer, frontLayerIds: [] })))}>Text on top</button>
            </div>
            <div className={`depth-stack ${draggedTextLayerId ? "is-dragging" : ""}`} aria-label="Complete layer depth order">
              <div className="stack-cap"><span>Deepest</span><span>Closest</span></div>
              <div className="base-layer"><span className="layer-swatch sky-base" /><span><b>Original photo</b><small>Base image · fixed</small></span></div>
              {depthStack.map((item, index) => <div className="depth-drop-target" key={`${item.kind}-${item.layer.id}-drop`} onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }} onDrop={(event) => { event.preventDefault(); const textLayerId = draggedTextLayerId || event.dataTransfer.getData("text/plain"); if (textLayerId) moveTextLayerToDepthIndex(textLayerId, index); setDraggedTextLayerId(null); }}>
                {item.kind === "semantic"
                  ? <DepthLayerRow layer={item.layer} />
                  : <TextDepthRow layer={item.layer} textIndex={item.textIndex} onDragStart={(event) => { setDraggedTextLayerId(item.layer.id); event.dataTransfer.effectAllowed = "move"; event.dataTransfer.setData("text/plain", item.layer.id); }} onDragEnd={() => setDraggedTextLayerId(null)} onNudge={(direction) => nudgeTextLayer(item.layer.id, direction)} />}
              </div>)}
              <div className="depth-drop-target depth-drop-end" onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }} onDrop={(event) => { event.preventDefault(); const textLayerId = draggedTextLayerId || event.dataTransfer.getData("text/plain"); if (textLayerId) moveTextLayerToDepthIndex(textLayerId, depthStack.length); setDraggedTextLayerId(null); }} />
            </div>
          </>}
          <Toggle checked={showMask} onChange={setShowMask} label="Show colored layer overlay" />
        </section>
        <section className="control-section animation-section">
          <div className="panel-heading"><span className="step">05</span><div><p className="label">Scene animation</p><p className="hint">Animate this scene; global beat sync aligns enabled entrances during reel playback and export.</p></div></div>
          <div className="motion-presets" aria-label="Motion presets">
            <article className="motion-preset preset-reel">
              <span className="preset-kicker">Social · 3 sec</span>
              <strong>Insta Edit</strong>
              <p>Depth-first parallax, soft motion blur, and a punchy type reveal.</p>
              <button type="button" onClick={applyInstaEdit} disabled={!imageName}>Apply Insta Edit</button>
            </article>
            <article className="motion-preset preset-cinema">
              <span className="preset-kicker">Film title · 3 sec</span>
              <strong>Slow Cinema</strong>
              <p>A patient push-in, layered atmosphere, and an understated title fade.</p>
              <button type="button" onClick={applySlowCinema} disabled={!imageName}>Apply Slow Cinema</button>
            </article>
            <article className="motion-preset preset-editorial">
              <span className="preset-kicker">Editorial · 3 sec</span>
              <strong>Editorial Flash</strong>
              <p>A white-flash open, sharp scene cuts, and quick staggered typography.</p>
              <button type="button" onClick={applyEditorialFlash} disabled={!imageName}>Apply Editorial Flash</button>
            </article>
          </div>
          <div className="animation-stage-controls">
            <label><span>Opening screen</span><select value={timeline.backgroundColor} onChange={(event) => setTimeline((current) => ({ ...current, backgroundColor: event.target.value as TimelineSettings["backgroundColor"] }))}><option value="#000000">Black</option><option value="#ffffff">White</option></select></label>
            <label><span>Length</span><select value={timeline.duration} onChange={(event) => { const duration = Number(event.target.value); setTimeline((current) => ({ ...current, duration })); setPlayhead((current) => Math.min(current, duration)); }}><option value={3000}>3 seconds</option><option value={5000}>5 seconds</option><option value={8000}>8 seconds</option><option value={12000}>12 seconds</option></select></label>
          </div>
          <div className="timeline-player">
            <button type="button" className="play-button" onClick={isPlaying ? stopPlayback : play} disabled={!imageName}>{isPlaying ? "Pause" : playhead >= timeline.duration ? "Replay" : "Play"}</button>
            <input aria-label="Animation playhead" type="range" min={0} max={timeline.duration} step={10} value={playhead} onChange={(event) => { stopPlayback(); setPlayhead(Number(event.target.value)); }} />
            <output>{(playhead / 1000).toFixed(1)}s</output>
          </div>
          <div className="animation-layer-list">
            <AnimationRow name="Original photo / sky" badge="BG" animation={timeline.baseAnimation} timelineDuration={timeline.duration} onChange={(updates) => setTimeline((current) => ({ ...current, baseAnimation: { ...current.baseAnimation, ...updates } }))} />
            {semanticLayers.map((layer, index) => <AnimationRow key={layer.id} name={layer.label} badge={`D${index + 1}`} animation={timeline.sceneAnimations[layer.id] ?? createLayerAnimation()} timelineDuration={timeline.duration} onChange={(updates) => updateSceneAnimation(layer.id, updates)} />)}
            {textLayers.map((layer, index) => <AnimationRow key={layer.id} name={layer.name} badge={`T${index + 1}`} animation={layer.animation ?? createLayerAnimation()} timelineDuration={timeline.duration} onChange={(updates) => setTextLayers((current) => current.map((candidate) => candidate.id === layer.id ? { ...candidate, animation: { ...(candidate.animation ?? createLayerAnimation()), ...updates } } : candidate))} />)}
          </div>
        </section>
        <div className="export-actions">
          <button className="download-button" disabled={!imageName || busy} onClick={downloadPoster}>{busy ? statusText : "Download PNG + project"}</button>
          <button className="animation-download-button" disabled={!imageName || busy} onClick={() => { void downloadAnimation().catch((error) => setExportStatus(error instanceof Error ? error.message : "Animation export failed.")); }}>Download animated WebM</button>
          <button className="reel-download-button" disabled={!reelReady || isReelPlaying} onClick={() => { void downloadReel().catch((error) => setExportStatus(error instanceof Error ? error.message : "Reel export failed.")); }}>Download full reel WebM{soundtrackName ? " + soundtrack" : ""}</button>
          {exportStatus && <p className="export-status" aria-live="polite">{exportStatus}</p>}
        </div>
      </aside>
      <section className="preview-panel" aria-label="Poster preview">
        <div className="preview-topline"><div><span className={`status-dot ${semanticLayers.length ? "ready" : ""}`} />{previewMode === "reel" ? "Full reel preview" : "Active scene canvas"}</div><span>{previewMode === "reel" ? `${sceneList.length} scene${sceneList.length === 1 ? "" : "s"} · ${(reelSchedule.duration / 1000).toFixed(1)}s` : imageName ? statusText : "Waiting for image"}</span></div>
        {!imageName ? <div className={`drop-zone ${isDragging ? "dragging" : ""}`} onClick={() => fileInput.current?.click()} onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInput.current?.click(); }}><span className="drop-mark">＋</span><strong>Drop photographs here</strong><small>or click to browse · JPEG, PNG, WebP</small></div> : <div className={`canvas-stage ${previewMode === "reel" ? "reel-mode" : ""}`}><canvas ref={canvasRef} aria-label={previewMode === "reel" ? "Full reel preview" : "Live poster preview"} />{busy && <div className="calculating-chip">{statusText}</div>}</div>}
        <footer className="preview-footer"><span><i className="key-swatch sky" />Sky / base</span><span><i className="key-swatch land" />Selected layers</span><span className="footer-tip">{previewMode === "reel" ? "Global playback uses the scene order, soundtrack, crossfades, and beat grid above." : "Edit and play this scene independently, then preview the full reel."}</span></footer>
      </section>
    </section>
  </main>;
}

function Range({ idPrefix = "poster", label, value, min, max, suffix, onChange }: { idPrefix?: string; label: string; value: number; min: number; max: number; suffix: string; onChange: (value: number) => void }) {
  const id = `range-${idPrefix}-${label.toLowerCase().replaceAll(" ", "-")}`;
  return <div className="range-field"><span><label htmlFor={id}><b>{label}</b></label><output htmlFor={id}><input aria-label={`${label} value`} type="number" min={min} max={max} value={value} onChange={(event) => onChange(Math.min(max, Math.max(min, Number(event.target.value))))} />{suffix}</output></span><input id={id} type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} /></div>;
}

function Toggle({ checked, onChange, label, disabled = false }: { checked: boolean; onChange: (value: boolean) => void; label: string; disabled?: boolean }) {
  return <label className={`toggle-row ${disabled ? "disabled" : ""}`}><input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} /><span className="toggle-track"><i /></span><b>{label}</b></label>;
}

function DepthLayerRow({ layer }: { layer: SemanticLayer }) {
  return <div className="depth-layer">
    <span className="layer-swatch" style={{ backgroundColor: `rgb(${layer.color.join(",")})` }} />
    <span className="layer-name">{layer.label}<small>Image · {Math.max(0.1, layer.coverage * 100).toFixed(1)}%</small></span>
  </div>;
}

function TextDepthRow({ layer, textIndex, onDragStart, onDragEnd, onNudge }: { layer: TextLayer; textIndex: number; onDragStart: (event: DragEvent<HTMLDivElement>) => void; onDragEnd: () => void; onNudge: (direction: -1 | 1) => void }) {
  return <div className="text-depth-layer" role="button" draggable onDragStart={onDragStart} onDragEnd={onDragEnd} tabIndex={0} onKeyDown={(event) => {
    if (event.key === "ArrowUp") { event.preventDefault(); onNudge(-1); }
    if (event.key === "ArrowDown") { event.preventDefault(); onNudge(1); }
  }} aria-label={`${layer.name}, text layer. Drag to reorder depth, or use up and down arrow keys.`}>
    <span className="text-depth-badge">T{textIndex + 1}</span>
    <span className="layer-name">{layer.name}<small>Text layer</small></span>
    <span className="drag-handle" aria-hidden="true"><i /><i /><i /></span>
  </div>;
}

function AnimationRow({ name, badge, animation, timelineDuration, onChange }: { name: string; badge: string; animation: LayerAnimation; timelineDuration: number; onChange: (updates: Partial<LayerAnimation>) => void }) {
  const latestDelay = Math.max(0, timelineDuration - Math.min(100, animation.duration));
  return <div className={`animation-row ${animation.enabled ? "active" : ""}`}>
    <div className="animation-row-title"><button type="button" className="animation-enable" onClick={() => onChange({ enabled: !animation.enabled })} aria-pressed={animation.enabled}>{animation.enabled ? "On" : "Off"}</button><span className="animation-badge">{badge}</span><b title={name}>{name}</b></div>
    {animation.enabled && <div className="animation-row-controls">
      <label><span>Entrance</span><select value={animation.effect} onChange={(event) => onChange({ effect: event.target.value as AnimationEffect })}><option value="fade">Fade</option><option value="rise">Rise</option><option value="drift">Drift</option><option value="zoom">Zoom</option><option value="reel">Reel push</option></select></label>
      <label><span>Starts</span><input type="number" min={0} max={latestDelay / 1000} step={0.1} value={animation.delay / 1000} onChange={(event) => onChange({ delay: Math.max(0, Math.min(latestDelay, Number(event.target.value) * 1000)) })} /><i>s</i></label>
      <label><span>Takes</span><input type="number" min={0.1} max={timelineDuration / 1000} step={0.1} value={animation.duration / 1000} onChange={(event) => onChange({ duration: Math.max(100, Math.min(timelineDuration, Number(event.target.value) * 1000)) })} /><i>s</i></label>
    </div>}
  </div>;
}

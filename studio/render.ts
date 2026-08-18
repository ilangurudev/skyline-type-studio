import type { BinaryMask, LayerAnimation, SemanticLayer, TextLayer, TimelineSettings } from "./types";

const maskCanvasCache = new WeakMap<BinaryMask, HTMLCanvasElement>();
const fittedMaskCache = new WeakMap<BinaryMask, Map<string, HTMLCanvasElement>>();
const maskedImageCache = new WeakMap<HTMLImageElement, WeakMap<BinaryMask, Map<string, HTMLCanvasElement>>>();
const inverseMaskedImageCache = new WeakMap<HTMLImageElement, WeakMap<BinaryMask, Map<string, HTMLCanvasElement>>>();
const mergedSceneMaskCache = new WeakMap<SemanticLayer[], BinaryMask | null>();

function createMaskCanvas(mask: BinaryMask) {
  const cached = maskCanvasCache.get(mask);
  if (cached) return cached;
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
  maskCanvasCache.set(mask, canvas);
  return canvas;
}

function mergeMasks(layers: SemanticLayer[], layerIds: string[]): BinaryMask | null {
  const selected = layers.filter((layer) => layerIds.includes(layer.id));
  if (!selected.length) return null;
  const { width, height } = selected[0];
  const data = new Uint8ClampedArray(width * height);
  for (const layer of selected) {
    for (let index = 0; index < data.length; index += 1) if (layer.data[index] > data[index]) data[index] = layer.data[index];
  }
  return { width, height, data };
}

function allSceneMask(layers: SemanticLayer[]) {
  if (mergedSceneMaskCache.has(layers)) return mergedSceneMaskCache.get(layers) ?? null;
  const mask = mergeMasks(layers, layers.map((layer) => layer.id));
  mergedSceneMaskCache.set(layers, mask);
  return mask;
}

const STATIC_ANIMATION: LayerAnimation = { enabled: false, effect: "fade", delay: 0, duration: 1 };

function animationFrame(animation: LayerAnimation | undefined, time: number | undefined, width: number, height: number) {
  if (time === undefined || !animation?.enabled) return { alpha: 1, x: 0, y: 0, scale: 1, rotation: 0, blur: 0 };
  const raw = Math.max(0, Math.min(1, (time - animation.delay) / Math.max(1, animation.duration)));
  const progress = 1 - Math.pow(1 - raw, 3);
  if (animation.effect === "rise") return { alpha: progress, x: 0, y: (1 - progress) * height * 0.055, scale: 1, rotation: 0, blur: 0 };
  if (animation.effect === "drift") return { alpha: progress, x: (1 - progress) * -width * 0.045, y: 0, scale: 1, rotation: 0, blur: 0 };
  if (animation.effect === "zoom") return { alpha: progress, x: 0, y: 0, scale: 0.92 + progress * 0.08, rotation: 0, blur: 0 };
  if (animation.effect === "reel") return {
    alpha: Math.min(1, raw * 1.7),
    x: (1 - progress) * -width * 0.018,
    y: (1 - progress) * height * 0.035,
    scale: 1.085 - progress * 0.085,
    rotation: (1 - progress) * -0.45,
    blur: (1 - progress) * 14,
  };
  return { alpha: progress, x: 0, y: 0, scale: 1, rotation: 0, blur: 0 };
}

function animationFinished(animation: LayerAnimation | undefined, time: number) {
  return !animation?.enabled || time >= animation.delay + animation.duration;
}

function drawTransformed(ctx: CanvasRenderingContext2D, source: CanvasImageSource, frame: ReturnType<typeof animationFrame>, width: number, height: number, operation?: GlobalCompositeOperation) {
  if (frame.alpha <= 0) return;
  ctx.save();
  if (operation) ctx.globalCompositeOperation = operation;
  ctx.globalAlpha = frame.alpha;
  ctx.filter = frame.blur > 0.1 ? `blur(${frame.blur}px)` : "none";
  ctx.translate(width / 2 + frame.x, height / 2 + frame.y);
  ctx.rotate((frame.rotation * Math.PI) / 180);
  ctx.scale(frame.scale, frame.scale);
  ctx.drawImage(source, -width / 2, -height / 2, width, height);
  ctx.restore();
}

function drawCover(ctx: CanvasRenderingContext2D, source: CanvasImageSource, sourceWidth: number, sourceHeight: number, width: number, height: number) {
  const scale = Math.max(width / sourceWidth, height / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  ctx.drawImage(source, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
}

function fittedMask(mask: BinaryMask, width: number, height: number, cover: boolean) {
  const source = createMaskCanvas(mask);
  if (!cover) return source;
  const cacheKey = `${width}x${height}`;
  const cached = fittedMaskCache.get(mask)?.get(cacheKey);
  if (cached) return cached;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  drawCover(canvas.getContext("2d")!, source, mask.width, mask.height, width, height);
  const entries = fittedMaskCache.get(mask) ?? new Map<string, HTMLCanvasElement>();
  entries.set(cacheKey, canvas);
  fittedMaskCache.set(mask, entries);
  return canvas;
}

function maskedImage(image: HTMLImageElement, mask: BinaryMask, width: number, height: number, cover: boolean) {
  const cacheKey = `${width}x${height}:${cover ? "cover" : "stretch"}`;
  const byMask = maskedImageCache.get(image) ?? new WeakMap<BinaryMask, Map<string, HTMLCanvasElement>>();
  const cached = byMask.get(mask)?.get(cacheKey);
  if (cached) return cached;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  if (cover) drawCover(ctx, image, image.naturalWidth, image.naturalHeight, width, height);
  else ctx.drawImage(image, 0, 0, width, height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(fittedMask(mask, width, height, cover), 0, 0, width, height);
  const entries = byMask.get(mask) ?? new Map<string, HTMLCanvasElement>();
  entries.set(cacheKey, canvas);
  byMask.set(mask, entries);
  maskedImageCache.set(image, byMask);
  return canvas;
}

function imageWithoutMask(image: HTMLImageElement, mask: BinaryMask, width: number, height: number, cover: boolean) {
  const cacheKey = `${width}x${height}:${cover ? "cover" : "stretch"}`;
  const byMask = inverseMaskedImageCache.get(image) ?? new WeakMap<BinaryMask, Map<string, HTMLCanvasElement>>();
  const cached = byMask.get(mask)?.get(cacheKey);
  if (cached) return cached;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  if (cover) drawCover(ctx, image, image.naturalWidth, image.naturalHeight, width, height);
  else ctx.drawImage(image, 0, 0, width, height);
  ctx.globalCompositeOperation = "destination-out";
  ctx.drawImage(fittedMask(mask, width, height, cover), 0, 0, width, height);
  const entries = byMask.get(mask) ?? new Map<string, HTMLCanvasElement>();
  entries.set(cacheKey, canvas);
  byMask.set(mask, entries);
  inverseMaskedImageCache.set(image, byMask);
  return canvas;
}

export function renderPoster({
  target,
  image,
  textLayers,
  semanticLayers,
  skyMask,
  activeTextLayerId,
  maskOverlay = false,
  maxDimension,
  outputDimensions,
  time,
  timeline,
}: {
  target: HTMLCanvasElement;
  image: HTMLImageElement;
  textLayers: TextLayer[];
  semanticLayers: SemanticLayer[];
  skyMask: BinaryMask | null;
  activeTextLayerId?: string;
  maskOverlay?: boolean;
  maxDimension?: number;
  outputDimensions?: { width: number; height: number };
  time?: number;
  timeline?: TimelineSettings;
}) {
  const longestEdge = Math.max(image.naturalWidth, image.naturalHeight);
  const scale = maxDimension ? Math.min(1, maxDimension / longestEdge) : 1;
  const width = outputDimensions?.width ?? Math.max(1, Math.round(image.naturalWidth * scale));
  const height = outputDimensions?.height ?? Math.max(1, Math.round(image.naturalHeight * scale));
  const cover = Boolean(outputDimensions);
  target.width = width;
  target.height = height;
  const ctx = target.getContext("2d");
  if (!ctx) throw new Error("The browser could not allocate the export canvas.");
  if (time !== undefined && timeline) {
    ctx.fillStyle = timeline.backgroundColor;
    ctx.fillRect(0, 0, width, height);
    const mergedMask = allSceneMask(semanticLayers);
    const baseCanvas = mergedMask ? imageWithoutMask(image, mergedMask, width, height, cover) : document.createElement("canvas");
    if (!mergedMask) {
      baseCanvas.width = width;
      baseCanvas.height = height;
      const baseCtx = baseCanvas.getContext("2d")!;
      if (cover) drawCover(baseCtx, image, image.naturalWidth, image.naturalHeight, width, height);
      else baseCtx.drawImage(image, 0, 0, width, height);
    }
    drawTransformed(ctx, baseCanvas, animationFrame(timeline.baseAnimation, time, width, height), width, height);
    for (const layer of [...semanticLayers].reverse()) {
      const animation = timeline.sceneAnimations[layer.id] ?? STATIC_ANIMATION;
      drawTransformed(ctx, maskedImage(image, layer, width, height, cover), animationFrame(animation, time, width, height), width, height);
    }
    const sceneSettled = animationFinished(timeline.baseAnimation, time)
      && semanticLayers.every((layer) => animationFinished(timeline.sceneAnimations[layer.id] ?? STATIC_ANIMATION, time));
    if (sceneSettled) {
      if (cover) drawCover(ctx, image, image.naturalWidth, image.naturalHeight, width, height);
      else ctx.drawImage(image, 0, 0, width, height);
    }
  } else {
    if (cover) drawCover(ctx, image, image.naturalWidth, image.naturalHeight, width, height);
    else ctx.drawImage(image, 0, 0, width, height);
  }

  for (const textLayer of textLayers) {
    const typeCanvas = document.createElement("canvas");
    typeCanvas.width = width;
    typeCanvas.height = height;
    const typeCtx = typeCanvas.getContext("2d");
    if (!typeCtx) throw new Error("The browser could not allocate a text canvas.");
    let sizePx = (textLayer.fontSize / 100) * height;
    const lines = textLayer.text.split("\n");
    const x = (textLayer.xPosition / 100) * width;
    typeCtx.font = `900 ${sizePx}px ${textLayer.font}`;
    if (outputDimensions) {
      const availableWidth = textLayer.alignment === "center" ? Math.min(x, width - x) * 2 : textLayer.alignment === "left" ? width - x : x;
      const longestLine = Math.max(1, ...lines.map((line) => typeCtx.measureText(line).width));
      sizePx *= Math.min(1, (availableWidth * 0.9) / longestLine);
      typeCtx.font = `900 ${sizePx}px ${textLayer.font}`;
    }
    const lineHeight = sizePx * (1 + textLayer.lineGap / 100);
    const totalHeight = Math.max(sizePx, (lines.length - 1) * lineHeight + sizePx);
    const startBaseline = (textLayer.yPosition / 100) * height - totalHeight / 2 + sizePx * 0.82;
    typeCtx.textAlign = textLayer.alignment;
    typeCtx.textBaseline = "alphabetic";
    typeCtx.lineJoin = "round";
    const angle = (textLayer.extrusionAngle * Math.PI) / 180;
    const extrusionLength = textLayer.extrusion ? sizePx * (textLayer.extrusionDepth / 100) : 0;
    const extrusionX = Math.cos(angle) * extrusionLength;
    const extrusionY = Math.sin(angle) * extrusionLength;
    lines.forEach((line, index) => {
      const baseline = startBaseline + index * lineHeight;
      if (textLayer.shadow) {
        typeCtx.fillStyle = textLayer.shadowColor;
        const offset = Math.max(3, sizePx * 0.045);
        typeCtx.fillText(line, x + extrusionX + offset, baseline + extrusionY + offset);
      }
      if (textLayer.extrusion && extrusionLength > 0) {
        typeCtx.fillStyle = textLayer.extrusionColor;
        const steps = Math.max(1, Math.ceil(extrusionLength));
        for (let step = steps; step >= 1; step -= 1) {
          const progress = step / steps;
          typeCtx.fillText(line, x + extrusionX * progress, baseline + extrusionY * progress);
        }
      }
      typeCtx.fillStyle = textLayer.textColor;
      typeCtx.fillText(line, x, baseline);
    });

    const frontLayers = semanticLayers.filter((layer) => textLayer.frontLayerIds.includes(layer.id));
    if (frontLayers.length) {
      typeCtx.globalCompositeOperation = "destination-out";
      typeCtx.imageSmoothingEnabled = true;
      for (const layer of frontLayers) {
        const animation = timeline?.sceneAnimations[layer.id] ?? STATIC_ANIMATION;
        drawTransformed(typeCtx, fittedMask(layer, width, height, cover), animationFrame(animation, time, width, height), width, height, "destination-out");
      }
    }
    drawTransformed(ctx, typeCanvas, animationFrame(textLayer.animation, time, width, height), width, height);
  }

  if (maskOverlay && (skyMask || semanticLayers.length)) {
    const reference = skyMask ?? semanticLayers[0];
    const overlay = document.createElement("canvas");
    overlay.width = reference.width;
    overlay.height = reference.height;
    const overlayCtx = overlay.getContext("2d")!;
    const pixels = overlayCtx.createImageData(reference.width, reference.height);
    const active = textLayers.find((layer) => layer.id === activeTextLayerId) ?? textLayers[0];
    for (let index = 0; index < reference.width * reference.height; index += 1) {
      const offset = index * 4;
      if (skyMask?.data[index]) {
        pixels.data[offset] = 73; pixels.data[offset + 1] = 175; pixels.data[offset + 2] = 255; pixels.data[offset + 3] = 64;
      }
      for (const layer of semanticLayers) {
        if (!layer.data[index]) continue;
        pixels.data[offset] = layer.color[0]; pixels.data[offset + 1] = layer.color[1]; pixels.data[offset + 2] = layer.color[2];
        pixels.data[offset + 3] = active?.frontLayerIds.includes(layer.id) ? 104 : 42;
        break;
      }
    }
    overlayCtx.putImageData(pixels, 0, 0);
    if (cover) drawCover(ctx, overlay, reference.width, reference.height, width, height);
    else ctx.drawImage(overlay, 0, 0, width, height);
  }
  return { width, height };
}

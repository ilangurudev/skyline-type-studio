import type { BinaryMask, LayerAnimation, SemanticLayer, TextLayer, TimelineSettings } from "./types";

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
    for (let index = 0; index < data.length; index += 1) if (layer.data[index] > data[index]) data[index] = layer.data[index];
  }
  return { width, height, data };
}

const STATIC_ANIMATION: LayerAnimation = { enabled: false, effect: "fade", delay: 0, duration: 1 };

function animationFrame(animation: LayerAnimation | undefined, time: number | undefined, width: number, height: number) {
  if (time === undefined || !animation?.enabled) return { alpha: 1, x: 0, y: 0, scale: 1 };
  const raw = Math.max(0, Math.min(1, (time - animation.delay) / Math.max(1, animation.duration)));
  const progress = 1 - Math.pow(1 - raw, 3);
  if (animation.effect === "rise") return { alpha: progress, x: 0, y: (1 - progress) * height * 0.055, scale: 1 };
  if (animation.effect === "drift") return { alpha: progress, x: (1 - progress) * -width * 0.045, y: 0, scale: 1 };
  if (animation.effect === "zoom") return { alpha: progress, x: 0, y: 0, scale: 0.92 + progress * 0.08 };
  return { alpha: progress, x: 0, y: 0, scale: 1 };
}

function drawTransformed(ctx: CanvasRenderingContext2D, source: CanvasImageSource, frame: ReturnType<typeof animationFrame>, width: number, height: number, operation?: GlobalCompositeOperation) {
  if (frame.alpha <= 0) return;
  ctx.save();
  if (operation) ctx.globalCompositeOperation = operation;
  ctx.globalAlpha = frame.alpha;
  ctx.translate(width / 2 + frame.x, height / 2 + frame.y);
  ctx.scale(frame.scale, frame.scale);
  ctx.drawImage(source, -width / 2, -height / 2, width, height);
  ctx.restore();
}

function maskedImage(image: HTMLImageElement, mask: BinaryMask, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(image, 0, 0, width, height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(createMaskCanvas(mask), 0, 0, width, height);
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
  time?: number;
  timeline?: TimelineSettings;
}) {
  const longestEdge = Math.max(image.naturalWidth, image.naturalHeight);
  const scale = maxDimension ? Math.min(1, maxDimension / longestEdge) : 1;
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  target.width = width;
  target.height = height;
  const ctx = target.getContext("2d");
  if (!ctx) throw new Error("The browser could not allocate the export canvas.");
  if (time !== undefined && timeline) {
    ctx.fillStyle = timeline.backgroundColor;
    ctx.fillRect(0, 0, width, height);
    const allSceneMask = mergeMasks(semanticLayers, semanticLayers.map((layer) => layer.id));
    const baseCanvas = document.createElement("canvas");
    baseCanvas.width = width;
    baseCanvas.height = height;
    const baseCtx = baseCanvas.getContext("2d")!;
    baseCtx.drawImage(image, 0, 0, width, height);
    if (allSceneMask) {
      baseCtx.globalCompositeOperation = "destination-out";
      baseCtx.drawImage(createMaskCanvas(allSceneMask), 0, 0, width, height);
    }
    drawTransformed(ctx, baseCanvas, animationFrame(timeline.baseAnimation, time, width, height), width, height);
    for (const layer of [...semanticLayers].reverse()) {
      const animation = timeline.sceneAnimations[layer.id] ?? STATIC_ANIMATION;
      drawTransformed(ctx, maskedImage(image, layer, width, height), animationFrame(animation, time, width, height), width, height);
    }
  } else {
    ctx.drawImage(image, 0, 0, width, height);
  }

  for (const textLayer of textLayers) {
    const typeCanvas = document.createElement("canvas");
    typeCanvas.width = width;
    typeCanvas.height = height;
    const typeCtx = typeCanvas.getContext("2d");
    if (!typeCtx) throw new Error("The browser could not allocate a text canvas.");
    const sizePx = (textLayer.fontSize / 100) * height;
    const lines = textLayer.text.split("\n");
    const lineHeight = sizePx * (1 + textLayer.lineGap / 100);
    const totalHeight = Math.max(sizePx, (lines.length - 1) * lineHeight + sizePx);
    const startBaseline = (textLayer.yPosition / 100) * height - totalHeight / 2 + sizePx * 0.82;
    const x = (textLayer.xPosition / 100) * width;
    typeCtx.font = `900 ${sizePx}px ${textLayer.font}`;
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
        drawTransformed(typeCtx, createMaskCanvas(layer), animationFrame(animation, time, width, height), width, height, "destination-out");
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
    ctx.drawImage(overlay, 0, 0, width, height);
  }
  return { width, height };
}

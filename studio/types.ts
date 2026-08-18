export type TextAlign = "left" | "center" | "right";
export type AnimationEffect = "fade" | "rise" | "drift" | "zoom";
export type LayerAnimation = {
  enabled: boolean;
  effect: AnimationEffect;
  delay: number;
  duration: number;
};

export type TimelineSettings = {
  duration: number;
  backgroundColor: "#000000" | "#ffffff";
  baseAnimation: LayerAnimation;
  sceneAnimations: Record<string, LayerAnimation>;
};

export type TextLayer = {
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
  animation: LayerAnimation;
};

export type BinaryMask = { width: number; height: number; data: Uint8ClampedArray };
export type BaseSemanticMask = BinaryMask & { label: string; sourceIndex: number; averageY: number };
export type SemanticLayer = BinaryMask & {
  id: string;
  label: string;
  color: [number, number, number];
  coverage: number;
  depthScore: number;
};
export type AnalysisQuality = "semantic" | "depth";
export type MaskStatus = "idle" | "loading-model" | "analyzing" | "ready" | "error";

export type SerializedMask = { width: number; height: number; data: string };
export type SerializedSemanticLayer = Omit<SemanticLayer, "data"> & { data: string };
export type CachedAnalysis = {
  quality: AnalysisQuality;
  skyMask: SerializedMask | null;
  layers: SerializedSemanticLayer[];
};

export type StudioRecipeV1 = {
  schemaVersion: 1;
  activeTextLayerId: string;
  textLayers: TextLayer[];
  timeline?: TimelineSettings;
};

export type SourceFingerprint = {
  name: string;
  size: number;
  width: number;
  height: number;
  sha256: string;
};

export type SkylineProjectV1 = {
  schemaVersion: 1;
  createdAt: string;
  source: SourceFingerprint;
  analysis: CachedAnalysis;
  recipe: StudioRecipeV1;
};

export const FONT_OPTIONS = [
  ["Impact", "Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif"],
  ["Arial Black", "'Arial Black', Arial, sans-serif"],
  ["Helvetica", "Helvetica, Arial, sans-serif"],
  ["Georgia", "Georgia, serif"],
  ["Times", "'Times New Roman', Times, serif"],
  ["Courier", "'Courier New', monospace"],
] as const;

export function createTextLayer(id: string, index: number, frontLayerIds: string[] = []): TextLayer {
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
    animation: { enabled: true, effect: "rise", delay: 500 + (index - 1) * 250, duration: 900 },
  };
}

export function createLayerAnimation(delay = 0): LayerAnimation {
  return { enabled: true, effect: "fade", delay, duration: 900 };
}

export function createTimelineSettings(layerIds: string[] = []): TimelineSettings {
  return {
    duration: 5000,
    backgroundColor: "#000000",
    baseAnimation: createLayerAnimation(0),
    sceneAnimations: Object.fromEntries(layerIds.map((id, index) => [id, createLayerAnimation(250 + index * 250)])),
  };
}

export function createRecipe(frontLayerIds: string[] = []): StudioRecipeV1 {
  const textLayer = createTextLayer("text-1", 1, frontLayerIds);
  return { schemaVersion: 1, activeTextLayerId: textLayer.id, textLayers: [textLayer], timeline: createTimelineSettings(frontLayerIds) };
}

function requireNumber(value: unknown, name: string, min: number, max: number) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${name} must be a number from ${min} to ${max}.`);
  }
}

export function validateRecipe(value: unknown): asserts value is StudioRecipeV1 {
  if (!value || typeof value !== "object") throw new Error("Recipe must be an object.");
  const recipe = value as Partial<StudioRecipeV1>;
  if (recipe.schemaVersion !== 1) throw new Error("Unsupported recipe schema version.");
  if (!Array.isArray(recipe.textLayers) || recipe.textLayers.length === 0) throw new Error("Recipe needs at least one text layer.");
  const ids = new Set<string>();
  for (const [index, layer] of recipe.textLayers.entries()) {
    if (!layer || typeof layer !== "object") throw new Error(`Text layer ${index + 1} is invalid.`);
    if (typeof layer.id !== "string" || !layer.id || ids.has(layer.id)) throw new Error(`Text layer ${index + 1} needs a unique id.`);
    ids.add(layer.id);
    if (typeof layer.name !== "string" || typeof layer.text !== "string" || typeof layer.font !== "string") throw new Error(`Text layer ${layer.id} has invalid text fields.`);
    if (!(["left", "center", "right"] as unknown[]).includes(layer.alignment)) throw new Error(`Text layer ${layer.id} has an invalid alignment.`);
    requireNumber(layer.fontSize, `${layer.id}.fontSize`, 1, 100);
    requireNumber(layer.lineGap, `${layer.id}.lineGap`, -90, 300);
    requireNumber(layer.xPosition, `${layer.id}.xPosition`, 0, 100);
    requireNumber(layer.yPosition, `${layer.id}.yPosition`, 0, 100);
    requireNumber(layer.extrusionDepth, `${layer.id}.extrusionDepth`, 0, 100);
    requireNumber(layer.extrusionAngle, `${layer.id}.extrusionAngle`, -360, 360);
    for (const colorKey of ["textColor", "shadowColor", "extrusionColor"] as const) {
      if (typeof layer[colorKey] !== "string" || !/^#[0-9a-f]{6}$/i.test(layer[colorKey])) throw new Error(`${layer.id}.${colorKey} must be a six-digit hex color.`);
    }
    if (typeof layer.shadow !== "boolean" || typeof layer.extrusion !== "boolean" || !Array.isArray(layer.frontLayerIds) || !layer.frontLayerIds.every((id) => typeof id === "string")) {
      throw new Error(`Text layer ${layer.id} has invalid effect or depth fields.`);
    }
    if (layer.animation !== undefined) validateAnimation(layer.animation, `${layer.id}.animation`);
  }
  if (typeof recipe.activeTextLayerId !== "string" || !ids.has(recipe.activeTextLayerId)) throw new Error("activeTextLayerId must identify an existing text layer.");
  if (recipe.timeline !== undefined) {
    requireNumber(recipe.timeline.duration, "timeline.duration", 500, 60000);
    if (!["#000000", "#ffffff"].includes(recipe.timeline.backgroundColor)) throw new Error("timeline.backgroundColor must be black or white.");
    validateAnimation(recipe.timeline.baseAnimation, "timeline.baseAnimation");
    if (!recipe.timeline.sceneAnimations || typeof recipe.timeline.sceneAnimations !== "object") throw new Error("timeline.sceneAnimations must be an object.");
    for (const [id, animation] of Object.entries(recipe.timeline.sceneAnimations)) validateAnimation(animation, `timeline.sceneAnimations.${id}`);
  }
}

function validateAnimation(value: unknown, name: string): asserts value is LayerAnimation {
  if (!value || typeof value !== "object") throw new Error(`${name} must be an animation object.`);
  const animation = value as Partial<LayerAnimation>;
  if (typeof animation.enabled !== "boolean" || !(["fade", "rise", "drift", "zoom"] as unknown[]).includes(animation.effect)) throw new Error(`${name} has invalid animation fields.`);
  requireNumber(animation.delay, `${name}.delay`, 0, 60000);
  requireNumber(animation.duration, `${name}.duration`, 100, 60000);
}

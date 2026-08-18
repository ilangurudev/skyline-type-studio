import { renderPoster } from "./render";
import { buildReelTimeline, reelDimensions, resolveReelFrame, syncSceneToBeats } from "./reel";
import type { ReelSettings } from "./reel";
import type { BinaryMask, SemanticLayer, TextLayer, TimelineSettings } from "./types";

export type RenderableReelScene = {
  id: string;
  image: HTMLImageElement;
  skyMask: BinaryMask | null;
  semanticLayers: SemanticLayer[];
  textLayers: TextLayer[];
  timeline: TimelineSettings;
};

function drawFrame(ctx: CanvasRenderingContext2D, source: HTMLCanvasElement, width: number, height: number, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.drawImage(source, 0, 0, width, height);
  ctx.restore();
}

function renderScene(scene: RenderableReelScene, localTime: number, beats: number[], dimensions: { width: number; height: number }) {
  const canvas = document.createElement("canvas");
  const synced = beats.length ? syncSceneToBeats(scene.timeline, scene.textLayers, scene.semanticLayers.map((layer) => layer.id), beats) : null;
  renderPoster({
    target: canvas,
    image: scene.image,
    textLayers: synced?.textLayers ?? scene.textLayers,
    semanticLayers: scene.semanticLayers,
    skyMask: scene.skyMask,
    timeline: synced?.timeline ?? scene.timeline,
    time: localTime,
    outputDimensions: dimensions,
  });
  return canvas;
}

export function renderReelFrame({
  target,
  scenes,
  settings,
  beatTimes,
  time,
  maxDimension,
}: {
  target: HTMLCanvasElement;
  scenes: RenderableReelScene[];
  settings: ReelSettings;
  beatTimes: number[];
  time: number;
  maxDimension?: number;
}) {
  const dimensions = reelDimensions(settings.aspect, maxDimension);
  target.width = dimensions.width;
  target.height = dimensions.height;
  const ctx = target.getContext("2d");
  if (!ctx) throw new Error("The browser could not allocate the reel canvas.");
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, dimensions.width, dimensions.height);
  if (!scenes.length) return { ...dimensions, duration: 0, sceneIndex: 0 };
  const schedule = buildReelTimeline(scenes.map((scene) => ({ id: scene.id, duration: scene.timeline.duration })), beatTimes, settings.beatSync);
  const frame = resolveReelFrame(schedule.scenes, time, settings.transition === "crossfade" ? settings.transitionDuration : 0);
  const timing = schedule.scenes[frame.sceneIndex];
  const localBeats = settings.beatSync ? beatTimes.filter((beat) => beat >= timing.start && beat < timing.end).map((beat) => (beat - timing.start) * timing.sourceDuration / Math.max(1, timing.duration)) : [];
  const sceneCanvas = renderScene(scenes[frame.sceneIndex], frame.localTime, localBeats, dimensions);
  drawFrame(ctx, sceneCanvas, dimensions.width, dimensions.height);
  if (frame.nextSceneIndex !== null && frame.transitionProgress > 0) {
    const nextTiming = schedule.scenes[frame.nextSceneIndex];
    const nextBeats = settings.beatSync ? beatTimes.filter((beat) => beat >= nextTiming.start && beat < nextTiming.end).map((beat) => (beat - nextTiming.start) * nextTiming.sourceDuration / Math.max(1, nextTiming.duration)) : [];
    const nextCanvas = renderScene(scenes[frame.nextSceneIndex], frame.nextLocalTime, nextBeats, dimensions);
    drawFrame(ctx, nextCanvas, dimensions.width, dimensions.height, frame.transitionProgress);
  }
  return { ...dimensions, duration: schedule.duration, sceneIndex: frame.sceneIndex };
}

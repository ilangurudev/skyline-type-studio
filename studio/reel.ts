import type { LayerAnimation, TextLayer, TimelineSettings } from "./types";

export type ReelAspect = "9:16" | "2:3" | "4:5" | "1:1" | "16:9";
export type ReelTransition = "cut" | "crossfade";

export type ReelSettings = {
  aspect: ReelAspect;
  transition: ReelTransition;
  transitionDuration: number;
  beatSync: boolean;
};

export type ReelSceneTiming = {
  id: string;
  start: number;
  end: number;
  duration: number;
  sourceDuration: number;
};

export type ReelFrame = {
  sceneIndex: number;
  localTime: number;
  nextSceneIndex: number | null;
  nextLocalTime: number;
  transitionProgress: number;
};

export const DEFAULT_REEL_SETTINGS: ReelSettings = {
  aspect: "9:16",
  transition: "crossfade",
  transitionDuration: 320,
  beatSync: false,
};

const ASPECT_DIMENSIONS: Record<ReelAspect, { width: number; height: number }> = {
  "9:16": { width: 1080, height: 1920 },
  "2:3": { width: 1200, height: 1800 },
  "4:5": { width: 1080, height: 1350 },
  "1:1": { width: 1080, height: 1080 },
  "16:9": { width: 1920, height: 1080 },
};

export function reelDimensions(aspect: ReelAspect, maxDimension?: number) {
  const base = ASPECT_DIMENSIONS[aspect];
  if (!maxDimension) return base;
  const scale = Math.min(1, maxDimension / Math.max(base.width, base.height));
  return { width: Math.round(base.width * scale), height: Math.round(base.height * scale) };
}

export function selectSoundtrackBeats(beatTimes: number[], sectionStart: number) {
  const start = Math.max(0, sectionStart);
  return beatTimes.filter((beat) => beat >= start).map((beat) => beat - start);
}

function nearestBeat(beats: number[], target: number, lowerBound: number) {
  let best: number | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const beat of beats) {
    if (beat < lowerBound) continue;
    const distance = Math.abs(beat - target);
    if (distance < bestDistance) {
      best = beat;
      bestDistance = distance;
    }
    if (beat > target && distance > bestDistance) break;
  }
  return best;
}

export function buildReelTimeline(
  scenes: Array<{ id: string; duration: number }>,
  beatTimes: number[] = [],
  beatSync = false,
) {
  const timings: ReelSceneTiming[] = [];
  let cursor = 0;
  for (const scene of scenes) {
    const sourceDuration = Math.max(500, scene.duration);
    let duration = sourceDuration;
    if (beatSync && beatTimes.length > 1) {
      const snappedEnd = nearestBeat(beatTimes, cursor + sourceDuration, cursor + Math.min(1000, sourceDuration * 0.55));
      if (snappedEnd !== null && Math.abs(snappedEnd - (cursor + sourceDuration)) <= Math.min(1250, sourceDuration * 0.35)) {
        duration = snappedEnd - cursor;
      }
    }
    timings.push({ id: scene.id, start: cursor, end: cursor + duration, duration, sourceDuration });
    cursor += duration;
  }
  return { scenes: timings, duration: cursor };
}

export function resolveReelFrame(timings: ReelSceneTiming[], time: number, transitionDuration: number): ReelFrame {
  if (!timings.length) return { sceneIndex: 0, localTime: 0, nextSceneIndex: null, nextLocalTime: 0, transitionProgress: 0 };
  const clamped = Math.max(0, Math.min(time, timings[timings.length - 1].end));
  let sceneIndex = timings.findIndex((scene) => clamped < scene.end);
  if (sceneIndex < 0) sceneIndex = timings.length - 1;
  const scene = timings[sceneIndex];
  const elapsed = Math.max(0, clamped - scene.start);
  const incomingTransition = sceneIndex > 0
    ? Math.min(Math.max(0, transitionDuration), timings[sceneIndex - 1].duration * 0.45)
    : 0;
  const localTime = Math.min(scene.sourceDuration, (elapsed + incomingTransition) * (scene.sourceDuration / Math.max(1, scene.duration)));
  const hasNext = sceneIndex < timings.length - 1;
  const transition = hasNext ? Math.min(Math.max(0, transitionDuration), scene.duration * 0.45) : 0;
  const transitionProgress = transition > 0 ? Math.max(0, Math.min(1, (elapsed - (scene.duration - transition)) / transition)) : 0;
  const nextScene = hasNext ? timings[sceneIndex + 1] : null;
  const nextLocalTime = nextScene && transitionProgress > 0
    ? transitionProgress * transition * (nextScene.sourceDuration / Math.max(1, nextScene.duration))
    : 0;
  return { sceneIndex, localTime, nextSceneIndex: hasNext ? sceneIndex + 1 : null, nextLocalTime, transitionProgress };
}

function cloneAnimation(animation: LayerAnimation, delay: number, duration: number): LayerAnimation {
  return { ...animation, delay, duration };
}

export function syncSceneToBeats(
  timeline: TimelineSettings,
  textLayers: TextLayer[],
  semanticLayerIds: string[],
  localBeatTimes: number[],
) {
  const fallbackGap = Math.max(180, Math.min(700, timeline.duration / Math.max(3, semanticLayerIds.length + textLayers.length + 1)));
  const usableBeats = localBeatTimes.filter((beat) => beat >= 0 && beat < timeline.duration);
  const beatAt = (index: number) => usableBeats[index] ?? Math.min(timeline.duration - 100, index * fallbackGap);
  const durationAt = (index: number) => {
    const interval = (usableBeats[index + 1] ?? beatAt(index) + fallbackGap) - beatAt(index);
    return Math.max(140, Math.min(700, interval * 0.72));
  };
  let index = 0;
  const baseAnimation = cloneAnimation(timeline.baseAnimation, beatAt(index), durationAt(index));
  index += 1;
  const sceneAnimations = { ...timeline.sceneAnimations };
  for (const id of [...semanticLayerIds].reverse()) {
    const animation = timeline.sceneAnimations[id];
    if (!animation) continue;
    sceneAnimations[id] = cloneAnimation(animation, beatAt(index), durationAt(index));
    index += 1;
  }
  const syncedTextLayers = textLayers.map((layer) => {
    if (!layer.animation.enabled) return layer;
    const animation = cloneAnimation(layer.animation, beatAt(index), durationAt(index));
    index += 1;
    return { ...layer, animation };
  });
  return { timeline: { ...timeline, baseAnimation, sceneAnimations }, textLayers: syncedTextLayers };
}

export function detectBeats(samples: Float32Array, sampleRate: number, sensitivity = 1) {
  if (!samples.length || !Number.isFinite(sampleRate) || sampleRate <= 0) return [];
  const frameSize = Math.max(256, Math.round(sampleRate * 0.02));
  const energies: number[] = [];
  for (let offset = 0; offset < samples.length; offset += frameSize) {
    let total = 0;
    const end = Math.min(samples.length, offset + frameSize);
    for (let index = offset; index < end; index += 1) total += samples[index] * samples[index];
    energies.push(Math.sqrt(total / Math.max(1, end - offset)));
  }
  const novelty = energies.map((energy, index) => {
    const start = Math.max(0, index - 18);
    let average = 0;
    for (let cursor = start; cursor < index; cursor += 1) average += energies[cursor];
    average /= Math.max(1, index - start);
    return average > 1e-5 ? Math.max(0, energy / average - 1) : 0;
  });
  const nonZero = novelty.filter((value) => value > 0).sort((a, b) => a - b);
  const median = nonZero[Math.floor(nonZero.length * 0.5)] ?? 0.15;
  const threshold = Math.max(0.1, Math.min(1.5, median * (1.5 / Math.max(0.5, Math.min(1.8, sensitivity)))));
  const minimumFrames = Math.max(1, Math.round(0.18 * sampleRate / frameSize));
  const beats: number[] = [];
  let lastFrame = -minimumFrames;
  for (let index = 1; index < novelty.length - 1; index += 1) {
    if (novelty[index] < threshold || novelty[index] < novelty[index - 1] || novelty[index] < novelty[index + 1]) continue;
    if (index - lastFrame < minimumFrames) {
      const previous = beats.length - 1;
      if (previous >= 0 && novelty[index] > novelty[lastFrame]) beats[previous] = index * frameSize / sampleRate * 1000;
      continue;
    }
    beats.push(index * frameSize / sampleRate * 1000);
    lastFrame = index;
  }
  return beats;
}

export function estimateTempo(beatTimes: number[]) {
  if (beatTimes.length < 2) return null;
  const intervals = beatTimes.slice(1).map((beat, index) => beat - beatTimes[index]).filter((interval) => interval >= 180 && interval <= 2000).sort((a, b) => a - b);
  if (!intervals.length) return null;
  let bpm = 60000 / intervals[Math.floor(intervals.length / 2)];
  while (bpm < 70) bpm *= 2;
  while (bpm > 180) bpm /= 2;
  return Math.round(bpm);
}

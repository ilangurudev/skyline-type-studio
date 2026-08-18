import assert from "node:assert/strict";
import test from "node:test";

import { buildReelTimeline, detectBeats, estimateTempo, reelDimensions, resolveReelFrame, selectSoundtrackBeats, syncSceneToBeats } from "../studio/reel.ts";
import { createTextLayer, createTimelineSettings } from "../studio/types.ts";

test("uses a 4-by-6 frame for the 2:3 reel format", () => {
  assert.deepEqual(reelDimensions("2:3"), { width: 1200, height: 1800 });
  assert.deepEqual(reelDimensions("2:3", 900), { width: 600, height: 900 });
});

test("detects a stable 120 BPM pulse train", () => {
  const sampleRate = 44100;
  const samples = new Float32Array(sampleRate * 6);
  for (let index = 0; index < samples.length; index += 1) samples[index] = Math.sin(index / sampleRate * Math.PI * 160) * 0.04;
  for (let beat = 0.5; beat < 6; beat += 0.5) {
    const start = Math.round(beat * sampleRate);
    const end = Math.min(samples.length, start + Math.round(sampleRate * 0.04));
    for (let index = start; index < end; index += 1) samples[index] += Math.sin(index / sampleRate * Math.PI * 1800) * 0.8;
  }
  const beats = detectBeats(samples, sampleRate);
  assert.ok(beats.length >= 10, `expected at least 10 beats, got ${beats.length}`);
  assert.equal(estimateTempo(beats), 120);
  assert.ok(Math.abs(beats[0] - 500) <= 25);
});

test("snaps scene boundaries to nearby beats and resolves crossfades", () => {
  const schedule = buildReelTimeline([
    { id: "one", duration: 3000 },
    { id: "two", duration: 3000 },
  ], [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], true);
  assert.equal(schedule.duration, 6000);
  assert.equal(schedule.scenes[1].start, 3000);
  const frame = resolveReelFrame(schedule.scenes, 2850, 300);
  assert.equal(frame.sceneIndex, 0);
  assert.equal(frame.nextSceneIndex, 1);
  assert.equal(frame.transitionProgress, 0.5);
});

test("incoming scene time stays continuous across a crossfade boundary", () => {
  const schedule = buildReelTimeline([
    { id: "one", duration: 3000 },
    { id: "two", duration: 3000 },
  ]).scenes;
  const beforeBoundary = resolveReelFrame(schedule, 2999, 320);
  const atBoundary = resolveReelFrame(schedule, 3000, 320);

  assert.equal(beforeBoundary.nextSceneIndex, 1);
  assert.ok(beforeBoundary.nextLocalTime > 300);
  assert.equal(atBoundary.sceneIndex, 1);
  const boundaryStep = atBoundary.localTime - beforeBoundary.nextLocalTime;
  assert.ok(
    boundaryStep >= 0 && boundaryStep <= 2,
    `incoming scene jumped from ${beforeBoundary.nextLocalTime}ms to ${atBoundary.localTime}ms`,
  );

  assert.equal(resolveReelFrame(schedule, 3000, 0).localTime, 0, "hard cuts should still start at zero");
});

test("rebases soundtrack beats to a selected song section", () => {
  assert.deepEqual(selectSoundtrackBeats([500, 1000, 1500, 2000, 2500], 1250), [250, 750, 1250]);
});

test("aligns base, depth, and text entrances to successive beats", () => {
  const timeline = createTimelineSettings(["far", "near"]);
  const text = createTextLayer("text-1", 1);
  const synced = syncSceneToBeats(timeline, [text], ["far", "near"], [0, 500, 1000, 1500]);
  assert.equal(synced.timeline.baseAnimation.delay, 0);
  assert.equal(synced.timeline.sceneAnimations.near.delay, 500);
  assert.equal(synced.timeline.sceneAnimations.far.delay, 1000);
  assert.equal(synced.textLayers[0].animation.delay, 1500);
});

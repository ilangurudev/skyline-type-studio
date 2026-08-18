import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";
import type { BinaryMask, CachedAnalysis, SemanticLayer, SkylineProjectV1 } from "./types";
import { validateRecipe } from "./types";

function bytesToBase64(input: ArrayLike<number>) {
  const bytes = Uint8Array.from(input);
  let result = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    result += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return btoa(result);
}

function base64ToBytes(value: string) {
  const binary = atob(value);
  const bytes = new Uint8ClampedArray(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

export function serializeMask(mask: BinaryMask) {
  return { width: mask.width, height: mask.height, data: bytesToBase64(mask.data) };
}

export function deserializeMask(mask: { width: number; height: number; data: string }): BinaryMask {
  const data = base64ToBytes(mask.data);
  if (data.length !== mask.width * mask.height) throw new Error("Cached mask dimensions do not match its data.");
  return { width: mask.width, height: mask.height, data };
}

export function serializeAnalysis(quality: "semantic" | "depth", skyMask: BinaryMask | null, layers: SemanticLayer[]): CachedAnalysis {
  return {
    quality,
    skyMask: skyMask ? serializeMask(skyMask) : null,
    layers: layers.map((layer) => ({ ...layer, data: bytesToBase64(layer.data) })),
  };
}

export function deserializeAnalysis(analysis: CachedAnalysis) {
  return {
    quality: analysis.quality,
    skyMask: analysis.skyMask ? deserializeMask(analysis.skyMask) : null,
    layers: analysis.layers.map((layer) => ({ ...layer, data: base64ToBytes(layer.data) })) satisfies SemanticLayer[],
  };
}

export function createProjectArchive(project: SkylineProjectV1) {
  validateProject(project);
  return zipSync({ "project.json": strToU8(JSON.stringify(project)) }, { level: 6 });
}

export function readProjectArchive(bytes: Uint8Array): SkylineProjectV1 {
  const files = unzipSync(bytes);
  const payload = files["project.json"];
  if (!payload) throw new Error("This project archive does not contain project.json.");
  const project = JSON.parse(strFromU8(payload)) as unknown;
  validateProject(project);
  return project;
}

export function validateProject(value: unknown): asserts value is SkylineProjectV1 {
  if (!value || typeof value !== "object") throw new Error("Project must be an object.");
  const project = value as Partial<SkylineProjectV1>;
  if (project.schemaVersion !== 1) throw new Error("Unsupported Skyline project version.");
  validateRecipe(project.recipe);
  if (!project.source || typeof project.source.sha256 !== "string" || !/^[0-9a-f]{64}$/i.test(project.source.sha256)) throw new Error("Project source fingerprint is invalid.");
  if (!project.analysis || !Array.isArray(project.analysis.layers) || !project.analysis.layers.length) throw new Error("Project does not contain cached depth layers.");
  deserializeAnalysis(project.analysis);
}

export async function sha256File(file: Blob) {
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

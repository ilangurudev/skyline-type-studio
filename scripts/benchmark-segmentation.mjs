import { mkdir, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import { pipeline } from "@huggingface/transformers";

const projectRoot = path.resolve(import.meta.dirname, "..");
const sourceDir = path.resolve(projectRoot, "../work/benchmark");
const outputDir = path.resolve(projectRoot, "../work/benchmark-review");
await mkdir(outputDir, { recursive: true });

const files = (await readdir(sourceDir))
  .filter((name) => /^\d\d-.*\.jpg$/i.test(name))
  .sort();

console.log("Loading SegFormer ADE20K sky model…");
const segmenter = await pipeline(
  "image-segmentation",
  "Xenova/segformer-b0-finetuned-ade-512-512",
  { dtype: "q8" },
);

const manifest = [];

function cleanSkyMask(input, width, height) {
  const sky = Uint8Array.from(input, (value) => (value > 127 ? 255 : 0));
  const visited = new Uint8Array(width * height);
  const queue = new Int32Array(width * height);
  const maxFloatingArea = Math.round(width * height * 0.035);

  for (let start = 0; start < sky.length; start += 1) {
    if (sky[start] || visited[start]) continue;
    let head = 0;
    let tail = 0;
    let minY = height;
    let maxY = 0;
    let touchesBottom = false;
    const component = [];
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
      const neighbors = [index - 1, index + 1, index - width, index + width];
      for (const next of neighbors) {
        if (next < 0 || next >= sky.length || visited[next] || sky[next]) continue;
        const nx = next % width;
        if (Math.abs(nx - x) > 1) continue;
        visited[next] = 1;
        queue[tail++] = next;
      }
    }

    const isFloatingArtifact =
      !touchesBottom &&
      component.length <= maxFloatingArea &&
      maxY - minY <= height * 0.34;
    if (isFloatingArtifact) {
      for (const index of component) sky[index] = 255;
    }
  }
  return sky;
}

for (const [index, file] of files.entries()) {
  const sourcePath = path.join(sourceDir, file);
  const stem = path.parse(file).name;
  console.log(`[${index + 1}/${files.length}] ${stem}`);
  const output = await segmenter(sourcePath);
  const labels = output.map((segment) => segment.label);
  const sky = output.find((segment) => segment.label?.toLowerCase() === "sky");
  if (!sky) {
    manifest.push({ file, labels, status: "missing-sky" });
    console.warn(`  No sky label. Labels: ${labels.join(", ")}`);
    continue;
  }

  const { width, height } = sky.mask;
  const maskPath = path.join(outputDir, `${stem}-sky-mask.png`);

  const skyData = Buffer.from(cleanSkyMask(sky.mask.data, width, height));
  await sharp(skyData, { raw: { width, height, channels: 1 } }).png().toFile(maskPath);
  const overlayData = Buffer.alloc(skyData.length * 4);
  const skyAlphaData = Buffer.alloc(skyData.length * 4);
  for (let i = 0; i < skyData.length; i += 1) {
    const isSky = skyData[i] > 127;
    const offset = i * 4;
    overlayData[offset] = isSky ? 73 : 217;
    overlayData[offset + 1] = isSky ? 175 : 255;
    overlayData[offset + 2] = isSky ? 255 : 72;
    overlayData[offset + 3] = isSky ? 82 : 54;
    skyAlphaData[offset] = 255;
    skyAlphaData[offset + 1] = 255;
    skyAlphaData[offset + 2] = 255;
    skyAlphaData[offset + 3] = isSky ? 255 : 0;
  }

  const overlayPath = path.join(outputDir, `${stem}-overlay.jpg`);
  await sharp(sourcePath)
    .composite([{ input: overlayData, raw: { width, height, channels: 4 } }])
    .jpeg({ quality: 90 })
    .toFile(overlayPath);

  const metadata = await sharp(sourcePath).metadata();
  const imageWidth = metadata.width ?? width;
  const imageHeight = metadata.height ?? height;
  const transitions = [];
  for (let x = 0; x < width; x += 1) {
    for (let y = Math.round(height * 0.08); y < height; y += 1) {
      if (!skyData[y * width + x]) {
        transitions.push(y);
        break;
      }
    }
  }
  transitions.sort((a, b) => a - b);
  const boundaryY = transitions[Math.floor(transitions.length / 2)] ?? Math.round(height * 0.52);
  const fontSize = Math.max(42, Math.round(imageHeight * 0.12));
  const titleY = Math.round((boundaryY / height) * imageHeight + fontSize * 0.12);
  const titleSvg = Buffer.from(`
    <svg width="${imageWidth}" height="${imageHeight}" xmlns="http://www.w3.org/2000/svg">
      <text x="50%" y="${titleY}" text-anchor="middle" dominant-baseline="middle"
        font-family="Impact, Arial Black, sans-serif" font-size="${fontSize}"
        font-weight="900" fill="#f6edd7" stroke="#33251f" stroke-width="3">BEHIND THE SKYLINE</text>
    </svg>
  `);
  const maskedTitle = await sharp({
    create: { width: imageWidth, height: imageHeight, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } },
  })
    .composite([{ input: titleSvg }, { input: skyAlphaData, raw: { width, height, channels: 4 }, blend: "dest-in" }])
    .png()
    .toBuffer();
  const posterPath = path.join(outputDir, `${stem}-poster.jpg`);
  await sharp(sourcePath)
    .composite([{ input: maskedTitle }])
    .jpeg({ quality: 91 })
    .toFile(posterPath);

  manifest.push({ file, labels, status: "ok", width, height, maskPath, overlayPath, posterPath });
}

await writeFile(path.join(outputDir, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log(`Wrote ${manifest.length} benchmark results to ${outputDir}`);
await segmenter.dispose();

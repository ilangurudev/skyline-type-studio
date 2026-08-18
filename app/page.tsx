"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";

type TextAlign = "left" | "center" | "right";
type SkyMask = { width: number; height: number; data: Uint8ClampedArray };
type MaskStatus = "idle" | "loading-model" | "analyzing" | "ready" | "error";
type Segment = { label?: string; mask: { width: number; height: number; data: ArrayLike<number> } };
type Segmenter = (input: HTMLCanvasElement) => Promise<Segment[]>;

const MODEL_ID = "Xenova/segformer-b0-finetuned-ade-512-512";
let segmenterPromise: Promise<Segmenter> | null = null;

const FONT_OPTIONS = [
  ["Impact", "Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif"],
  ["Arial Black", "'Arial Black', Arial, sans-serif"],
  ["Helvetica", "Helvetica, Arial, sans-serif"],
  ["Georgia", "Georgia, serif"],
  ["Times", "'Times New Roman', Times, serif"],
  ["Courier", "'Courier New', monospace"],
] as const;

async function getSegmenter() {
  if (!segmenterPromise) {
    segmenterPromise = import("@huggingface/transformers").then(async ({ pipeline }) => {
      const model = await pipeline("image-segmentation", MODEL_ID, { dtype: "q8", device: "wasm" });
      return model as unknown as Segmenter;
    });
  }
  return segmenterPromise;
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

function createMaskCanvas(mask: SkyMask, foreground: boolean) {
  const canvas = document.createElement("canvas");
  canvas.width = mask.width;
  canvas.height = mask.height;
  const ctx = canvas.getContext("2d")!;
  const pixels = ctx.createImageData(mask.width, mask.height);
  for (let index = 0; index < mask.data.length; index += 1) {
    const offset = index * 4;
    pixels.data[offset] = pixels.data[offset + 1] = pixels.data[offset + 2] = 255;
    pixels.data[offset + 3] = foreground ? 255 - mask.data[index] : mask.data[index];
  }
  ctx.putImageData(pixels, 0, 0);
  return canvas;
}

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const analysisSequence = useRef(0);
  const [imageName, setImageName] = useState("");
  const [dimensions, setDimensions] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [maskStatus, setMaskStatus] = useState<MaskStatus>("idle");
  const [maskError, setMaskError] = useState("");
  const [skyMask, setSkyMask] = useState<SkyMask | null>(null);
  const [text, setText] = useState("ONE DESERT\nAFTER\nANOTHER");
  const [font, setFont] = useState(FONT_OPTIONS[0][1]);
  const [fontSize, setFontSize] = useState(14);
  const [lineGap, setLineGap] = useState(18);
  const [xPosition, setXPosition] = useState(50);
  const [yPosition, setYPosition] = useState(38);
  const [alignment, setAlignment] = useState<TextAlign>("center");
  const [textColor, setTextColor] = useState("#f6edd7");
  const [shadowColor, setShadowColor] = useState("#3a2a22");
  const [shadow, setShadow] = useState(true);
  const [behindLand, setBehindLand] = useState(true);
  const [showMask, setShowMask] = useState(false);

  const drawPoster = useCallback((target: HTMLCanvasElement, maskOverlay = false) => {
    const image = imageRef.current;
    if (!image) return;
    const scale = Math.min(1, 7000 / Math.max(image.naturalWidth, image.naturalHeight));
    const width = Math.round(image.naturalWidth * scale);
    const height = Math.round(image.naturalHeight * scale);
    target.width = width;
    target.height = height;
    const ctx = target.getContext("2d")!;
    ctx.drawImage(image, 0, 0, width, height);

    const typeLayer = document.createElement("canvas");
    typeLayer.width = width;
    typeLayer.height = height;
    const typeCtx = typeLayer.getContext("2d")!;
    const sizePx = (fontSize / 100) * height;
    const lines = text.split("\n");
    const lineHeight = sizePx * (1 + lineGap / 100);
    const totalHeight = Math.max(sizePx, (lines.length - 1) * lineHeight + sizePx);
    const startBaseline = (yPosition / 100) * height - totalHeight / 2 + sizePx * 0.82;
    const x = (xPosition / 100) * width;
    typeCtx.font = `900 ${sizePx}px ${font}`;
    typeCtx.textAlign = alignment;
    typeCtx.textBaseline = "alphabetic";
    typeCtx.lineJoin = "round";
    lines.forEach((line, index) => {
      const baseline = startBaseline + index * lineHeight;
      if (shadow) {
        typeCtx.fillStyle = shadowColor;
        const offset = Math.max(3, sizePx * 0.045);
        typeCtx.fillText(line, x + offset, baseline + offset);
      }
      typeCtx.fillStyle = textColor;
      typeCtx.fillText(line, x, baseline);
    });
    ctx.drawImage(typeLayer, 0, 0);

    if (behindLand && skyMask) {
      const foreground = document.createElement("canvas");
      foreground.width = width;
      foreground.height = height;
      const foregroundCtx = foreground.getContext("2d")!;
      foregroundCtx.drawImage(image, 0, 0, width, height);
      foregroundCtx.globalCompositeOperation = "destination-in";
      foregroundCtx.imageSmoothingEnabled = true;
      foregroundCtx.drawImage(createMaskCanvas(skyMask, true), 0, 0, width, height);
      ctx.drawImage(foreground, 0, 0);
    }

    if (maskOverlay && skyMask) {
      const overlay = document.createElement("canvas");
      overlay.width = skyMask.width;
      overlay.height = skyMask.height;
      const overlayCtx = overlay.getContext("2d")!;
      const pixels = overlayCtx.createImageData(skyMask.width, skyMask.height);
      for (let index = 0; index < skyMask.data.length; index += 1) {
        const isSky = skyMask.data[index] > 127;
        const offset = index * 4;
        pixels.data[offset] = isSky ? 73 : 217;
        pixels.data[offset + 1] = isSky ? 175 : 255;
        pixels.data[offset + 2] = isSky ? 255 : 72;
        pixels.data[offset + 3] = isSky ? 82 : 54;
      }
      overlayCtx.putImageData(pixels, 0, 0);
      ctx.drawImage(overlay, 0, 0, width, height);
    }
  }, [alignment, behindLand, font, fontSize, lineGap, shadow, shadowColor, skyMask, text, textColor, xPosition, yPosition]);

  const analyzeImage = useCallback(async (image: HTMLImageElement) => {
    const sequence = ++analysisSequence.current;
    setSkyMask(null);
    setMaskError("");
    setMaskStatus("loading-model");
    try {
      const segmenter = await getSegmenter();
      if (sequence !== analysisSequence.current) return;
      setMaskStatus("analyzing");
      const scale = Math.min(1, 1024 / Math.max(image.naturalWidth, image.naturalHeight));
      const analysis = document.createElement("canvas");
      analysis.width = Math.max(1, Math.round(image.naturalWidth * scale));
      analysis.height = Math.max(1, Math.round(image.naturalHeight * scale));
      analysis.getContext("2d")!.drawImage(image, 0, 0, analysis.width, analysis.height);
      const segments = await segmenter(analysis);
      if (sequence !== analysisSequence.current) return;
      const sky = segments.find((segment) => segment.label?.toLowerCase() === "sky");
      if (!sky) throw new Error("No sky was detected in this photograph.");
      setSkyMask({ width: sky.mask.width, height: sky.mask.height, data: cleanSkyMask(sky.mask.data, sky.mask.width, sky.mask.height) });
      setMaskStatus("ready");
    } catch (error) {
      console.error(error);
      setMaskError(error instanceof Error ? error.message : "The sky model could not run in this browser.");
      setMaskStatus("error");
    }
  }, []);

  useEffect(() => { if (canvasRef.current) drawPoster(canvasRef.current, showMask); }, [drawPoster, showMask]);

  const loadFile = (file?: File) => {
    if (!file || !file.type.startsWith("image/")) return;
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      setImageName(file.name);
      setDimensions(`${image.naturalWidth.toLocaleString()} × ${image.naturalHeight.toLocaleString()}`);
      URL.revokeObjectURL(objectUrl);
      void analyzeImage(image);
    };
    image.onerror = () => { URL.revokeObjectURL(objectUrl); setMaskError("That image could not be opened."); setMaskStatus("error"); };
    image.src = objectUrl;
  };
  const handleFile = (event: ChangeEvent<HTMLInputElement>) => loadFile(event.target.files?.[0]);
  const handleDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setIsDragging(false); loadFile(event.dataTransfer.files?.[0]); };
  const downloadPoster = () => {
    if (!imageRef.current) return;
    const exportCanvas = document.createElement("canvas");
    drawPoster(exportCanvas, false);
    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${imageName.replace(/\.[^.]+$/, "") || "poster"}-skyline-poster.png`;
      anchor.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  };

  const busy = maskStatus === "loading-model" || maskStatus === "analyzing";
  const statusText = maskStatus === "loading-model" ? "Loading precision model…" : maskStatus === "analyzing" ? "Separating sky…" : maskStatus === "ready" ? "Semantic mask ready" : maskStatus === "error" ? "Mask unavailable" : "Waiting for image";

  return <main className="studio-shell">
    <header className="masthead"><div><p className="eyebrow">Browser-based poster maker</p><h1>Skyline Type Studio</h1></div><p className="privacy-note">Your photograph stays in this browser.</p></header>
    <section className="workspace">
      <aside className="control-panel" aria-label="Poster controls">
        <section className="control-section">
          <div className="panel-heading"><span className="step">01</span><div><p className="label">Source image</p><p className="hint">Mountains, coasts, cities, and trees all work.</p></div></div>
          <button className="upload-button" onClick={() => fileInput.current?.click()}>{imageName ? "Replace photograph" : "Choose a photograph"}</button>
          <input ref={fileInput} type="file" accept="image/*" hidden onChange={handleFile} />
          {imageName && <p className="file-meta"><span>{imageName}</span><span>{dimensions}</span></p>}
        </section>
        <section className="control-section">
          <div className="panel-heading"><span className="step">02</span><div><p className="label">Poster text</p><p className="hint">Line breaks are preserved.</p></div></div>
          <textarea value={text} onChange={(event) => setText(event.target.value)} rows={4} aria-label="Poster text" />
          <div className="field-grid">
            <label><span>Typeface</span><select value={font} onChange={(event) => setFont(event.target.value)}>{FONT_OPTIONS.map(([name, value]) => <option value={value} key={name}>{name}</option>)}</select></label>
            <label><span>Alignment</span><select value={alignment} onChange={(event) => setAlignment(event.target.value as TextAlign)}><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option></select></label>
          </div>
          <Range label="Font size" value={fontSize} min={4} max={30} suffix="%" onChange={setFontSize} />
          <Range label="Line spacing" value={lineGap} min={-20} max={80} suffix="%" onChange={setLineGap} />
          <Range label="Horizontal position" value={xPosition} min={0} max={100} suffix="%" onChange={setXPosition} />
          <Range label="Vertical position" value={yPosition} min={0} max={100} suffix="%" onChange={setYPosition} />
          <div className="color-grid">
            <label><span>Type</span><input type="color" value={textColor} onChange={(event) => setTextColor(event.target.value)} /></label>
            <label><span>Shadow</span><input type="color" value={shadowColor} onChange={(event) => setShadowColor(event.target.value)} /></label>
          </div>
          <Toggle checked={shadow} onChange={setShadow} label="Hard offset shadow" />
        </section>
        <section className="control-section mask-section">
          <div className="panel-heading"><span className="step">03</span><div><p className="label">Sky mask</p><p className="hint">A semantic model traces detailed silhouettes automatically.</p></div></div>
          <div className={`mask-readout ${maskStatus}`}><span className="status-dot" />{statusText}</div>
          {maskError && <p className="mask-error">{maskError}</p>}
          <Toggle checked={behindLand} onChange={setBehindLand} label="Place text behind foreground" />
          <Toggle checked={showMask} onChange={setShowMask} label="Show sky / foreground overlay" />
        </section>
        <button className="download-button" disabled={!imageName || busy || (behindLand && !skyMask)} onClick={downloadPoster}>{busy ? statusText : "Download full-resolution PNG"}</button>
      </aside>
      <section className="preview-panel" aria-label="Poster preview">
        <div className="preview-topline"><div><span className={`status-dot ${skyMask ? "ready" : ""}`} />Live canvas</div><span>{imageName ? statusText : "Waiting for image"}</span></div>
        {!imageName ? <div className={`drop-zone ${isDragging ? "dragging" : ""}`} onClick={() => fileInput.current?.click()} onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInput.current?.click(); }}><span className="drop-mark">＋</span><strong>Drop a photograph here</strong><small>or click to browse · JPEG, PNG, WebP</small></div> : <div className="canvas-stage"><canvas ref={canvasRef} aria-label="Live poster preview" />{busy && <div className="calculating-chip">{statusText}</div>}</div>}
        <footer className="preview-footer"><span><i className="key-swatch sky" />Sky</span><span><i className="key-swatch land" />Foreground</span><span className="footer-tip">Blue stays behind the type; lime is redrawn in front for the cut-through effect.</span></footer>
      </section>
    </section>
  </main>;
}

function Range({ label, value, min, max, suffix, onChange }: { label: string; value: number; min: number; max: number; suffix: string; onChange: (value: number) => void }) {
  const id = `range-${label.toLowerCase().replaceAll(" ", "-")}`;
  return <div className="range-field"><span><label htmlFor={id}><b>{label}</b></label><output htmlFor={id}>{value}{suffix}</output></span><input id={id} type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} /></div>;
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (value: boolean) => void; label: string }) {
  return <label className="toggle-row"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span className="toggle-track"><i /></span><b>{label}</b></label>;
}

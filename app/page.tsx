"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";

type TextAlign = "left" | "center" | "right";
type Ridge = { width: number; height: number; points: number[] };

const FONT_OPTIONS = [
  ["Impact", "Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif"],
  ["Arial Black", "'Arial Black', Arial, sans-serif"],
  ["Helvetica", "Helvetica, Arial, sans-serif"],
  ["Georgia", "Georgia, serif"],
  ["Times", "'Times New Roman', Times, serif"],
  ["Courier", "'Courier New', monospace"],
] as const;

function colorDistance(a: number[], b: number[]) {
  const dr = a[0] - b[0];
  const dg = a[1] - b[1];
  const db = a[2] - b[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function calculateRidge(image: HTMLImageElement, guidePercent: number): Ridge {
  const width = Math.min(720, image.naturalWidth);
  const height = Math.max(1, Math.round((image.naturalHeight / image.naturalWidth) * width));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(image, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const sample = (x: number, y: number) => {
    const i = (Math.max(0, Math.min(height - 1, y)) * width + Math.max(0, Math.min(width - 1, x))) * 4;
    return [pixels[i], pixels[i + 1], pixels[i + 2]];
  };

  const guide = (guidePercent / 100) * height;
  const radius = Math.max(16, Math.round(height * 0.22));
  const minY = Math.max(6, Math.round(guide - radius));
  const maxY = Math.min(height - 7, Math.round(guide + radius));
  const raw = new Array<number>(width);

  for (let x = 0; x < width; x += 1) {
    let bestY = Math.round(guide);
    let bestScore = -Infinity;
    for (let y = minY; y <= maxY; y += 1) {
      const above = [0, 0, 0];
      const below = [0, 0, 0];
      let count = 0;
      for (let dx = -2; dx <= 2; dx += 1) {
        for (let band = 3; band <= 6; band += 1) {
          const a = sample(x + dx, y - band);
          const b = sample(x + dx, y + band);
          above[0] += a[0]; above[1] += a[1]; above[2] += a[2];
          below[0] += b[0]; below[1] += b[1]; below[2] += b[2];
          count += 1;
        }
      }
      above[0] /= count; above[1] /= count; above[2] /= count;
      below[0] /= count; below[1] /= count; below[2] /= count;

      const edge = colorDistance(above, below);
      const distancePenalty = Math.abs(y - guide) * 0.035;
      const upperPreference = ((maxY - y) / Math.max(1, maxY - minY)) * 1.3;
      const score = edge - distancePenalty + upperPreference;
      if (score > bestScore) {
        bestScore = score;
        bestY = y;
      }
    }
    raw[x] = bestY;
  }

  // Two small median passes suppress texture spikes while retaining peaks.
  let points = raw;
  for (const halfWindow of [5, 3]) {
    points = points.map((_, x) => {
      const window: number[] = [];
      for (let i = Math.max(0, x - halfWindow); i <= Math.min(width - 1, x + halfWindow); i += 1) {
        window.push(points[i]);
      }
      return median(window);
    });
  }

  return { width, height, points };
}

function ridgePath(ctx: CanvasRenderingContext2D, ridge: Ridge, width: number, height: number) {
  const sx = width / ridge.width;
  const sy = height / ridge.height;
  ctx.beginPath();
  ctx.moveTo(0, height);
  for (let x = 0; x < ridge.width; x += 1) {
    ctx.lineTo(x * sx, ridge.points[x] * sy);
  }
  ctx.lineTo(width, height);
  ctx.closePath();
}

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [imageName, setImageName] = useState("");
  const [dimensions, setDimensions] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isCalculating, setIsCalculating] = useState(false);
  const [ridge, setRidge] = useState<Ridge | null>(null);
  const [text, setText] = useState("ONE DESERT\nAFTER\nANOTHER");
  const [font, setFont] = useState(FONT_OPTIONS[0][1]);
  const [fontSize, setFontSize] = useState(14);
  const [lineGap, setLineGap] = useState(18);
  const [xPosition, setXPosition] = useState(50);
  const [yPosition, setYPosition] = useState(38);
  const [alignment, setAlignment] = useState<TextAlign>("center");
  const [guide, setGuide] = useState(55);
  const [textColor, setTextColor] = useState("#f6edd7");
  const [shadowColor, setShadowColor] = useState("#3a2a22");
  const [shadow, setShadow] = useState(true);
  const [behindLand, setBehindLand] = useState(true);
  const [showMask, setShowMask] = useState(false);

  const drawPoster = useCallback((target: HTMLCanvasElement, maskOverlay = false) => {
    const image = imageRef.current;
    if (!image) return;
    const maxSide = 7000;
    const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
    const width = Math.round(image.naturalWidth * scale);
    const height = Math.round(image.naturalHeight * scale);
    target.width = width;
    target.height = height;
    const ctx = target.getContext("2d")!;
    ctx.clearRect(0, 0, width, height);
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
    for (let index = 0; index < lines.length; index += 1) {
      const baseline = startBaseline + index * lineHeight;
      if (shadow) {
        typeCtx.fillStyle = shadowColor;
        const offset = Math.max(3, sizePx * 0.045);
        typeCtx.fillText(lines[index], x + offset, baseline + offset);
      }
      typeCtx.fillStyle = textColor;
      typeCtx.fillText(lines[index], x, baseline);
    }
    ctx.drawImage(typeLayer, 0, 0);

    if (behindLand && ridge) {
      ctx.save();
      ridgePath(ctx, ridge, width, height);
      ctx.clip();
      ctx.drawImage(image, 0, 0, width, height);
      ctx.restore();
    }

    if (maskOverlay && ridge) {
      ctx.save();
      ridgePath(ctx, ridge, width, height);
      ctx.fillStyle = "rgba(217, 255, 72, .30)";
      ctx.fill();
      ctx.restore();
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(width, 0);
      for (let xIndex = ridge.width - 1; xIndex >= 0; xIndex -= 1) {
        ctx.lineTo((xIndex / ridge.width) * width, (ridge.points[xIndex] / ridge.height) * height);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(73, 175, 255, .24)";
      ctx.fill();
      ctx.restore();
    }
  }, [alignment, behindLand, font, fontSize, lineGap, ridge, shadow, shadowColor, text, textColor, xPosition, yPosition]);

  useEffect(() => {
    if (!imageRef.current) return;
    setIsCalculating(true);
    const id = window.setTimeout(() => {
      setRidge(calculateRidge(imageRef.current!, guide));
      setIsCalculating(false);
    }, 60);
    return () => window.clearTimeout(id);
  }, [guide, imageName]);

  useEffect(() => {
    if (canvasRef.current) drawPoster(canvasRef.current, showMask);
  }, [drawPoster, showMask]);

  const loadFile = (file?: File) => {
    if (!file || !file.type.startsWith("image/")) return;
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      setImageName(file.name);
      setDimensions(`${image.naturalWidth.toLocaleString()} × ${image.naturalHeight.toLocaleString()}`);
      URL.revokeObjectURL(objectUrl);
    };
    image.src = objectUrl;
  };

  const handleFile = (event: ChangeEvent<HTMLInputElement>) => loadFile(event.target.files?.[0]);
  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    loadFile(event.dataTransfer.files?.[0]);
  };

  const downloadPoster = () => {
    if (!imageRef.current) return;
    const exportCanvas = document.createElement("canvas");
    drawPoster(exportCanvas, false);
    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      const baseName = imageName.replace(/\.[^.]+$/, "") || "poster";
      anchor.href = url;
      anchor.download = `${baseName}-skyline-poster.png`;
      anchor.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  };

  return (
    <main className="studio-shell">
      <header className="masthead">
        <div>
          <p className="eyebrow">Browser-based poster maker</p>
          <h1>Skyline Type Studio</h1>
        </div>
        <p className="privacy-note">Your image stays on this device.</p>
      </header>

      <section className="workspace">
        <aside className="control-panel" aria-label="Poster controls">
          <section className="control-section">
            <div className="panel-heading">
              <span className="step">01</span>
              <div><p className="label">Source image</p><p className="hint">Landscape photos work best.</p></div>
            </div>
            <button className="upload-button" onClick={() => fileInput.current?.click()}>
              {imageName ? "Replace photograph" : "Choose a photograph"}
            </button>
            <input ref={fileInput} type="file" accept="image/*" hidden onChange={handleFile} />
            {imageName && <p className="file-meta"><span>{imageName}</span><span>{dimensions}</span></p>}
          </section>

          <section className="control-section">
            <div className="panel-heading">
              <span className="step">02</span>
              <div><p className="label">Poster text</p><p className="hint">Line breaks are preserved.</p></div>
            </div>
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
            <div className="panel-heading">
              <span className="step">03</span>
              <div><p className="label">Sky mask</p><p className="hint">Guide the detector near the horizon.</p></div>
            </div>
            <Range label="Horizon guide" value={guide} min={20} max={90} suffix="%" onChange={setGuide} />
            <Toggle checked={behindLand} onChange={setBehindLand} label="Place text behind foreground" />
            <Toggle checked={showMask} onChange={setShowMask} label="Show sky / land overlay" />
          </section>

          <button className="download-button" disabled={!imageName || isCalculating} onClick={downloadPoster}>
            {isCalculating ? "Calculating mask…" : "Download full-resolution PNG"}
          </button>
        </aside>

        <section className="preview-panel" aria-label="Poster preview">
          <div className="preview-topline">
            <div><span className={`status-dot ${imageName ? "ready" : ""}`} />Live canvas</div>
            <span>{isCalculating ? "Tracing skyline…" : imageName ? dimensions : "Waiting for image"}</span>
          </div>
          {!imageName ? (
            <div
              className={`drop-zone ${isDragging ? "dragging" : ""}`}
              onClick={() => fileInput.current?.click()}
              onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInput.current?.click(); }}
            >
              <span className="drop-mark">＋</span>
              <strong>Drop a photograph here</strong>
              <small>or click to browse · JPEG, PNG, WebP</small>
            </div>
          ) : (
            <div className="canvas-stage">
              <canvas ref={canvasRef} aria-label="Live poster preview" />
              {isCalculating && <div className="calculating-chip">Tracing the ridge…</div>}
            </div>
          )}
          <footer className="preview-footer">
            <span><i className="key-swatch sky" />Sky</span>
            <span><i className="key-swatch land" />Foreground</span>
            <span className="footer-tip">Adjust “Horizon guide” if the trace catches clouds or ground texture.</span>
          </footer>
        </section>
      </section>
    </main>
  );
}

function Range({ label, value, min, max, suffix, onChange }: { label: string; value: number; min: number; max: number; suffix: string; onChange: (value: number) => void }) {
  const id = `range-${label.toLowerCase().replaceAll(" ", "-")}`;
  return (
    <div className="range-field">
      <span><label htmlFor={id}><b>{label}</b></label><output htmlFor={id}>{value}{suffix}</output></span>
      <input id={id} type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </div>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (value: boolean) => void; label: string }) {
  return (
    <label className="toggle-row">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-track"><i /></span>
      <b>{label}</b>
    </label>
  );
}

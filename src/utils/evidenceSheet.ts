import type {
  AnalysisMode,
  AnalysisResult,
  EvidenceEntry,
  ExifMetadata,
  RoiRect,
  VlmResult,
} from '../engine/types';
import { haversineDistanceKm } from '../engine/exif';

function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
  maxLines: number,
): number {
  const words = text.replace(/\s+/g, ' ').trim().split(' ');
  let line = '';
  let curY = y;
  let count = 0;

  for (let i = 0; i < words.length; i++) {
    const test = line ? `${line} ${words[i]}` : words[i];
    if (ctx.measureText(test).width > maxWidth && line) {
      count++;
      if (count >= maxLines) {
        ctx.fillText(`${line}…`, x, curY);
        return curY + lineHeight;
      }
      ctx.fillText(line, x, curY);
      line = words[i];
      curY += lineHeight;
    } else {
      line = test;
    }
  }
  if (line) {
    ctx.fillText(line, x, curY);
    curY += lineHeight;
  }
  return curY;
}

async function svgToImage(svgEl: SVGSVGElement): Promise<HTMLImageElement | null> {
  try {
    const clone = svgEl.cloneNode(true) as SVGSVGElement;
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    // Inline computed CSS vars used by WorldMap
    let raw = new XMLSerializer().serializeToString(clone);
    const varMap: Record<string, string> = {
      'var(--border-subtle)': '#e5e3dc',
      'var(--map-land)': '#e7e5dd',
      'var(--map-border)': '#cfccc2',
      'var(--accent)': '#0f766e',
      'var(--amber)': '#d97706',
      'var(--bg-surface)': '#ffffff',
      'var(--text-primary)': '#18181b',
      'var(--success)': '#15803d',
    };
    for (const [k, v] of Object.entries(varMap)) {
      raw = raw.split(k).join(v);
    }
    const blob = new Blob([raw], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = await new Promise<HTMLImageElement | null>((resolve) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => resolve(null);
      el.src = url;
    });
    URL.revokeObjectURL(url);
    return img;
  } catch {
    return null;
  }
}

export async function downloadEvidenceSheetPng(params: {
  imageBitmap: ImageBitmap;
  imageName: string;
  mode: AnalysisMode;
  roi: RoiRect;
  mapSvgEl: SVGSVGElement | null;
  result: AnalysisResult;
  exif: ExifMetadata | null;
  gpsRevealed: boolean;
  vlmResult: VlmResult | null;
  evidenceEntries: EvidenceEntry[];
}): Promise<void> {
  const W = 1600;
  const H = 1040;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Day-mode warm background
  ctx.fillStyle = '#f6f5f0';
  ctx.fillRect(0, 0, W, H);

  // Header bar
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, 68);
  ctx.strokeStyle = '#e5e3dc';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, 68.5);
  ctx.lineTo(W, 68.5);
  ctx.stroke();

  ctx.fillStyle = '#0f766e';
  ctx.font = 'bold 22px system-ui, -apple-system, sans-serif';
  ctx.fillText("Investigations Lab's AutoBolt — Evidence Sheet", 32, 42);

  ctx.fillStyle = '#64748b';
  ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, monospace';
  const metaHeader = `${params.imageName || 'Photo'} · ${params.result.backend.toUpperCase()} · ${params.result.latencyMs.toFixed(0)} ms`;
  ctx.fillText(metaHeader, W - 32 - ctx.measureText(metaHeader).width, 42);

  // Left card: Evidence Photo
  const cardPad = 24;
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#e5e3dc';
  ctx.fillRect(32, 96, 740, 560);
  ctx.strokeRect(32, 96, 740, 560);

  const imgBoxW = 740 - cardPad * 2;
  const imgBoxH = 490;
  const scale = Math.min(imgBoxW / params.imageBitmap.width, imgBoxH / params.imageBitmap.height);
  const drawW = params.imageBitmap.width * scale;
  const drawH = params.imageBitmap.height * scale;
  const drawX = 32 + cardPad + (imgBoxW - drawW) / 2;
  const drawY = 96 + cardPad + (imgBoxH - drawH) / 2;
  ctx.drawImage(params.imageBitmap, drawX, drawY, drawW, drawH);

  if (params.mode === 'roi') {
    ctx.strokeStyle = '#d97706';
    ctx.lineWidth = 3;
    ctx.strokeRect(
      drawX + params.roi.x * drawW,
      drawY + params.roi.y * drawH,
      params.roi.w * drawW,
      params.roi.h * drawH,
    );
  }

  ctx.fillStyle = '#475569';
  ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText(
    `Input: ${params.imageBitmap.width}×${params.imageBitmap.height} · Crop Mode: ${params.mode.toUpperCase()}`,
    32 + cardPad,
    638,
  );

  // Right top card: Map Snapshot
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(800, 96, 768, 420);
  ctx.strokeRect(800, 96, 768, 420);
  if (params.mapSvgEl) {
    const mapImg = await svgToImage(params.mapSvgEl);
    if (mapImg) {
      ctx.drawImage(mapImg, 812, 108, 744, 396);
    }
  }

  // Right bottom card: Leads & Confidence Summary
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(800, 536, 768, 472);
  ctx.strokeRect(800, 536, 768, 472);

  const topC = params.result.countries[0];
  const topR = params.result.regionsHierarchical[0];
  const topCity = params.result.citiesHierarchical[0];

  ctx.fillStyle = '#0f766e';
  ctx.font = 'bold 13px system-ui, sans-serif';
  ctx.fillText('PRIMARY GEOLOCATION LEAD & BAYESIAN EVIDENCE', 824, 568);

  ctx.fillStyle = '#0f172a';
  ctx.font = 'bold 26px system-ui, sans-serif';
  ctx.fillText(
    topCity ? `${topCity.name}, ${topCity.region}` : topR?.name ?? topC?.name ?? 'Unknown',
    824,
    604,
  );

  ctx.fillStyle = '#334155';
  ctx.font = '16px system-ui, sans-serif';
  const lvl = params.result.confidence.levels;
  ctx.fillText(
    `${topC?.name ?? '—'} (${((topC?.prob ?? 0) * 100).toFixed(1)}%) · Country: ${lvl.country} | Region: ${lvl.region} | City: ${lvl.city}`,
    824,
    632,
  );

  let cy = 664;
  if (params.result.geocell) {
    ctx.fillStyle = '#0f766e';
    ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(
      `Coords (${params.result.geocell.source}): ${params.result.geocell.lat.toFixed(4)}, ${params.result.geocell.lng.toFixed(4)} (±${Math.round(params.result.geocell.spreadKm)} km)`,
      824,
      cy,
    );
    cy += 26;
  }

  if (params.gpsRevealed && params.exif?.gps && topCity) {
    const errKm = haversineDistanceKm(
      params.exif.gps.lat,
      params.exif.gps.lng,
      topCity.lat,
      topCity.lng,
    );
    ctx.fillStyle = '#15803d';
    ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(
      `EXIF Ground Truth: ${params.exif.gps.lat.toFixed(5)}, ${params.exif.gps.lng.toFixed(5)} (${errKm.toFixed(1)} km to #1)`,
      824,
      cy,
    );
    cy += 26;
  }

  ctx.fillStyle = '#1e293b';
  ctx.font = '14px system-ui, sans-serif';
  cy = wrapText(ctx, params.result.confidence.brief, 824, cy + 6, 720, 21, 5);

  const activeEv = params.evidenceEntries.filter((e) => e.active);
  if (activeEv.length > 0) {
    ctx.fillStyle = '#0f766e';
    ctx.font = 'bold 13px system-ui, sans-serif';
    ctx.fillText(`BAYESIAN CLUE BOARD (${activeEv.length} ACTIVE)`, 824, cy + 16);
    cy += 38;
    ctx.fillStyle = '#334155';
    ctx.font = '13px system-ui, sans-serif';
    for (const ev of activeEv.slice(0, 4)) {
      ctx.fillText(`[${ev.status.toUpperCase()}] ${ev.label}`, 824, cy);
      cy += 20;
    }
  }

  // Left bottom card: Qwen-VL Summary or Top Visual Axes
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(32, 676, 740, 332);
  ctx.strokeRect(32, 676, 740, 332);

  if (params.vlmResult?.text) {
    ctx.fillStyle = '#0f766e';
    ctx.font = 'bold 13px system-ui, sans-serif';
    ctx.fillText(
      `DEEP VLM CLUE CHECK — ${params.vlmResult.modelLabel.toUpperCase()} (${(params.vlmResult.latencyMs / 1000).toFixed(1)}s)`,
      56,
      706,
    );
    ctx.fillStyle = '#1e293b';
    ctx.font = '13px system-ui, sans-serif';
    wrapText(ctx, params.vlmResult.text, 56, 734, 692, 20, 12);
  } else {
    ctx.fillStyle = '#0f766e';
    ctx.font = 'bold 13px system-ui, sans-serif';
    ctx.fillText('TOP REGIONAL, CITY & VISUAL CUES', 56, 706);
    ctx.fillStyle = '#1e293b';
    ctx.font = '14px system-ui, sans-serif';
    let ly = 736;
    for (const c of params.result.countries.slice(0, 4)) {
      ctx.fillText(
        `${c.rank}. ${c.name} — ${(c.prob * 100).toFixed(1)}% · Drives ${c.driving}`,
        56,
        ly,
      );
      ly += 24;
    }
    ly += 8;
    for (const ax of params.result.visualAxes.slice(0, 5)) {
      ctx.fillText(
        `• ${ax.title}: ${ax.topItem.label} (${(ax.topItem.prob * 100).toFixed(0)}%)`,
        56,
        ly,
      );
      ly += 24;
    }
  }

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) return;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `autobolt-evidence-${Date.now()}.png`;
  a.click();
  URL.revokeObjectURL(url);
}

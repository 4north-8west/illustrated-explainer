import dotenv from 'dotenv';
dotenv.config({ override: true });
import express from 'express';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { classifyImage, buildGenerationContext, resolveStyleFromClassified } from './analysis/classify.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const XAI_API_KEY = process.env.XAI_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const CACHE_VERSION = 'v1';
const GENERATED_DIR = path.join(__dirname, 'generated');
const CONFIG_PATH = path.join(__dirname, 'model-config.json');
const ANALYSIS_DIR = path.join(GENERATED_DIR, '_analysis');
const LLAMA_SERVER_URL = process.env.LLAMA_SERVER_URL || 'http://localhost:8080';

if (!XAI_API_KEY) {
  console.warn('XAI_API_KEY is not set. xAI cloud models will be unavailable.');
}

fs.mkdirSync(GENERATED_DIR, { recursive: true });

// --- Page-to-folder mapping (persisted to disk) ---

const METADATA_PATH = path.join(GENERATED_DIR, 'metadata.json');

function loadMetadata() {
  try { return JSON.parse(fs.readFileSync(METADATA_PATH, 'utf8')); }
  catch { return {}; }
}

function saveMetadata(meta) {
  fs.writeFileSync(METADATA_PATH, JSON.stringify(meta, null, 2));
}

function slugify(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60);
}

let pageMeta = loadMetadata();

// --- Model configuration (persisted) ---

const DEFAULT_MODEL_CONFIG = {
  localOnly: false,
  // ── Initial run ─────────────────────────────────────────────────────
  generation: { provider: 'xai', model: 'grok-imagine-image' },   // first-page IMAGE generation
  classify:   { provider: 'local', model: 'auto' },               // upload-time / generated-image classification (Phase A)
  // ── Subsequent information ──────────────────────────────────────────
  analysis:   { provider: 'local', model: 'auto' },               // Learn-panel description + explanation
  // ── Drill in ────────────────────────────────────────────────────────
  editing:    { provider: 'xai', model: 'grok-imagine-image' },   // drill IMAGE generation
  drillText:  { provider: 'local', model: 'auto' },               // drill TEXT generation (markdown / chart / table / diagram)
  // ── Cloud-fallback policy ───────────────────────────────────────────
  allowClassifyCloudFallback: false,
};

function loadModelConfig() {
  try { return { ...DEFAULT_MODEL_CONFIG, ...JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) }; }
  catch { return { ...DEFAULT_MODEL_CONFIG }; }
}

function saveModelConfig(config) {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
}

let modelConfig = loadModelConfig();

function enforceLocalOnly(provider, capability) {
  if (modelConfig.localOnly && provider !== 'local') {
    throw new Error(`Keep it local is enabled. ${capability} is blocked until a local ${capability.toLowerCase()} model is configured.`);
  }
}

// Available models registry — add new providers/models here
const MODEL_REGISTRY = {
  xai: {
    name: 'xAI (Grok)',
    models: {
      'grok-imagine-image': { name: 'Grok Imagine', capabilities: ['generation', 'editing'] },
      'grok-4-1-fast-non-reasoning': { name: 'Grok 4.1 Fast', capabilities: ['analysis'] },
    },
    generationUrl: 'https://api.x.ai/v1/images/generations',
    editingUrl: 'https://api.x.ai/v1/images/edits',
    chatUrl: 'https://api.x.ai/v1/chat/completions',
    authHeader: () => `Bearer ${XAI_API_KEY}`,
  },
  openai: {
    name: 'OpenAI',
    models: {
      'gpt-image-2': { name: 'GPT Image 2', capabilities: ['generation', 'editing'] },
      'gpt-4o': { name: 'GPT-4o', capabilities: ['analysis'] },
      'gpt-4o-mini': { name: 'GPT-4o Mini', capabilities: ['analysis'] },
    },
    generationUrl: 'https://api.openai.com/v1/images/generations',
    editingUrl: 'https://api.openai.com/v1/images/edits',
    chatUrl: 'https://api.openai.com/v1/chat/completions',
    authHeader: () => `Bearer ${OPENAI_API_KEY}`,
  },
  gemini: {
    name: 'Google Gemini',
    models: {
      'gemini-2.5-flash-image': { name: 'Gemini 2.5 Flash Image', capabilities: ['generation', 'editing'] },
      'gemini-3-pro-image-preview': { name: 'Gemini 3 Pro Image', capabilities: ['generation', 'editing'] },
      'gemini-2.5-flash': { name: 'Gemini 2.5 Flash', capabilities: ['analysis'] },
      'gemini-2.0-flash': { name: 'Gemini 2.0 Flash', capabilities: ['analysis'] },
    },
    chatUrl: `https://generativelanguage.googleapis.com/v1beta/models`,
    authHeader: () => null,
  },
  local: {
    // Each model entry can specify its own chatUrl to target a specific
    // llama-server port — useful when running multiple llama-server instances
    // (e.g. one model per port). When chatUrl is omitted, the provider-level
    // LLAMA_SERVER_URL is used.
    name: 'Local (llama-server)',
    models: {
      'auto':                        { name: 'Auto-detect (whatever 8080 has loaded)', capabilities: ['analysis', 'classify', 'drillText'] },
      'gemma-4-E2B':                 { name: 'Gemma 4 E2B  · 2B vision · fastest, lower quality', capabilities: ['analysis', 'classify', 'drillText'] },
      'gemma-4-E4B':                 { name: 'Gemma 4 E4B  · 4B vision · balanced', capabilities: ['analysis', 'classify', 'drillText'] },
      'gemma-4-E4B-it-Q4_K_S':       { name: 'Gemma 4 E4B Q4_K_S  · 4B vision · current 8080 default', capabilities: ['analysis', 'classify', 'drillText'] },
      'gemma-4-26B':                 { name: 'Gemma 4 26B  · MoE vision · medium', capabilities: ['analysis', 'classify', 'drillText'], chatUrl: 'http://localhost:8082/v1/chat/completions' },
      'gemma-4-26B-A4B-APEX-I-Mini': { name: 'Gemma 4 26B-A4B APEX  · MoE vision · medium variant', capabilities: ['analysis', 'classify', 'drillText'], chatUrl: 'http://localhost:8082/v1/chat/completions' },
      'gemma-4-31B':                 { name: 'Gemma 4 31B  · vision · highest quality, slowest', capabilities: ['analysis', 'classify', 'drillText'], chatUrl: 'http://localhost:8083/v1/chat/completions' },
      'GLM-OCR':                     { name: 'GLM-OCR  · vision · OCR specialist', capabilities: ['analysis', 'classify'], chatUrl: 'http://localhost:8084/v1/chat/completions' },
      'olmOCR-2-7B-1025':            { name: 'olmOCR-2 7B  · vision · OCR specialist', capabilities: ['analysis', 'classify'], chatUrl: 'http://localhost:8085/v1/chat/completions' },
    },
    chatUrl: LLAMA_SERVER_URL + '/v1/chat/completions',
    authHeader: () => null,
  },
};

// Resolve the chat endpoint for a given provider/model — uses the model-level
// chatUrl override when present, falls back to the provider's chatUrl. Lets
// users target different llama-server ports per pipeline (e.g. classify on
// the 4B server, analysis on the 26B server).
function resolveChatUrl(provider, model) {
  const providerConfig = MODEL_REGISTRY[provider];
  if (!providerConfig) return null;
  const modelConfigEntry = providerConfig.models?.[model];
  return modelConfigEntry?.chatUrl || providerConfig.chatUrl || null;
}

app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));
app.use('/generated', express.static(GENERATED_DIR));

// --- Deterministic hashing ---

function hash(input) {
  return crypto.createHash('sha256').update(input).digest('hex').slice(0, 16);
}

function normalize(query) {
  return query.trim().replace(/\s+/g, ' ').toLowerCase();
}

function firstPageId(query, mode) {
  return hash(`first${CACHE_VERSION}${mode}${normalize(query)}`);
}

function childPageId(parentId, x, y, responseKind = 'image', intent = '', responseDepth = 'explain', language = '') {
  return hash(`child${CACHE_VERSION}${responseKind}${parentId}${x.toFixed(2)}${y.toFixed(2)}${normalize(intent)}${responseDepth}${normalize(language)}`);
}

function uploadPageId(buffer, label) {
  const fileHash = crypto.createHash('sha256').update(buffer).digest('hex').slice(0, 12);
  return hash(`upload${CACHE_VERSION}${fileHash}${normalize(label)}`);
}

function uploadContextPageId(pageId) {
  return hash(`upload-context${CACHE_VERSION}${pageId}`);
}

// --- Mode templates ---

const FALLBACK_MODES = {
  illustration: {
    style: `Painting style (must remain consistent across every page):
- Light warm paper background with generous margins
- Clean, even dark gray or black ink outlines, consistent thin line weight
- Soft watercolor washes, pale palette: ivory, pale green, pale blue, light gray, with restrained warm accents
- A large serif title printed at the top center of the image
- Calm, well-composed scene with breathing room

Strict exclusions:
- No decorative borders, seals, parchment aging, ornate fonts, or vintage texture
- No 3D render, photorealism, neon, dark themes, or modern app UI cards
- No dense paragraphs of text, watermarks, or tiny unreadable labels
- No tourist map roads, landmarks, transit, or "traveler-guide" framing`,

    firstPagePrompt: (query, style) => `${style}

Subject: ${query}

Compose a single 16:9 illustrated explainer page about the subject above.
Let the scene's content (objects, layout, metaphor) be whatever best
explains the subject - cross-section, exploded view, timeline, anatomy,
flow, comparison, or scene - chosen to fit this specific topic.

Output a single PNG image, 16:9. Print the title clearly inside the image.`,

    childPagePrompt: (style) => `${style}

You are continuing an illustrated explainer book.
The provided image is the previous page. A red circle marks
the area the reader pointed at.

Generate the next page: a single 16:9 image that goes deeper
into whatever the red circle is on - zoom in, expand its inner
structure, or show its mechanism.

Critical: match the painting style of the provided image exactly
- same line weight, same paper tone, same pastel palette, same
title typography. The two pages must feel like consecutive spreads
in the same hand-drawn book.

Do NOT include the red circle or any cursor mark in the output.

Output a single PNG image, 16:9.`,
  },

  historical_map: {
    style: `Cartographic style (must remain consistent across every page):
- Warm cream or light parchment-toned background — clean and modern, NOT artificially aged
- Accurate geographic shapes and proportions — coastlines, borders, and landmasses should be recognizable and geographically correct
- Territories and regions filled with distinct muted watercolor washes: sage green, dusty blue, warm sand, pale terracotta, soft lavender — each region clearly distinguishable
- Clean dark ink outlines for borders and coastlines, consistent medium line weight
- Bodies of water in pale blue-gray with subtle wave texture or horizontal line hatching
- A large serif title at the top center with the time period or date range below it
- Clear, legible geographic labels: region names, city names, key landmarks — placed inside or adjacent to their locations
- Arrows, routes, or movement lines in a contrasting color (deep red or dark blue) where relevant
- A simple legend or key in one corner if the map uses symbols or color coding
- Important dates, battle sites, or events marked with small icons or numbered markers

Strict exclusions:
- No heavy parchment aging, burn marks, coffee stains, or pirate-map aesthetics
- No 3D terrain rendering, satellite imagery, or photorealistic landscapes
- No modern UI elements, app-style cards, or infographic chrome
- No dense paragraphs of text — labels and annotations only
- No fantasy or fictional geography — accuracy matters`,

    firstPagePrompt: (query, style) => `${style}

Subject: ${query}

Create a single 16:9 historical map page about the subject above.
Show the relevant geography accurately — correct continent shapes, coastlines,
borders for the time period. Use color-coded regions, labeled cities and landmarks,
movement arrows for campaigns or migrations, and date annotations.

Choose the right geographic scope: if the subject is about a single battle, show
the regional theater; if it's about an empire, show its full territorial extent;
if it's about a migration or trade route, show the full path with origin and
destination. Include a brief date range or time period near the title.

Output a single PNG image, 16:9. Print the title and time period clearly inside the image.`,

    childPagePrompt: (style) => `${style}

You are continuing a historical atlas.
The provided image is the previous map page. A red circle marks
the area the reader pointed at.

Generate the next page: a single 16:9 map that zooms into the region
or location marked by the red circle. Show greater geographic detail —
more cities, battle positions, territorial divisions, trade routes,
or chronological progression within that area.

If the red circle is on a city, show a detailed regional map around it.
If it's on a territory, show its internal divisions and key locations.
If it's on an arrow or route, show the detailed stages of that journey.
If it's on a date marker or event label, show a detailed map of that
specific event — troop movements, siege positions, or territorial changes.

Critical: match the cartographic style of the provided image exactly
- same color palette for regions, same ink weight, same label typography,
same background tone. The two pages must feel like consecutive spreads
in the same historical atlas.

Do NOT include the red circle or any cursor mark in the output.

Output a single PNG image, 16:9.`,
  },
};

const MODES_DIR = path.join(__dirname, 'modes');

function renderTemplate(template, values) {
  return String(template || '').replace(/\{\{(\w+)\}\}/g, (_match, key) => values[key] ?? '');
}

function normalizeModeConfig(mode, fallbackId = null) {
  const id = mode.id || fallbackId;
  if (!id || !/^[a-z0-9_]+$/.test(id)) throw new Error(`Invalid mode id: ${id}`);
  if (!mode.style) throw new Error(`Mode ${id} is missing style`);

  const firstPageTemplate = mode.firstPageTemplate || (mode.firstPagePrompt ? mode.firstPagePrompt('{{query}}', '{{style}}') : null);
  const childPageTemplate = mode.childPageTemplate || (mode.childPagePrompt ? mode.childPagePrompt('{{style}}') : null);
  if (!firstPageTemplate || !childPageTemplate) throw new Error(`Mode ${id} is missing prompt templates`);

  return {
    id,
    label: mode.label || id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    tagLabel: mode.tagLabel || id.replace(/_/g, ' '),
    placeholder: mode.placeholder || "Type a topic...",
    description: mode.description || '',
    style: mode.style,
    firstPageTemplate,
    childPageTemplate,
    modeLabelForPrompt: mode.modeLabelForPrompt || mode.label || id.replace(/_/g, ' '),
  };
}

function loadModes() {
  const modes = {};
  for (const [id, mode] of Object.entries(FALLBACK_MODES)) {
    modes[id] = normalizeModeConfig({ ...mode, id }, id);
  }

  try {
    const entries = fs.readdirSync(MODES_DIR, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
      const filePath = path.join(MODES_DIR, entry.name);
      try {
        const loaded = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        const mode = normalizeModeConfig(loaded, path.basename(entry.name, '.json'));
        modes[mode.id] = mode;
      } catch (err) {
        console.warn(`[modes] Skipping ${filePath}: ${err.message}`);
      }
    }
  } catch (err) {
    console.warn(`[modes] Using fallback modes only: ${err.message}`);
  }

  return modes;
}

let MODES = loadModes();
let VALID_MODES = Object.keys(MODES);

function inferModeFromQuery(query) {
  const text = normalize(query);
  const hasMode = id => VALID_MODES.includes(id);

  if (hasMode('historical_map') && /\b(map|atlas|territor|empire|kingdom|dynasty|border|migration|trade route|campaign|battle|war|invasion|colonial|ancient|medieval|revolution|civilization|rome|roman|mongol|ottoman|byzantine|aztec|inca|maya)\b/.test(text)) {
    return 'historical_map';
  }

  if (hasMode('math_equation') && /(\b(equation|formula|theorem|proof|derive|derivative|integral|calculus|algebra|geometry|matrix|vector|probability|statistics|function|graph|slope|limit|variable|polynomial|trigonometry)\b|[=∫∑√π])/.test(text)) {
    return 'math_equation';
  }

  if (hasMode('science_process') && /\b(process|cycle|reaction|molecule|cell|organ|anatomy|photosynthesis|respiration|ecosystem|climate|weather|volcano|earthquake|plate tectonic|evolution|gravity|force|energy|electricity|magnetism|atom|protein|dna|rna|enzyme|immune|planet|star|orbit)\b/.test(text)) {
    return 'science_process';
  }

  return hasMode('illustration') ? 'illustration' : VALID_MODES[0];
}

function publicMode(mode) {
  return {
    id: mode.id,
    label: mode.label,
    tagLabel: mode.tagLabel,
    placeholder: mode.placeholder,
    description: mode.description,
  };
}

// --- Red marker compositing ---

async function compositeRedMarker(imagePath, nx, ny) {
  const image = sharp(imagePath);
  const metadata = await image.metadata();
  const w = metadata.width;
  const h = metadata.height;
  const cx = Math.round(nx * w);
  const cy = Math.round(ny * h);
  const radius = Math.round(w * 0.04);
  const innerRadius = Math.round(radius * 0.3);

  const svg = `<svg width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg">
    <circle cx="${cx}" cy="${cy}" r="${radius}" fill="rgba(255,0,0,0.25)" stroke="rgba(255,0,0,0.9)" stroke-width="3"/>
    <circle cx="${cx}" cy="${cy}" r="${innerRadius}" fill="rgba(255,0,0,0.9)"/>
  </svg>`;

  return await sharp(imagePath)
    .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
    .png()
    .toBuffer();
}

async function compositeFocusImage(imagePath, nx, ny) {
  const metadata = await sharp(imagePath).metadata();
  const w = metadata.width;
  const h = metadata.height;
  const cx = Math.round(nx * w);
  const cy = Math.round(ny * h);
  const cropSize = Math.round(Math.min(w, h) * 0.38);
  const cropW = Math.min(w, cropSize);
  const cropH = Math.min(h, cropSize);
  const left = Math.round(Math.max(0, Math.min(w - cropW, cx - cropW / 2)));
  const top = Math.round(Math.max(0, Math.min(h - cropH, cy - cropH / 2)));
  const cropCenterX = cx - left;
  const cropCenterY = cy - top;

  const markerSvg = (width, height, x, y, label) => {
    const radius = Math.round(Math.min(width, height) * 0.08);
    const innerRadius = Math.max(6, Math.round(radius * 0.28));
    return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="${width}" height="34" fill="rgba(255,255,255,0.88)"/>
      <text x="14" y="23" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#333">${label}</text>
      <line x1="${x - radius * 1.35}" y1="${y}" x2="${x + radius * 1.35}" y2="${y}" stroke="rgba(220,0,0,0.95)" stroke-width="5"/>
      <line x1="${x}" y1="${y - radius * 1.35}" x2="${x}" y2="${y + radius * 1.35}" stroke="rgba(220,0,0,0.95)" stroke-width="5"/>
      <circle cx="${x}" cy="${y}" r="${radius}" fill="rgba(255,0,0,0.20)" stroke="rgba(220,0,0,1)" stroke-width="6"/>
      <circle cx="${x}" cy="${y}" r="${innerRadius}" fill="rgba(220,0,0,1)"/>
    </svg>`;
  };

  const fullPanel = await sharp(imagePath)
    .resize(900, 720, { fit: 'inside', background: '#faf8f5' })
    .extend({ top: 0, bottom: 0, left: 0, right: 0, background: '#faf8f5' })
    .toBuffer();
  const fullMeta = await sharp(fullPanel).metadata();
  const fullScale = Math.min(fullMeta.width / w, fullMeta.height / h);
  const fullX = Math.round(cx * fullScale);
  const fullY = Math.round(cy * fullScale);
  const markedFull = await sharp(fullPanel)
    .composite([{ input: Buffer.from(markerSvg(fullMeta.width, fullMeta.height, fullX, fullY, 'FULL IMAGE - CLICK MARKED')), top: 0, left: 0 }])
    .png()
    .toBuffer();

  const cropPanel = await sharp(imagePath)
    .extract({ left, top, width: cropW, height: cropH })
    .resize(900, 720, { fit: 'inside', background: '#faf8f5' })
    .toBuffer();
  const cropMeta = await sharp(cropPanel).metadata();
  const cropScale = Math.min(cropMeta.width / cropW, cropMeta.height / cropH);
  const cropX = Math.round(cropCenterX * cropScale);
  const cropY = Math.round(cropCenterY * cropScale);
  const markedCrop = await sharp(cropPanel)
    .composite([{ input: Buffer.from(markerSvg(cropMeta.width, cropMeta.height, cropX, cropY, 'ZOOMED CLICK REGION - ANSWER ABOUT THIS')), top: 0, left: 0 }])
    .png()
    .toBuffer();

  return sharp({
    create: {
      width: 1840,
      height: 760,
      channels: 4,
      background: '#faf8f5',
    },
  })
    .composite([
      { input: markedFull, left: 10, top: 20 },
      { input: markedCrop, left: 930, top: 20 },
    ])
    .png()
    .toBuffer();
}

// --- Serialized generation lock ---

let generationQueue = Promise.resolve();

function enqueueGeneration(fn) {
  generationQueue = generationQueue.then(fn, fn);
  return generationQueue;
}

// --- API calls (model-agnostic) ---

async function callGeminiGenerate(model, prompt, referenceImageBase64) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`;
  const parts = [];
  if (referenceImageBase64) {
    parts.push({ inline_data: { mime_type: 'image/png', data: referenceImageBase64 } });
  }
  parts.push({ text: prompt });

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts }],
      generationConfig: { responseModalities: ['IMAGE'] },
    }),
    signal: AbortSignal.timeout(180000),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Gemini API error ${response.status}: ${err.slice(0, 300)}`);
  }

  const data = await response.json();
  // Find the image part in the response (Gemini uses camelCase: inlineData)
  for (const candidate of data.candidates || []) {
    for (const part of candidate.content?.parts || []) {
      const imgData = part.inlineData || part.inline_data;
      if (imgData?.mimeType?.startsWith('image/') || imgData?.mime_type?.startsWith('image/')) {
        return Buffer.from(imgData.data, 'base64');
      }
    }
  }
  throw new Error('Gemini returned no image in response');
}

async function callGeneration(prompt) {
  const cfg = modelConfig.generation;
  const provider = MODEL_REGISTRY[cfg.provider];
  enforceLocalOnly(cfg.provider, 'Image generation');

  if (!provider?.generationUrl && cfg.provider !== 'gemini') {
    throw new Error(`No local image generation model is configured yet. Disable "keep it local" or add a local generation provider.`);
  }

  if (cfg.provider === 'gemini') {
    return callGeminiGenerate(cfg.model, prompt, null);
  }

  let body, headers;

  if (cfg.provider === 'openai') {
    body = JSON.stringify({
      model: cfg.model,
      prompt,
      size: '1536x1024',
      n: 1,
    });
    headers = { 'Content-Type': 'application/json', 'Authorization': provider.authHeader() };
  } else {
    // xAI and compatible
    body = JSON.stringify({
      model: cfg.model,
      prompt,
      aspect_ratio: '16:9',
      response_format: 'b64_json',
      n: 1,
    });
    headers = { 'Content-Type': 'application/json', 'Authorization': provider.authHeader() };
  }

  const response = await fetch(provider.generationUrl, {
    method: 'POST', headers, body,
    signal: AbortSignal.timeout(180000),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`${cfg.provider} API error ${response.status}: ${err.slice(0, 300)}`);
  }
  const data = await response.json();
  return Buffer.from(data.data[0].b64_json, 'base64');
}

async function callEditing(prompt, imageBuffer) {
  const cfg = modelConfig.editing;
  const provider = MODEL_REGISTRY[cfg.provider];
  enforceLocalOnly(cfg.provider, 'Image editing');

  if (!provider?.editingUrl && cfg.provider !== 'gemini') {
    throw new Error(`No local image editing model is configured yet. Disable "keep it local" or add a local editing provider.`);
  }

  if (cfg.provider === 'gemini') {
    const base64Image = imageBuffer.toString('base64');
    return callGeminiGenerate(cfg.model, prompt, base64Image);
  }

  if (cfg.provider === 'openai') {
    // OpenAI editing uses multipart/form-data with image as a file upload
    const { Blob } = await import('node:buffer');
    const formData = new FormData();
    formData.append('model', cfg.model);
    formData.append('prompt', prompt);
    formData.append('size', '1536x1024');
    formData.append('n', '1');
    const imageBlob = new Blob([imageBuffer], { type: 'image/png' });
    formData.append('image[]', imageBlob, 'parent.png');

    const response = await fetch(provider.editingUrl, {
      method: 'POST',
      headers: { 'Authorization': provider.authHeader() },
      body: formData,
      signal: AbortSignal.timeout(180000),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`${cfg.provider} API error ${response.status}: ${err.slice(0, 300)}`);
    }
    const data = await response.json();
    return Buffer.from(data.data[0].b64_json, 'base64');
  }

  // xAI and compatible — JSON body with base64 image
  const base64Image = imageBuffer.toString('base64');
  const response = await fetch(provider.editingUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': provider.authHeader(),
    },
    body: JSON.stringify({
      model: cfg.model,
      prompt,
      image: {
        url: `data:image/png;base64,${base64Image}`,
        type: 'image_url',
      },
      aspect_ratio: '16:9',
      response_format: 'b64_json',
      n: 1,
    }),
    signal: AbortSignal.timeout(180000),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`${cfg.provider} API error ${response.status}: ${err.slice(0, 300)}`);
  }
  const data = await response.json();
  return Buffer.from(data.data[0].b64_json, 'base64');
}

async function generateFirstPage(query, mode) {
  const modeConfig = MODES[mode];
  const prompt = renderTemplate(modeConfig.firstPageTemplate, { style: modeConfig.style, query });
  return callGeneration(prompt);
}

async function generateChildPage(compositedImageBuffer, mode, intent = '', contextTrail = [], parentClassified = null) {
  const modeConfig = MODES[mode];
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const intentPrompt = intent
    ? `\n\nLearner intent for this drill-down: ${intent}\nUse that intent to decide what details to reveal while still focusing on the marked region.`
    : '';
  const sourcePrompt = uploadContextPrompt(uploadContext, 2500);
  // Phase D: when the parent has a classified style preset, use it as the BASE
  // style for the drill child instead of the mode's stock template. Falls back
  // to the mode's style when classification is absent or the preset is unknown.
  // This is the lever that stops every drill from coming back as watercolor
  // when the parent is, say, a flat-color poster or a technical schematic.
  const styleFromClassified = resolveStyleFromClassified(parentClassified);
  const styleForPrompt = styleFromClassified || modeConfig.style;

  // Phase A.2: append the parent's classified payload as additional context
  // (category-specific facts: chart axes, flyer dates, diagram components, etc.).
  const parentCtx = buildGenerationContext(parentClassified);
  // We've already swapped the style template above, so the framing changes
  // from "match the parent" to "this is what the parent shows" — content reuse
  // without re-stating the style command twice.
  const parentCtxBlock = parentCtx
    ? `\n\n${parentCtx}\n\nUse this context to decide what to draw; the style instructions above already match the parent.`
    : '';
  const prompt = renderTemplate(modeConfig.childPageTemplate, { style: styleForPrompt }) + intentPrompt + sourcePrompt + parentCtxBlock;
  return callEditing(prompt, compositedImageBuffer);
}

function responseDepthInstruction(responseDepth, language = '') {
  const languageInstruction = language
    ? `\nLanguage: respond in ${language}. Preserve formulas, code, variable names, and exact source quotations where needed.`
    : '';
  if (responseDepth === 'extract') {
    return `Response mode: Extract. Return what is visibly present in the selected region as literally as possible. Minimize interpretation. If a translation language is requested, include both "Original extraction" and "Translation".${languageInstruction}`;
  }
  if (responseDepth === 'teach') {
    return `Response mode: Teach concepts. Use the selected region as the starting point, then teach the broader concepts needed to understand it. Stay grounded in the selected region.${languageInstruction}`;
  }
  return `Response mode: Explain here. Explain the selected region specifically in the context of this image or document. Do not drift to unrelated concepts.${languageInstruction}`;
}

function pageJsonPath(folder, id) {
  return path.join(GENERATED_DIR, folder, `${id}.json`);
}

function loadPageContent(folder, id) {
  try { return JSON.parse(fs.readFileSync(pageJsonPath(folder, id), 'utf8')); }
  catch { return null; }
}

function savePageContent(folder, id, page) {
  fs.writeFileSync(pageJsonPath(folder, id), JSON.stringify(page, null, 2));
}

// Load the cached classified payload for a page, or null if not yet classified
// (or page is not an image). Phase A.2 uses this to thread parent context into
// the drill child generation prompts.
function loadParentClassified(parentId) {
  if (!parentId) return null;
  const analysis = loadAnalysis(parentId);
  return analysis?.classified || null;
}

function buildContextTrail(pageId) {
  const trail = [];
  let currentId = pageId;
  const seen = new Set();
  while (currentId && !seen.has(currentId)) {
    seen.add(currentId);
    const meta = pageMeta[currentId];
    if (!meta) break;
    trail.unshift({
      id: currentId,
      type: meta.type || 'image',
      query: meta.query || '',
      mode: meta.mode || 'illustration',
      parentClick: meta.parentClick || null,
      intent: meta.intent || '',
    });
    currentId = meta.parentId;
  }
  return trail;
}

function findNearestUploadContext(pageId) {
  let currentId = pageId;
  const seen = new Set();
  while (currentId && !seen.has(currentId)) {
    seen.add(currentId);
    const meta = pageMeta[currentId];
    if (!meta) break;
    if (meta.contextPageId) {
      const context = loadPageContent(meta.folder, meta.contextPageId);
      if (context) return context;
    }
    if (meta.type === 'upload_context') {
      const context = loadPageContent(meta.folder, currentId);
      if (context) return context;
    }
    currentId = meta.parentId;
  }
  return null;
}

function uploadContextPrompt(uploadContext, maxChars = 5000) {
  if (!uploadContext?.content) return '';
  return `\n\nSource upload OCR/transcription and analysis:
${uploadContext.content.slice(0, maxChars)}

Use this source context as primary evidence for uploaded images, especially visible text, labels, equations, axis names, legends, and table/chart structure. If the focused crop is ambiguous, prefer values and labels supported by this context and explicitly mark uncertainty.`;
}

async function generateTextDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '', parentClassified = null) {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContextPrompt(uploadContext);
  const parentCtx = buildGenerationContext(parentClassified);
  const parentCtxBlock = parentCtx ? `\n\n${parentCtx}` : '';

  const systemPrompt = `You are an expert STEM-capable educator. Explain only the exact selected region in an image, not the image as a whole. Prefer precise text, equations, tables, or concise examples over generating another image. Use Markdown. If math is needed, use valid LaTeX with inline math in $...$ and display math in $$...$$.`;
  const userPrompt = `The image has two panels. The left panel shows the full page with the click marked. The right panel is a zoomed crop centered on the clicked location and labeled "ZOOMED CLICK REGION - ANSWER ABOUT THIS".

Focus your answer on the feature at the center of the right zoomed panel. Use the full left panel only for context.

The learner clicked this region in a ${modeLabel}.

Learner intent: ${intent || 'Explain what is marked and why it matters.'}

${responseDepthInstruction(responseDepth, language)}

Drill-down context:
${contextText || 'No prior context available.'}${uploadContextText}${parentCtxBlock}

Respond with a focused Markdown learning page:
- Start with a short title.
- Identify what the center of the right zoomed region appears to be.
- Answer the learner intent directly.
- Include equations, definitions, or small tables when they help.
- Put important equations on their own lines using $$...$$.
- Mention uncertainty explicitly if the image does not provide enough information.
- Do not explain unrelated areas of the image.`;

  const cfg = modelConfig.drillText;  // drill text
  try {
    const text = await callVisionChat(cfg.provider, cfg.model, imageBase64, systemPrompt, userPrompt);
    return { text, source: cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})` };
  } catch (err) {
    if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
      const text = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
      return { text, source: 'grok (fallback)' };
    }
    throw err;
  }
}

function parseJsonResponse(text) {
  const raw = String(text || '').trim();
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const jsonText = fenced ? fenced[1].trim() : raw;
  const start = jsonText.indexOf('{');
  const end = jsonText.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) throw new Error('No JSON object found in model response');
  return JSON.parse(jsonText.slice(start, end + 1));
}

function normalizeChartSpec(spec) {
  const chart = spec && typeof spec === 'object' ? spec : {};
  const type = ['line', 'bar', 'scatter'].includes(chart.type) ? chart.type : 'line';
  const chartability = ['high', 'medium', 'low'].includes(chart.chartability) ? chart.chartability : 'medium';
  const fallbackRecommendation = ['chart', 'table', 'text', 'diagram'].includes(chart.fallbackRecommendation)
    ? chart.fallbackRecommendation
    : (chartability === 'low' ? 'text' : 'chart');
  const confidence = Number.isFinite(Number(chart.confidence))
    ? Math.max(0, Math.min(1, Number(chart.confidence)))
    : null;
  const points = Array.isArray(chart.points) ? chart.points
    .map((point, index) => ({
      label: String(point.label ?? point.x ?? index + 1),
      x: Number.isFinite(Number(point.x)) ? Number(point.x) : index,
      y: Number.isFinite(Number(point.y)) ? Number(point.y) : 0,
      evidence: String(point.evidence || '').slice(0, 240),
      confidence: Number.isFinite(Number(point.confidence)) ? Math.max(0, Math.min(1, Number(point.confidence))) : null,
    }))
    .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y))
    .slice(0, 80) : [];

  return {
    type,
    chartability,
    fallbackRecommendation,
    requiresUserConfirmation: Boolean(chart.requiresUserConfirmation),
    confidence,
    title: String(chart.title || 'Chart drill-down').slice(0, 120),
    xLabel: String(chart.xLabel || 'x').slice(0, 80),
    yLabel: String(chart.yLabel || 'y').slice(0, 80),
    points,
    sourceEvidence: Array.isArray(chart.sourceEvidence) ? chart.sourceEvidence.map(item => String(item).slice(0, 240)).filter(Boolean).slice(0, 8) : [],
    notes: Array.isArray(chart.notes) ? chart.notes.map(note => String(note).slice(0, 240)).slice(0, 6) : [],
    uncertainty: String(chart.uncertainty || '').slice(0, 500),
  };
}

function validateChartSpec(chart) {
  const issues = [];
  if (!['line', 'bar', 'scatter'].includes(chart.type)) issues.push('Chart type must be line, bar, or scatter.');
  if (!['high', 'medium', 'low'].includes(chart.chartability)) issues.push('chartability must be high, medium, or low.');
  if (!['chart', 'table', 'text', 'diagram'].includes(chart.fallbackRecommendation)) issues.push('fallbackRecommendation must be chart, table, text, or diagram.');
  if (chart.chartability !== 'low' && chart.points.length < 2) issues.push('A chartable response needs at least two plotted points.');
  if (chart.chartability === 'low' && chart.fallbackRecommendation === 'chart') issues.push('Low chartability needs a non-chart fallbackRecommendation.');
  if (chart.chartability !== 'low' && (!chart.title || chart.title === 'Chart drill-down')) issues.push('Chart title is missing or generic.');
  if (chart.chartability !== 'low' && (!chart.xLabel || chart.xLabel === 'x' || !chart.yLabel || chart.yLabel === 'y')) {
    issues.push('Axis labels are missing or generic.');
  }
  const missingEvidence = chart.points.filter(point => !point.evidence).length;
  if (chart.chartability !== 'low' && missingEvidence) issues.push(`${missingEvidence} plotted point(s) lack evidence.`);
  const missingConfidence = chart.points.filter(point => point.confidence === null).length;
  if (chart.chartability !== 'low' && missingConfidence) issues.push(`${missingConfidence} plotted point(s) lack confidence.`);
  if (chart.confidence !== null && chart.confidence < 0.45 && chart.chartability === 'high') {
    issues.push('Overall confidence is too low for high chartability.');
  }
  if (chart.requiresUserConfirmation && !chart.uncertainty) issues.push('requiresUserConfirmation is true but uncertainty is blank.');

  return {
    issues,
    shouldRetry: issues.some(issue => (
      issue.includes('at least two') ||
      issue.includes('generic') ||
      issue.includes('lack evidence') ||
      issue.includes('lack confidence') ||
      issue.includes('non-chart')
    )),
  };
}

function chartFallbackMarkdown(chart) {
  const recommendation = chart.fallbackRecommendation && chart.fallbackRecommendation !== 'chart'
    ? chart.fallbackRecommendation
    : 'text';
  const evidence = [
    ...(chart.sourceEvidence || []),
    ...(chart.points || []).filter(point => point.evidence).map(point => `${point.label}: ${point.evidence}`),
  ].slice(0, 8);
  const notes = chart.notes || [];
  const validation = chart.validationIssues || [];
  return `# Chart Not Recommended

The selected region does not have enough reliable chart evidence to render a trustworthy chart.

**Recommended format:** ${recommendation}

${chart.uncertainty ? `**Uncertainty:** ${chart.uncertainty}\n` : ''}
${chart.confidence !== null ? `**Model confidence:** ${Math.round(chart.confidence * 100)}%\n` : ''}

${evidence.length ? `## Evidence\n${evidence.map(item => `- ${item}`).join('\n')}\n` : ''}
${notes.length ? `## Notes\n${notes.map(item => `- ${item}`).join('\n')}\n` : ''}
${validation.length ? `## Validation Notes\n${validation.map(item => `- ${item}`).join('\n')}\n` : ''}`;
}

function normalizeTableSpec(spec) {
  const table = spec && typeof spec === 'object' ? spec : {};
  const columns = Array.isArray(table.columns)
    ? table.columns.map(column => String(column).slice(0, 80)).filter(Boolean).slice(0, 8)
    : [];
  const rows = Array.isArray(table.rows)
    ? table.rows.map(row => {
      const values = Array.isArray(row) ? row : columns.map(column => row?.[column] ?? '');
      return values.map(value => String(value ?? '').slice(0, 240)).slice(0, columns.length || 8);
    }).filter(row => row.length).slice(0, 40)
    : [];
  const safeColumns = columns.length ? columns : rows[0]?.map((_value, index) => `Column ${index + 1}`) || ['Item', 'Explanation'];

  return {
    title: String(table.title || 'Table drill-down').slice(0, 120),
    columns: safeColumns,
    rows: rows.map(row => safeColumns.map((_column, index) => row[index] ?? '')),
    notes: Array.isArray(table.notes) ? table.notes.map(note => String(note).slice(0, 240)).slice(0, 6) : [],
    uncertainty: String(table.uncertainty || '').slice(0, 500),
  };
}

function normalizeDiagramSpec(spec) {
  const diagram = spec && typeof spec === 'object' ? spec : {};
  const rawNodes = Array.isArray(diagram.nodes) ? diagram.nodes : [];
  const nodes = rawNodes.map((node, index) => ({
    id: String(node.id || `n${index + 1}`).replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 30) || `n${index + 1}`,
    label: String(node.label || node.id || `Step ${index + 1}`).slice(0, 90),
    detail: String(node.detail || '').slice(0, 220),
    kind: ['concept', 'process', 'data', 'warning', 'result'].includes(node.kind) ? node.kind : 'concept',
  })).slice(0, 16);
  const nodeIds = new Set(nodes.map(node => node.id));
  const edges = Array.isArray(diagram.edges) ? diagram.edges
    .map(edge => ({
      from: String(edge.from || ''),
      to: String(edge.to || ''),
      label: String(edge.label || '').slice(0, 70),
    }))
    .filter(edge => nodeIds.has(edge.from) && nodeIds.has(edge.to) && edge.from !== edge.to)
    .slice(0, 24) : [];

  return {
    title: String(diagram.title || 'Diagram drill-down').slice(0, 120),
    direction: diagram.direction === 'LR' ? 'LR' : 'TB',
    nodes: nodes.length ? nodes : [{ id: 'n1', label: 'Selected region', detail: 'The model did not identify enough structure for a diagram.', kind: 'concept' }],
    edges,
    notes: Array.isArray(diagram.notes) ? diagram.notes.map(note => String(note).slice(0, 240)).slice(0, 6) : [],
    uncertainty: String(diagram.uncertainty || '').slice(0, 500),
  };
}

async function generateChartDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '', parentClassified = null) {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContextPrompt(uploadContext);
  const parentCtx = buildGenerationContext(parentClassified);
  const parentCtxBlock = parentCtx ? `\n\n${parentCtx}` : '';

  const systemPrompt = `You convert only the exact selected image region into a simple educational chart specification. Return JSON only. Use inferred or approximate values only when the selected region, OCR/transcription context, and surrounding page context support them. Describe uncertainty and cite evidence for every plotted point.`;
  const userPrompt = `The image has two panels. The left panel shows the full page with the click marked. The right panel is a zoomed crop centered on the clicked location and labeled "ZOOMED CLICK REGION - ANSWER ABOUT THIS".

Focus on the feature at the center of the right zoomed panel. Use the full left panel only for context.

The learner clicked this region in a ${modeLabel}.

Learner intent: ${intent || 'Turn the marked region into a useful chart or graph if appropriate.'}

${responseDepthInstruction(responseDepth, language)}

Drill-down context:
${contextText || 'No prior context available.'}${uploadContextText}${parentCtxBlock}

Return one JSON object only, with this shape:
{
  "type": "line" | "bar" | "scatter",
  "chartability": "high" | "medium" | "low",
  "fallbackRecommendation": "chart" | "table" | "text" | "diagram",
  "requiresUserConfirmation": false,
  "confidence": 0.0,
  "title": "short chart title",
  "xLabel": "x axis label",
  "yLabel": "y axis label",
  "points": [
    { "label": "visible label", "x": 0, "y": 0, "evidence": "visible source for this point", "confidence": 0.0 }
  ],
  "sourceEvidence": ["short quote or visual cue used to build the chart"],
  "notes": ["short teaching note"],
  "uncertainty": "what is inferred or approximate"
}

Rules:
- Use 2-20 points unless the image clearly supports more.
- If the marked region is not numeric data, make a conceptual sequence chart using ordinal x values only when that chart would teach the selected region better than text/table/diagram.
- Use chartability "low" when a chart would be misleading, when the evidence is too thin, or when a table/text/diagram is the better format.
- Use fallbackRecommendation to name the better format when chartability is low.
- Set requiresUserConfirmation to true when values are approximate, inferred, or need user review before being trusted.
- Every point must include evidence and confidence.
- Do not invent precision. Prefer simple approximate values and clear uncertainty.`;

  const cfg = modelConfig.drillText;  // drill chart
  let result;
  let usedFallback = false;
  try {
    result = await callVisionChat(cfg.provider, cfg.model, imageBase64, systemPrompt, userPrompt);
  } catch (err) {
    if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
      result = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
      usedFallback = true;
    } else {
      throw err;
    }
  }

  let chart = normalizeChartSpec(parseJsonResponse(result));
  let validation = validateChartSpec(chart);
  if (validation.shouldRetry) {
    const retryPrompt = `${userPrompt}

Your previous JSON failed validation:
${validation.issues.map(issue => `- ${issue}`).join('\n')}

Previous JSON:
${JSON.stringify(chart, null, 2)}

Return corrected JSON only. If the selected region cannot support a trustworthy chart, set chartability to "low", set fallbackRecommendation to the best non-chart format, leave points empty, and explain why in uncertainty.`;
    const retryResult = await callVisionChat(
      usedFallback ? 'xai' : cfg.provider,
      usedFallback ? 'grok-4-1-fast-non-reasoning' : cfg.model,
      imageBase64,
      systemPrompt,
      retryPrompt
    );
    chart = normalizeChartSpec(parseJsonResponse(retryResult));
    validation = validateChartSpec(chart);
  }

  chart.validationIssues = validation.issues;
  chart.isReliable = validation.issues.length === 0 && chart.chartability !== 'low';
  const source = usedFallback ? 'grok (fallback)' : (cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})`);
  return { chart, source };
}

async function generateTableDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '', parentClassified = null) {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContextPrompt(uploadContext);

  const parentCtx = buildGenerationContext(parentClassified);
  const parentCtxBlock = parentCtx ? `\n\n${parentCtx}` : '';

  const systemPrompt = `You convert only the exact selected image region into a locally renderable educational table. Return JSON only. Use uploaded OCR/transcription context as primary evidence for visible text, labels, equations, and values. Preserve visible wording when extraction is requested.`;
  const userPrompt = `The image has two panels. The left panel shows the full page with the click marked. The right panel is a zoomed crop centered on the clicked location and labeled "ZOOMED CLICK REGION - ANSWER ABOUT THIS".

Focus on the feature at the center of the right zoomed panel. Use the full left panel only for context.

The learner clicked this region in a ${modeLabel}.

Learner intent: ${intent || 'Organize the marked region as a useful table.'}

${responseDepthInstruction(responseDepth, language)}

Drill-down context:
${contextText || 'No prior context available.'}${uploadContextText}${parentCtxBlock}

Return one JSON object only, with this shape:
{
  "title": "short table title",
  "columns": ["Column A", "Column B"],
  "rows": [
    ["cell", "cell"]
  ],
  "notes": ["short teaching note"],
  "uncertainty": "what is inferred or approximate"
}

Rules:
- Use tables for comparisons, extracted text blocks, equation parts, data labels, vocabulary, steps, or cause/effect relationships.
- Use 2-8 columns and 2-40 rows.
- Do not invent precise source text. If text is unclear, mark it as uncertain.`;

  const cfg = modelConfig.drillText;  // drill table
  let result;
  try {
    result = await callVisionChat(cfg.provider, cfg.model, imageBase64, systemPrompt, userPrompt);
  } catch (err) {
    if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
      result = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
      const table = normalizeTableSpec(parseJsonResponse(result));
      return { table, source: 'grok (fallback)' };
    }
    throw err;
  }

  const table = normalizeTableSpec(parseJsonResponse(result));
  return { table, source: cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})` };
}

async function generateDiagramDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '', parentClassified = null) {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContextPrompt(uploadContext);

  const parentCtx = buildGenerationContext(parentClassified);
  const parentCtxBlock = parentCtx ? `\n\n${parentCtx}` : '';

  const systemPrompt = `You convert only the exact selected image region into a locally renderable educational diagram specification. Return JSON only. Use uploaded OCR/transcription context as primary evidence for visible text, labels, equations, and relationships. The app will render the diagram itself as SVG; do not request image generation.`;
  const userPrompt = `The image has two panels. The left panel shows the full page with the click marked. The right panel is a zoomed crop centered on the clicked location and labeled "ZOOMED CLICK REGION - ANSWER ABOUT THIS".

Focus on the feature at the center of the right zoomed panel. Use the full left panel only for context.

The learner clicked this region in a ${modeLabel}.

Learner intent: ${intent || 'Turn the marked region into a concept or process diagram.'}

${responseDepthInstruction(responseDepth, language)}

Drill-down context:
${contextText || 'No prior context available.'}${uploadContextText}${parentCtxBlock}

Return one JSON object only, with this shape:
{
  "title": "short diagram title",
  "direction": "TB" | "LR",
  "nodes": [
    { "id": "n1", "label": "short node label", "detail": "optional detail", "kind": "concept" | "process" | "data" | "warning" | "result" }
  ],
  "edges": [
    { "from": "n1", "to": "n2", "label": "optional relationship label" }
  ],
  "notes": ["short teaching note"],
  "uncertainty": "what is inferred or approximate"
}

Rules:
- Use diagrams for process flows, dependency chains, equation relationships, systems, mechanisms, and cause/effect.
- Use 2-12 nodes unless the selected region clearly needs more.
- Keep labels short enough to render inside boxes.`;

  const cfg = modelConfig.drillText;  // drill diagram
  let result;
  try {
    result = await callVisionChat(cfg.provider, cfg.model, imageBase64, systemPrompt, userPrompt);
  } catch (err) {
    if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
      result = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
      const diagram = normalizeDiagramSpec(parseJsonResponse(result));
      return { diagram, source: 'grok (fallback)' };
    }
    throw err;
  }

  const diagram = normalizeDiagramSpec(parseJsonResponse(result));
  return { diagram, source: cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})` };
}

// --- Model config API ---

app.get('/api/modes', (_req, res) => {
  res.json({
    modes: VALID_MODES.map(id => publicMode(MODES[id])),
    defaultMode: VALID_MODES.includes('illustration') ? 'illustration' : VALID_MODES[0],
  });
});

app.get('/api/models', (_req, res) => {
  const registry = {};
  for (const [providerId, provider] of Object.entries(MODEL_REGISTRY)) {
    registry[providerId] = {
      name: provider.name,
      models: Object.entries(provider.models).map(([id, m]) => ({
        id, name: m.name, capabilities: m.capabilities,
      })),
    };
  }
  res.json({ current: modelConfig, registry });
});

app.post('/api/models', (req, res) => {
  const { generation, editing, analysis, classify, drillText, localOnly, allowClassifyCloudFallback } = req.body;
  if (typeof localOnly === 'boolean') modelConfig.localOnly = localOnly;
  if (typeof allowClassifyCloudFallback === 'boolean') modelConfig.allowClassifyCloudFallback = allowClassifyCloudFallback;
  for (const [key, val] of [
    ['generation', generation],
    ['editing', editing],
    ['analysis', analysis],
    ['classify', classify],
    ['drillText', drillText],
  ]) {
    if (val) {
      if (!MODEL_REGISTRY[val.provider]?.models[val.model]) {
        return res.status(400).json({ error: `Unknown model: ${val.provider}/${val.model}` });
      }
      modelConfig[key] = val;
    }
  }
  saveModelConfig(modelConfig);
  res.json({ current: modelConfig });
});

// --- Upload API ---

app.post('/api/upload', async (req, res) => {
  const { imageData, label, mode: reqMode } = req.body;

  if (!imageData || !label) {
    return res.status(400).json({ error: 'imageData (base64) and label are required' });
  }

  const trimmedLabel = label.trim();
  if (trimmedLabel.length < 1 || trimmedLabel.length > 300) {
    return res.status(400).json({ error: 'Label must be 1-300 characters' });
  }

  const mode = (reqMode && VALID_MODES.includes(reqMode)) ? reqMode : 'illustration';

  try {
    // Decode base64 (strip data URI prefix if present)
    const base64Clean = imageData.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Clean, 'base64');

    // Convert to PNG via sharp (handles jpg, webp, etc.)
    const pngBuffer = await sharp(imageBuffer).png().toBuffer();

    const id = uploadPageId(pngBuffer, trimmedLabel);
    const folder = `upload-${slugify(trimmedLabel)}`;
    const folderPath = path.join(GENERATED_DIR, folder);
    fs.mkdirSync(folderPath, { recursive: true });

    const imagePath = path.join(folderPath, `${id}.png`);
    const imageUrl = `/generated/${folder}/${id}.png`;

    fs.writeFileSync(imagePath, pngBuffer);

    pageMeta[id] = { folder, query: trimmedLabel, mode, type: 'image', parentId: null };
    saveMetadata(pageMeta);

    console.log(`[upload] Saved: ${imagePath} (${pngBuffer.length} bytes)`);

    // Fire-and-forget classification. The user gets the page back immediately;
    // the classified payload becomes available shortly after via GET /api/classify/:pageId.
    enqueueClassify(id);

    res.json({
      page: {
        id,
        type: 'image',
        imageUrl,
        parentId: null,
        parentClick: null,
        initialQuery: trimmedLabel,
        mode,
      },
    });
  } catch (err) {
    console.error('Upload error:', err.message);
    res.status(500).json({ error: 'Upload failed: ' + err.message });
  }
});

async function generateUploadContextPage(uploadPageIdValue) {
  const meta = pageMeta[uploadPageIdValue];
  if (!meta) throw new Error('Upload page metadata not found');
  if (meta.type !== 'image') throw new Error('Upload context can only be generated from image pages');

  const contextId = uploadContextPageId(uploadPageIdValue);
  const cached = loadPageContent(meta.folder, contextId);
  if (cached) return cached;

  const imagePath = path.join(GENERATED_DIR, meta.folder, `${uploadPageIdValue}.png`);
  if (!fs.existsSync(imagePath)) throw new Error('Upload image file not found');

  const resized = await sharp(imagePath)
    .resize({ width: 1400, withoutEnlargement: true })
    .png()
    .toBuffer();
  const imageBase64 = resized.toString('base64');

  const systemPrompt = `You analyze uploaded educational images before learners drill into them. Perform OCR-style extraction of visible text first, then identify equations, chart structure, diagrams, and likely learning targets. Use Markdown. Use valid LaTeX with inline math in $...$ and display math in $$...$$. Be precise and mark uncertainty.`;
  const userPrompt = `Analyze this uploaded source image so future click-based drill-downs have context.

Return a concise structured Markdown page with these sections:

# Source Context

## Image Type
Classify the image, choosing from handwritten_math, printed_math, chart, graph, diagram, screenshot, photograph, mixed, or unknown. Briefly explain why.

## OCR / Transcription
Transcribe all visible text, labels, equations, axis names, legends, annotations, table cells, and headings. Preserve math as LaTeX where possible. If handwritten or unclear, say what is uncertain.

## Chart And Data Extraction
If the image includes a chart, graph, table, number line, axes, legend, or plotted marks, list the visible data structure, axis labels, units, categories, approximate values, and any uncertainty.

## Structure
Describe the spatial layout and important regions from top to bottom and left to right.

## Key Concepts
List the main concepts, variables, relationships, or data patterns the learner is likely working with.
`;

  const result = await analyzeImage(imageBase64, systemPrompt, userPrompt);
  const page = {
    id: contextId,
    type: 'markdown',
    title: 'Source context',
    content: result.text,
    source: result.source,
    parentId: uploadPageIdValue,
    parentClick: null,
    initialQuery: meta.query,
    mode: meta.mode,
    intent: 'Analyze uploaded source image',
    generatedAt: new Date().toISOString(),
  };

  savePageContent(meta.folder, contextId, page);
  pageMeta[contextId] = {
    folder: meta.folder,
    query: meta.query,
    mode: meta.mode,
    type: 'upload_context',
    parentId: uploadPageIdValue,
    parentClick: null,
    intent: page.intent,
  };
  pageMeta[uploadPageIdValue] = { ...meta, contextPageId: contextId };
  saveMetadata(pageMeta);
  console.log(`[context] Saved upload context: ${pageJsonPath(meta.folder, contextId)}`);
  return page;
}

app.get('/api/context/:pageId', (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }
  const meta = pageMeta[pageId];
  if (!meta) return res.status(404).json({ error: 'Page not found in metadata' });
  const contextId = meta.contextPageId || (meta.type === 'upload_context' ? pageId : null);
  const context = contextId ? loadPageContent(meta.folder, contextId) : findNearestUploadContext(pageId);
  res.json({ page: context });
});

app.post('/api/context/:pageId', async (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }

  try {
    const page = await enqueueGeneration(() => generateUploadContextPage(pageId));
    res.json({ page });
  } catch (err) {
    console.error('Context generation error:', err.message);
    res.status(500).json({ error: `Context generation failed: ${err.message}` });
  }
});

async function callTextChat(provider, model, systemPrompt, userPrompt) {
  enforceLocalOnly(provider, 'Text analysis');
  const providerConfig = MODEL_REGISTRY[provider];
  if (provider === 'gemini') {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: systemPrompt }] },
        contents: [{ parts: [{ text: userPrompt }] }],
        generationConfig: { temperature: 0.2, maxOutputTokens: 4096 },
      }),
      signal: AbortSignal.timeout(180000),
    });
    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Gemini error ${response.status}: ${err.slice(0, 200)}`);
    }
    const data = await response.json();
    const textParts = data.candidates?.[0]?.content?.parts?.filter(p => p.text) || [];
    return textParts.map(p => p.text).join('\n');
  }

  const chatUrl = resolveChatUrl(provider, model);
  if (!chatUrl) throw new Error(`No chat URL for provider: ${provider}`);
  const headers = { 'Content-Type': 'application/json' };
  const auth = providerConfig.authHeader();
  if (auth) headers['Authorization'] = auth;
  const response = await fetch(chatUrl, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      ...(provider !== 'local' ? { model } : {}),
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.2,
      max_tokens: 4096,
    }),
    signal: AbortSignal.timeout(180000),
  });
  if (!response.ok) {
    const err = await response.text();
    throw new Error(`${provider} error ${response.status}: ${err.slice(0, 200)}`);
  }
  const data = await response.json();
  return data.choices[0].message.content;
}

async function translatePageContent(page, language) {
  const cfg = modelConfig.analysis;
  const sourceName = cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})`;
  const systemPrompt = `Translate educational content accurately. Preserve Markdown structure, LaTeX, formulas, code, variable names, numeric values, and proper nouns unless a standard translation exists.`;
  const sourceText = (() => {
    if (page.type === 'chart') {
      return `Chart title: ${page.chart?.title || page.title}\nX label: ${page.chart?.xLabel || ''}\nY label: ${page.chart?.yLabel || ''}\nNotes:\n${(page.chart?.notes || []).map(n => `- ${n}`).join('\n')}\nUncertainty: ${page.chart?.uncertainty || ''}`;
    }
    if (page.type === 'table') {
      return `Table title: ${page.table?.title || page.title}\nColumns: ${(page.table?.columns || []).join(' | ')}\nRows:\n${(page.table?.rows || []).map(row => `- ${row.join(' | ')}`).join('\n')}\nNotes:\n${(page.table?.notes || []).map(n => `- ${n}`).join('\n')}\nUncertainty: ${page.table?.uncertainty || ''}`;
    }
    if (page.type === 'diagram') {
      return `Diagram title: ${page.diagram?.title || page.title}\nNodes:\n${(page.diagram?.nodes || []).map(n => `- ${n.label}${n.detail ? `: ${n.detail}` : ''}`).join('\n')}\nEdges:\n${(page.diagram?.edges || []).map(e => `- ${e.from} -> ${e.to}${e.label ? `: ${e.label}` : ''}`).join('\n')}\nNotes:\n${(page.diagram?.notes || []).map(n => `- ${n}`).join('\n')}\nUncertainty: ${page.diagram?.uncertainty || ''}`;
    }
    return page.content || '';
  })();
  const userPrompt = `Translate the following content into ${language}. Return Markdown only.\n\n${sourceText}`;
  try {
    const text = await callTextChat(cfg.provider, cfg.model, systemPrompt, userPrompt);
    return { text, source: sourceName };
  } catch (err) {
    if (!modelConfig.localOnly && cfg.provider === 'local' && err.message.includes('ECONNREFUSED')) {
      const text = await callTextChat('xai', 'grok-4-1-fast-non-reasoning', systemPrompt, userPrompt);
      return { text, source: 'grok (fallback)' };
    }
    throw err;
  }
}

app.post('/api/translate/:pageId', async (req, res) => {
  const { pageId } = req.params;
  const language = typeof req.body?.language === 'string' ? req.body.language.trim().slice(0, 80) : '';
  if (!(/^[a-f0-9]{16}$/.test(pageId))) return res.status(400).json({ error: 'Invalid page ID' });
  if (!language) return res.status(400).json({ error: 'Language is required' });
  const sourcePage = pageFromMetadata(pageId);
  const sourceMeta = pageMeta[pageId];
  if (!sourcePage || !sourceMeta) return res.status(404).json({ error: 'Page not found' });
  if (!['markdown', 'chart', 'table', 'diagram', 'upload_context'].includes(sourceMeta.type)) {
    return res.status(400).json({ error: 'Only text, chart, table, diagram, and context pages can be translated' });
  }

  try {
    const id = hash(`translate${CACHE_VERSION}${pageId}${normalize(language)}`);
    const cached = loadPageContent(sourceMeta.folder, id);
    if (cached) return res.json({ page: cached });
    const result = await translatePageContent(sourcePage, language);
    const page = {
      id,
      type: 'markdown',
      title: `${sourcePage.title || 'Page'} (${language})`,
      content: result.text,
      source: result.source,
      parentId: pageId,
      parentClick: null,
      initialQuery: sourceMeta.query,
      mode: sourceMeta.mode,
      intent: `Translate to ${language}`,
      responseDepth: 'translate',
      language,
    };
    savePageContent(sourceMeta.folder, id, page);
    pageMeta[id] = { folder: sourceMeta.folder, query: sourceMeta.query, mode: sourceMeta.mode, type: 'markdown', parentId: pageId, parentClick: null, intent: page.intent, language };
    saveMetadata(pageMeta);
    res.json({ page });
  } catch (err) {
    console.error('Translation error:', err.message);
    res.status(500).json({ error: `Translation failed: ${err.message}` });
  }
});

// --- Page generation API ---

function pageFromMetadata(pageId) {
  const meta = pageMeta[pageId];
  if (!meta) return null;

  if (meta.type === 'markdown' || meta.type === 'chart' || meta.type === 'table' || meta.type === 'diagram' || meta.type === 'upload_context') {
    const content = loadPageContent(meta.folder, pageId);
    if (!content) return null;
    return content;
  }

  const imagePath = path.join(GENERATED_DIR, meta.folder, `${pageId}.png`);
  if (!fs.existsSync(imagePath)) return null;
  // Include the slim subset of classified data the client needs for the
  // drill confirm bar (category + drill hints + style preset). The full
  // structured payload is still available via GET /api/classify/:pageId.
  // Returns null if the image hasn't been classified yet — client renders
  // its existing default UI in that case.
  const cached = loadAnalysis(pageId);
  const c = cached?.classified;
  const classifiedSlim = c && !c.fallback_used ? {
    category: c.category,
    confidence: c.category_confidence,
    style_preset: c.style?.preset || null,
    drill_hints: c.drill_hints || null,
  } : null;
  return {
    id: pageId,
    type: 'image',
    imageUrl: `/generated/${meta.folder}/${pageId}.png`,
    parentId: meta.parentId || null,
    parentClick: meta.parentClick || null,
    initialQuery: meta.query || null,
    mode: meta.mode || 'illustration',
    intent: meta.intent || '',
    contextPageId: meta.contextPageId || null,
    classified: classifiedSlim,
  };
}

app.get('/api/page/:pageId', (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }
  const page = pageFromMetadata(pageId);
  if (!page) return res.status(404).json({ error: 'Page not found' });
  res.json({ page });
});

// Return all descendants of a page (children, grandchildren, ...) so the
// frontend can rebuild drill-target overlays after a session restore or a
// single-page load from the gallery.
app.get('/api/page/:pageId/children', (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }
  if (!pageMeta[pageId]) {
    return res.status(404).json({ error: 'Page not found' });
  }
  const descendants = [];
  const queue = [pageId];
  const seen = new Set([pageId]);
  while (queue.length) {
    const current = queue.shift();
    for (const [id, meta] of Object.entries(pageMeta)) {
      if (meta.parentId === current && !seen.has(id)) {
        seen.add(id);
        const child = pageFromMetadata(id);
        if (child) {
          descendants.push(child);
          queue.push(id);
        }
      }
    }
  }
  res.json({ children: descendants });
});

app.post('/api/page', async (req, res) => {
  const { query, parentId, parentClick, mode: reqMode, intent: reqIntent, responseKind: reqResponseKind, responseDepth: reqResponseDepth, language: reqLanguage } = req.body;
  const responseKind = ['markdown', 'chart', 'table', 'diagram'].includes(reqResponseKind) ? reqResponseKind : 'image';
  const intent = typeof reqIntent === 'string' ? reqIntent.trim().slice(0, 500) : '';
  const responseDepth = ['extract', 'explain', 'teach'].includes(reqResponseDepth) ? reqResponseDepth : 'explain';
  const language = typeof reqLanguage === 'string' ? reqLanguage.trim().slice(0, 80) : '';

  const isFirst = typeof query === 'string';
  const isChild = typeof parentId === 'string' && parentClick && typeof parentClick.x === 'number' && typeof parentClick.y === 'number';

  if (!isFirst && !isChild) {
    return res.status(400).json({ error: 'Provide either { query } or { parentId, parentClick: { x, y } }' });
  }

  if (isFirst) {
    const trimmed = query.trim();
    if (trimmed.length < 1 || trimmed.length > 300) {
      return res.status(400).json({ error: 'Query must be 1-300 characters' });
    }
  }

  if (isChild) {
    if (!(/^[a-f0-9]{16}$/.test(parentId))) {
      return res.status(400).json({ error: 'Invalid parentId format' });
    }
    const { x, y } = parentClick;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > 1 || y < 0 || y > 1) {
      return res.status(400).json({ error: 'Coordinates must be finite floats in [0, 1]' });
    }
  }

  try {
    let mode = 'illustration';
    if (isFirst) {
      mode = (reqMode && reqMode !== 'auto' && VALID_MODES.includes(reqMode)) ? reqMode : inferModeFromQuery(query);
    } else {
      const parentMeta = pageMeta[parentId];
      mode = parentMeta?.mode || 'illustration';
    }

    const page = await enqueueGeneration(async () => {
      let id, parentPageId = null, parentClickData = null, initialQuery = null;
      let folder;

      if (isFirst) {
        id = firstPageId(query.trim(), mode);
        initialQuery = query.trim();
        folder = slugify(initialQuery);
        if (mode !== 'illustration') folder = `${mode}-${folder}`;
      } else {
        const rx = Math.round(parentClick.x * 100) / 100;
        const ry = Math.round(parentClick.y * 100) / 100;
        id = childPageId(parentId, rx, ry, responseKind, intent, responseDepth, language);
        parentPageId = parentId;
        parentClickData = { x: rx, y: ry };
        const parentMeta = pageMeta[parentId];
        if (!parentMeta) throw new Error('Parent page metadata not found');
        folder = parentMeta.folder;
      }

      const folderPath = path.join(GENERATED_DIR, folder);
      fs.mkdirSync(folderPath, { recursive: true });

      const imagePath = path.join(folderPath, `${id}.png`);
      const imageUrl = `/generated/${folder}/${id}.png`;

      if (fs.existsSync(imagePath) && fs.statSync(imagePath).size > 0) {
        return { id, type: 'image', imageUrl, parentId: parentPageId, parentClick: parentClickData, initialQuery, mode };
      }

      let imageBuffer;
      if (isFirst) {
        console.log(`[${mode}] Generating first page: "${initialQuery}" -> ${folder}/${id}`);
        imageBuffer = await generateFirstPage(initialQuery, mode);
      } else {
        let parentPath = path.join(folderPath, `${parentId}.png`);
        if (!fs.existsSync(parentPath)) parentPath = path.join(GENERATED_DIR, `${parentId}.png`);
        if (!fs.existsSync(parentPath)) throw new Error('Parent image not found');
        console.log(`[${mode}] Generating child page: ${parentId} @ (${parentClickData.x}, ${parentClickData.y}) -> ${folder}/${id}`);
        const composited = await compositeRedMarker(parentPath, parentClick.x, parentClick.y);
        const focusedComposite = responseKind === 'markdown' || responseKind === 'chart' || responseKind === 'table' || responseKind === 'diagram'
          ? await compositeFocusImage(parentPath, parentClick.x, parentClick.y)
          : null;
        const contextTrail = buildContextTrail(parentId);
        // Phase A.2: pull the parent's classified payload (if any) so generators
        // can match style and reference category-specific facts when generating
        // the drill child.
        const parentClassified = loadParentClassified(parentId);
        if (responseKind === 'markdown') {
          const textPath = pageJsonPath(folder, id);
          if (fs.existsSync(textPath)) {
            const cached = loadPageContent(folder, id);
            if (cached) return cached;
          }
          const result = await generateTextDrillPage(focusedComposite, mode, intent, contextTrail, responseDepth, language, parentClassified);
          const page = {
            id,
            type: 'markdown',
            title: intent || 'Drill-down explanation',
            content: result.text,
            source: result.source,
            parentId: parentPageId,
            parentClick: parentClickData,
            initialQuery,
            mode,
            intent,
            responseDepth,
            language,
          };
          savePageContent(folder, id, page);
          pageMeta[id] = { folder, query: pageMeta[parentId]?.query, mode, type: 'markdown', parentId, parentClick: parentClickData, intent, responseDepth, language };
          saveMetadata(pageMeta);
          console.log(`Saved: ${textPath}`);
          return page;
        }
        if (responseKind === 'chart') {
          const chartPath = pageJsonPath(folder, id);
          if (fs.existsSync(chartPath)) {
            const cached = loadPageContent(folder, id);
            if (cached) return cached;
          }
          const result = await generateChartDrillPage(focusedComposite, mode, intent, contextTrail, responseDepth, language, parentClassified);
          if (result.chart.chartability === 'low') {
            const page = {
              id,
              type: 'markdown',
              title: 'Chart not recommended',
              content: chartFallbackMarkdown(result.chart),
              source: result.source,
              parentId: parentPageId,
              parentClick: parentClickData,
              initialQuery,
              mode,
              intent,
              responseDepth,
              language,
              chart: result.chart,
            };
            savePageContent(folder, id, page);
            pageMeta[id] = { folder, query: pageMeta[parentId]?.query, mode, type: 'markdown', parentId, parentClick: parentClickData, intent, responseDepth, language };
            saveMetadata(pageMeta);
            console.log(`Saved: ${chartPath}`);
            return page;
          }
          const page = {
            id,
            type: 'chart',
            title: result.chart.title || intent || 'Chart drill-down',
            chart: result.chart,
            source: result.source,
            parentId: parentPageId,
            parentClick: parentClickData,
            initialQuery,
            mode,
            intent,
            responseDepth,
            language,
          };
          savePageContent(folder, id, page);
          pageMeta[id] = { folder, query: pageMeta[parentId]?.query, mode, type: 'chart', parentId, parentClick: parentClickData, intent, responseDepth, language };
          saveMetadata(pageMeta);
          console.log(`Saved: ${chartPath}`);
          return page;
        }
        if (responseKind === 'table') {
          const tablePath = pageJsonPath(folder, id);
          if (fs.existsSync(tablePath)) {
            const cached = loadPageContent(folder, id);
            if (cached) return cached;
          }
          const result = await generateTableDrillPage(focusedComposite, mode, intent, contextTrail, responseDepth, language, parentClassified);
          const page = {
            id,
            type: 'table',
            title: result.table.title || intent || 'Table drill-down',
            table: result.table,
            source: result.source,
            parentId: parentPageId,
            parentClick: parentClickData,
            initialQuery,
            mode,
            intent,
            responseDepth,
            language,
          };
          savePageContent(folder, id, page);
          pageMeta[id] = { folder, query: pageMeta[parentId]?.query, mode, type: 'table', parentId, parentClick: parentClickData, intent, responseDepth, language };
          saveMetadata(pageMeta);
          console.log(`Saved: ${tablePath}`);
          return page;
        }
        if (responseKind === 'diagram') {
          const diagramPath = pageJsonPath(folder, id);
          if (fs.existsSync(diagramPath)) {
            const cached = loadPageContent(folder, id);
            if (cached) return cached;
          }
          const result = await generateDiagramDrillPage(focusedComposite, mode, intent, contextTrail, responseDepth, language, parentClassified);
          const page = {
            id,
            type: 'diagram',
            title: result.diagram.title || intent || 'Diagram drill-down',
            diagram: result.diagram,
            source: result.source,
            parentId: parentPageId,
            parentClick: parentClickData,
            initialQuery,
            mode,
            intent,
            responseDepth,
            language,
          };
          savePageContent(folder, id, page);
          pageMeta[id] = { folder, query: pageMeta[parentId]?.query, mode, type: 'diagram', parentId, parentClick: parentClickData, intent, responseDepth, language };
          saveMetadata(pageMeta);
          console.log(`Saved: ${diagramPath}`);
          return page;
        }
        imageBuffer = await generateChildPage(composited, mode, intent, contextTrail, parentClassified);
      }

      fs.writeFileSync(imagePath, imageBuffer);
      pageMeta[id] = { folder, query: initialQuery || pageMeta[parentId]?.query, mode, type: 'image', parentId: parentPageId, parentClick: parentClickData, intent };
      saveMetadata(pageMeta);
      console.log(`Saved: ${imagePath} (${imageBuffer.length} bytes)`);

      // Phase A: every image (first-page or drill child) gets classified in the
      // background so the next drill from this page can reuse the classification.
      enqueueClassify(id);

      return { id, type: 'image', imageUrl, parentId: parentPageId, parentClick: parentClickData, initialQuery, mode };
    });

    res.json({ page });
  } catch (err) {
    console.error('Generation error:', err.message);
    const message = err.message?.includes('Keep it local is enabled') || err.message?.includes('No local image')
      ? err.message
      : 'Generation failed. Try again or click elsewhere.';
    res.status(500).json({ error: message });
  }
});

// --- Image Analysis API (local Gemma via llama-server) ---

fs.mkdirSync(ANALYSIS_DIR, { recursive: true });

function analysisPath(pageId) {
  return path.join(ANALYSIS_DIR, `${pageId}.json`);
}

function loadAnalysis(pageId) {
  try { return JSON.parse(fs.readFileSync(analysisPath(pageId), 'utf8')); }
  catch { return null; }
}

function saveAnalysis(pageId, data) {
  fs.writeFileSync(analysisPath(pageId), JSON.stringify(data, null, 2));
}

async function callVisionChat(provider, model, imageBase64, systemPrompt, userPrompt) {
  enforceLocalOnly(provider, 'Vision analysis');
  const providerConfig = MODEL_REGISTRY[provider];

  // Gemini uses its own API format
  if (provider === 'gemini') {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: systemPrompt }] },
        contents: [{
          parts: [
            { inline_data: { mime_type: 'image/png', data: imageBase64 } },
            { text: userPrompt },
          ],
        }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 4096 },
      }),
      signal: AbortSignal.timeout(180000),
    });
    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Gemini error ${response.status}: ${err.slice(0, 200)}`);
    }
    const data = await response.json();
    const textParts = data.candidates?.[0]?.content?.parts?.filter(p => p.text) || [];
    return textParts.map(p => p.text).join('\n');
  }

  // OpenAI / xAI / local — all use OpenAI-compatible chat format
  const chatUrl = resolveChatUrl(provider, model);
  if (!chatUrl) throw new Error(`No chat URL for provider: ${provider}`);

  const headers = { 'Content-Type': 'application/json' };
  const auth = providerConfig.authHeader();
  if (auth) headers['Authorization'] = auth;

  const response = await fetch(chatUrl, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      ...(provider !== 'local' ? { model } : {}),
      messages: [
        { role: 'system', content: systemPrompt },
        {
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: `data:image/png;base64,${imageBase64}` } },
            { type: 'text', text: userPrompt },
          ],
        },
      ],
      temperature: 0.3,
      max_tokens: 4096,
    }),
    signal: AbortSignal.timeout(180000),
  });

  if (!response.ok) {
    const err = await response.text();
    if (err.includes('mmproj') || err.includes('image input is not supported')) {
      throw new Error('VISION_NOT_SUPPORTED');
    }
    throw new Error(`${provider} error ${response.status}: ${err.slice(0, 200)}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

async function analyzeImage(imageBase64, systemPrompt, userPrompt) {
  const cfg = modelConfig.analysis;
  const sourceName = cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})`;

  try {
    const text = await callVisionChat(cfg.provider, cfg.model, imageBase64, systemPrompt, userPrompt);
    return { text, source: sourceName };
  } catch (err) {
    if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
      // Fallback to Grok
      console.log(`[analysis] Local model unavailable (${err.message}), falling back to Grok...`);
      const text = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
      return { text, source: 'grok (fallback)' };
    }
    throw err;
  }
}

// Classification-specific vision call. Distinct from analyzeImage because:
// 1. Uses modelConfig.classify (separately pinnable model — defaults to local Gemma).
// 2. Cloud fallback is gated by the explicit modelConfig.allowClassifyCloudFallback
//    flag, NOT by modelConfig.localOnly. This keeps classification local-by-default
//    even when the user has enabled cloud fallback for analysis.
async function analyzeImageForClassify(systemPrompt, userPrompt, imageBase64) {
  const cfg = modelConfig.classify || { provider: 'local', model: 'auto' };
  const sourceName = cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})`;
  try {
    const text = await callVisionChat(cfg.provider, cfg.model, imageBase64, systemPrompt, userPrompt);
    return { text, source: sourceName };
  } catch (err) {
    const isLocalOutage = cfg.provider === 'local'
      && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'));
    if (isLocalOutage && modelConfig.allowClassifyCloudFallback === true) {
      console.log(`[classify] Local model unavailable (${err.message}); cloud fallback authorized — calling Grok.`);
      const text = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
      return { text, source: 'grok (fallback)' };
    }
    throw err;
  }
}

// Background classify runner. Fire-and-forget from upload / generation paths;
// errors are logged but never block the caller. Result lands in
// _analysis/<pageId>.json under the .classified field, alongside the existing
// description/explanation fields (which are unaffected).
const classifyInFlight = new Map(); // pageId -> Promise (deduplicates concurrent triggers)

async function runClassifyForPage(pageId) {
  const meta = pageMeta[pageId];
  if (!meta || meta.type !== 'image') return null;
  const imagePath = path.join(GENERATED_DIR, meta.folder, `${pageId}.png`);
  if (!fs.existsSync(imagePath)) {
    console.warn(`[classify] ${pageId}: image file not found at ${imagePath}`);
    return null;
  }
  const buf = fs.readFileSync(imagePath);
  const imageBase64 = buf.toString('base64');
  const modelName = (modelConfig.classify?.model && modelConfig.classify.model !== 'auto')
    ? modelConfig.classify.model
    : 'gemma-4-E4B';

  const started = Date.now();
  const payload = await classifyImage(imageBase64, analyzeImageForClassify, modelName);
  const elapsed = Date.now() - started;

  const existing = loadAnalysis(pageId) || {};
  existing.classified = payload;
  existing.classifiedAt = payload.classified_at;
  saveAnalysis(pageId, existing);

  if (payload.fallback_used) {
    console.warn(`[classify] ${pageId}: ${elapsed}ms — FALLBACK (${payload.fallback_reason})`);
    // Surface the most common configuration issue with a concrete remediation hint.
    if (payload.fallback_reason && payload.fallback_reason.includes('exceed_context_size_error')) {
      console.warn('[classify] >>> Local llama-server slot is too small for the classify prompt.');
      console.warn('[classify] >>> Suggested fix: restart llama-server with reduced concurrency,');
      console.warn('[classify] >>>   e.g. -c 32768 -np 4   (gives 8192 tokens per slot)');
      console.warn('[classify] >>>   or   -c 65536 -np 8   (gives 8192 tokens per slot)');
    }
  } else {
    console.log(`[classify] ${pageId}: ${elapsed}ms — ${payload.category} (conf ${payload.category_confidence.toFixed(2)})`);
  }
  return payload;
}

function enqueueClassify(pageId) {
  if (classifyInFlight.has(pageId)) return classifyInFlight.get(pageId);
  const p = runClassifyForPage(pageId)
    .catch(err => {
      console.error(`[classify] ${pageId} failed:`, err.message);
      return null;
    })
    .finally(() => classifyInFlight.delete(pageId));
  classifyInFlight.set(pageId, p);
  return p;
}

const DESCRIPTION_SYSTEM = `You are an image analyst. Describe images thoroughly and accurately. Include ALL visible text exactly as it appears.`;

const DESCRIPTION_PROMPT = `Analyze this image and provide a comprehensive description.

Structure your response in these sections:

## Visual Description
Describe what the image shows — the scene, layout, objects, colors, style, and composition. Be specific about spatial relationships (what is where).

## Text Content
List ALL text visible in the image, exactly as written. Include titles, labels, annotations, legends, dates, and any other text. Preserve the original formatting and grouping.

## Image Type & Style
What kind of image is this? (illustration, map, diagram, photograph, etc.) Describe the artistic style, color palette, and visual technique used.`;

const EXPLANATION_SYSTEM = `You are an expert educator who explains complex concepts clearly. When analyzing educational illustrations or maps, break down every component for a learner who is encountering this subject for the first time.`;

const EXPLANATION_PROMPT = `This image is from an educational explainer. Analyze it for a learner.

Structure your response in these sections:

## Overview
What is this image teaching? Summarize the main concept or topic in 2-3 sentences.

## Key Components
For EACH labeled element, object, region, or notable feature in the image:
- **[Name/Label]**: What it is, what it does, and why it matters in this context.

Be thorough — cover every component you can identify, even small details.

## How It Works
Explain the process, relationship, or story the image is showing. How do the components connect? What causes what? What sequence or flow is being depicted?

## Context & Significance
Why does this matter? What broader concepts does this connect to? What would a student want to explore next?

## Key Takeaways
3-5 bullet points summarizing the most important things a learner should remember from this image.`;

app.get('/api/analysis/:pageId', (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }
  const cached = loadAnalysis(pageId);
  if (cached) return res.json(cached);
  res.json({ description: null, explanation: null });
});

// --- Classify (Phase A) ---
// GET returns the cached classified payload (or null + a status flag if a
// classify pass is currently running in the background).
// POST forces a re-classify even if one is cached.
app.get('/api/classify/:pageId', (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) return res.status(400).json({ error: 'Invalid page ID' });
  const cached = loadAnalysis(pageId);
  res.json({
    classified: cached?.classified || null,
    inFlight: classifyInFlight.has(pageId),
  });
});

app.post('/api/classify/:pageId', async (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) return res.status(400).json({ error: 'Invalid page ID' });
  const meta = pageMeta[pageId];
  if (!meta) return res.status(404).json({ error: 'Page not found' });
  if (meta.type !== 'image') return res.status(400).json({ error: 'Only image pages can be classified' });
  const force = req.body?.force === true;
  if (!force) {
    const cached = loadAnalysis(pageId);
    if (cached?.classified) return res.json({ classified: cached.classified, cached: true });
  }
  try {
    const payload = await enqueueClassify(pageId);
    if (!payload) return res.status(500).json({ error: 'Classification produced no result (see server logs).' });
    res.json({ classified: payload, cached: false });
  } catch (err) {
    res.status(500).json({ error: `Classification failed: ${err.message}` });
  }
});

app.post('/api/analysis/:pageId', async (req, res) => {
  const { pageId } = req.params;
  const { type } = req.body; // 'description', 'explanation', or 'both'

  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }

  // Find the image file
  const meta = pageMeta[pageId];
  if (!meta) return res.status(404).json({ error: 'Page not found in metadata' });

  const imagePath = path.join(GENERATED_DIR, meta.folder, `${pageId}.png`);
  if (!fs.existsSync(imagePath)) {
    return res.status(404).json({ error: 'Image file not found' });
  }

  // Load existing analysis (may have one half already)
  const existing = loadAnalysis(pageId) || { description: null, explanation: null };

  try {
    // Resize image for the vision model (keep it reasonable for local inference)
    const resized = await sharp(imagePath)
      .resize({ width: 1024, withoutEnlargement: true })
      .png()
      .toBuffer();
    const imageBase64 = resized.toString('base64');

    const runDescription = (type === 'description' || type === 'both') && !existing.description;
    const runExplanation = (type === 'explanation' || type === 'both') && !existing.explanation;

    if (runDescription) {
      console.log(`[analysis] Generating description for ${pageId}...`);
      const result = await analyzeImage(imageBase64, DESCRIPTION_SYSTEM, DESCRIPTION_PROMPT);
      existing.description = result.text;
      existing.descriptionAt = new Date().toISOString();
      existing.descriptionSource = result.source;
      saveAnalysis(pageId, existing);
    }

    if (runExplanation) {
      console.log(`[analysis] Generating explanation for ${pageId}...`);
      const result = await analyzeImage(imageBase64, EXPLANATION_SYSTEM, EXPLANATION_PROMPT);
      existing.explanation = result.text;
      existing.explanationAt = new Date().toISOString();
      existing.explanationSource = result.source;
      saveAnalysis(pageId, existing);
    }

    res.json(existing);
  } catch (err) {
    console.error('Analysis error:', err.message);
    // Save partial results even on error
    saveAnalysis(pageId, existing);
    res.status(500).json({ error: `Analysis failed: ${err.message}`, ...existing });
  }
});

// --- Gallery API ---

app.get('/api/gallery', (_req, res) => {
  const folders = [];
  try {
    const entries = fs.readdirSync(GENERATED_DIR, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      if (entry.name.startsWith('_')) continue;
      const folderPath = path.join(GENERATED_DIR, entry.name);
      const images = fs.readdirSync(folderPath)
        .filter(f => f.endsWith('.png'))
        .map(f => {
          const stat = fs.statSync(path.join(folderPath, f));
          const pageId = f.replace('.png', '');
          const meta = pageMeta[pageId] || {};
          return { id: pageId, filename: f, url: `/generated/${entry.name}/${f}`, size: stat.size, created: stat.birthtime, mode: meta.mode || 'illustration' };
        })
        .sort((a, b) => new Date(a.created) - new Date(b.created));
      const texts = fs.readdirSync(folderPath)
        .filter(f => f.endsWith('.json'))
        .map(f => {
          const stat = fs.statSync(path.join(folderPath, f));
          const pageId = f.replace('.json', '');
          const meta = pageMeta[pageId] || {};
          const content = loadPageContent(entry.name, pageId) || {};
          return {
            id: pageId,
            filename: f,
            title: content.title || meta.intent || 'Text drill-down',
            size: stat.size,
            created: stat.birthtime,
            mode: meta.mode || 'illustration',
          };
        })
        .sort((a, b) => new Date(a.created) - new Date(b.created));

      const firstMeta = Object.values(pageMeta).find(m => m.folder === entry.name);
      folders.push({
        name: entry.name, query: firstMeta?.query || entry.name, mode: firstMeta?.mode || 'illustration',
        path: folderPath,
        images,
        texts,
        imageCount: images.length,
        textCount: texts.length,
        totalSize: images.reduce((sum, img) => sum + img.size, 0) + texts.reduce((sum, text) => sum + text.size, 0),
      });
    }
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
  folders.sort((a, b) => {
    const aCreated = [...a.images, ...a.texts].sort((x, y) => new Date(y.created) - new Date(x.created))[0]?.created || 0;
    const bCreated = [...b.images, ...b.texts].sort((x, y) => new Date(y.created) - new Date(x.created))[0]?.created || 0;
    return new Date(bCreated) - new Date(aCreated);
  });
  res.json({ generatedDir: GENERATED_DIR, folders });
});

app.listen(PORT, () => {
  console.log(`Illustrated Explainer running at http://localhost:${PORT}`);
});

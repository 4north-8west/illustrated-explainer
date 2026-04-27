import dotenv from 'dotenv';
dotenv.config({ override: true });
import express from 'express';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

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
  console.error('XAI_API_KEY is required in .env');
  process.exit(1);
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
  generation: { provider: 'xai', model: 'grok-imagine-image' },
  editing: { provider: 'xai', model: 'grok-imagine-image' },
  analysis: { provider: 'local', model: 'auto' },
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
    name: 'Local (llama-server)',
    models: {
      'auto': { name: 'Auto-detect (whatever is running)', capabilities: ['analysis'] },
      'gemma-4-E2B': { name: 'Gemma 4 E2B (Tiny, 2B)', capabilities: ['analysis'] },
      'gemma-4-E4B': { name: 'Gemma 4 E4B (Small, 4B)', capabilities: ['analysis'] },
      'gemma-4-26B': { name: 'Gemma 4 26B (Medium, MoE)', capabilities: ['analysis'] },
    },
    chatUrl: LLAMA_SERVER_URL + '/v1/chat/completions',
    authHeader: () => null,
  },
};

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

async function generateChildPage(compositedImageBuffer, mode, intent = '', contextTrail = []) {
  const modeConfig = MODES[mode];
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const intentPrompt = intent
    ? `\n\nLearner intent for this drill-down: ${intent}\nUse that intent to decide what details to reveal while still focusing on the marked region.`
    : '';
  const sourcePrompt = uploadContext?.content
    ? `\n\nSource upload analysis for context:\n${uploadContext.content.slice(0, 2500)}`
    : '';
  const prompt = renderTemplate(modeConfig.childPageTemplate, { style: modeConfig.style }) + intentPrompt + sourcePrompt;
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

async function generateTextDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '') {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContext?.content
    ? `\n\nSource upload analysis:\n${uploadContext.content.slice(0, 5000)}`
    : '';

  const systemPrompt = `You are an expert STEM-capable educator. Explain only the exact selected region in an image, not the image as a whole. Prefer precise text, equations, tables, or concise examples over generating another image. Use Markdown. If math is needed, use valid LaTeX with inline math in $...$ and display math in $$...$$.`;
  const userPrompt = `The image has two panels. The left panel shows the full page with the click marked. The right panel is a zoomed crop centered on the clicked location and labeled "ZOOMED CLICK REGION - ANSWER ABOUT THIS".

Focus your answer on the feature at the center of the right zoomed panel. Use the full left panel only for context.

The learner clicked this region in a ${modeLabel}.

Learner intent: ${intent || 'Explain what is marked and why it matters.'}

${responseDepthInstruction(responseDepth, language)}

Drill-down context:
${contextText || 'No prior context available.'}${uploadContextText}

Respond with a focused Markdown learning page:
- Start with a short title.
- Identify what the center of the right zoomed region appears to be.
- Answer the learner intent directly.
- Include equations, definitions, or small tables when they help.
- Put important equations on their own lines using $$...$$.
- Mention uncertainty explicitly if the image does not provide enough information.
- Do not explain unrelated areas of the image.`;

  const cfg = modelConfig.analysis;
  try {
    const text = await callVisionChat(cfg.provider, cfg.model, imageBase64, systemPrompt, userPrompt);
    return { text, source: cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})` };
  } catch (err) {
    if (cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
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
  const points = Array.isArray(chart.points) ? chart.points
    .map((point, index) => ({
      label: String(point.label ?? point.x ?? index + 1),
      x: Number.isFinite(Number(point.x)) ? Number(point.x) : index,
      y: Number.isFinite(Number(point.y)) ? Number(point.y) : 0,
    }))
    .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y))
    .slice(0, 80) : [];

  return {
    type,
    title: String(chart.title || 'Chart drill-down').slice(0, 120),
    xLabel: String(chart.xLabel || 'x').slice(0, 80),
    yLabel: String(chart.yLabel || 'y').slice(0, 80),
    points,
    notes: Array.isArray(chart.notes) ? chart.notes.map(note => String(note).slice(0, 240)).slice(0, 6) : [],
    uncertainty: String(chart.uncertainty || '').slice(0, 500),
  };
}

async function generateChartDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '') {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContext?.content
    ? `\n\nSource upload analysis:\n${uploadContext.content.slice(0, 5000)}`
    : '';

  const systemPrompt = `You convert only the exact selected image region into a simple educational chart specification. Return JSON only. Use inferred or approximate values only when the selected region and context support them, and describe uncertainty.`;
  const userPrompt = `The image has two panels. The left panel shows the full page with the click marked. The right panel is a zoomed crop centered on the clicked location and labeled "ZOOMED CLICK REGION - ANSWER ABOUT THIS".

Focus on the feature at the center of the right zoomed panel. Use the full left panel only for context.

The learner clicked this region in a ${modeLabel}.

Learner intent: ${intent || 'Turn the marked region into a useful chart or graph if appropriate.'}

${responseDepthInstruction(responseDepth, language)}

Drill-down context:
${contextText || 'No prior context available.'}${uploadContextText}

Return one JSON object only, with this shape:
{
  "type": "line" | "bar" | "scatter",
  "title": "short chart title",
  "xLabel": "x axis label",
  "yLabel": "y axis label",
  "points": [
    { "label": "visible label", "x": 0, "y": 0 }
  ],
  "notes": ["short teaching note"],
  "uncertainty": "what is inferred or approximate"
}

Rules:
- Use 2-20 points unless the image clearly supports more.
- If the marked region is not numeric data, make a conceptual sequence chart using ordinal x values.
- Do not invent precision. Prefer simple approximate values and clear uncertainty.`;

  const cfg = modelConfig.analysis;
  let result;
  try {
    result = await callVisionChat(cfg.provider, cfg.model, imageBase64, systemPrompt, userPrompt);
  } catch (err) {
    if (cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
      result = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
      const chart = normalizeChartSpec(parseJsonResponse(result));
      return { chart, source: 'grok (fallback)' };
    }
    throw err;
  }

  const chart = normalizeChartSpec(parseJsonResponse(result));
  return { chart, source: cfg.provider === 'local' ? `local (${cfg.model})` : `${cfg.provider} (${cfg.model})` };
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
  const { generation, editing, analysis, localOnly } = req.body;
  if (typeof localOnly === 'boolean') modelConfig.localOnly = localOnly;
  for (const [key, val] of [['generation', generation], ['editing', editing], ['analysis', analysis]]) {
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

  const systemPrompt = `You analyze uploaded educational images before learners drill into them. Identify text, equations, chart structure, diagrams, and likely learning targets. Use Markdown. Use valid LaTeX with inline math in $...$ and display math in $$...$$. Be precise and mark uncertainty.`;
  const userPrompt = `Analyze this uploaded source image so future click-based drill-downs have context.

Return a concise structured Markdown page with these sections:

# Source Context

## Image Type
Classify the image, choosing from handwritten_math, printed_math, chart, graph, diagram, screenshot, photograph, mixed, or unknown. Briefly explain why.

## Transcription
Transcribe all visible text, labels, equations, axis names, legends, annotations, and headings. Preserve math as LaTeX where possible. If handwritten or unclear, say what is uncertain.

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

  if (!providerConfig?.chatUrl) throw new Error(`No chat URL for provider: ${provider}`);
  const headers = { 'Content-Type': 'application/json' };
  const auth = providerConfig.authHeader();
  if (auth) headers['Authorization'] = auth;
  const response = await fetch(providerConfig.chatUrl, {
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
  const chartText = page.type === 'chart'
    ? `Chart title: ${page.chart?.title || page.title}\nX label: ${page.chart?.xLabel || ''}\nY label: ${page.chart?.yLabel || ''}\nNotes:\n${(page.chart?.notes || []).map(n => `- ${n}`).join('\n')}\nUncertainty: ${page.chart?.uncertainty || ''}`
    : page.content || '';
  const userPrompt = `Translate the following content into ${language}. Return Markdown only.\n\n${chartText}`;
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
  if (!['markdown', 'chart', 'upload_context'].includes(sourceMeta.type)) {
    return res.status(400).json({ error: 'Only text, chart, and context pages can be translated' });
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

  if (meta.type === 'markdown' || meta.type === 'chart' || meta.type === 'upload_context') {
    const content = loadPageContent(meta.folder, pageId);
    if (!content) return null;
    return content;
  }

  const imagePath = path.join(GENERATED_DIR, meta.folder, `${pageId}.png`);
  if (!fs.existsSync(imagePath)) return null;
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

app.post('/api/page', async (req, res) => {
  const { query, parentId, parentClick, mode: reqMode, intent: reqIntent, responseKind: reqResponseKind, responseDepth: reqResponseDepth, language: reqLanguage } = req.body;
  const responseKind = ['markdown', 'chart'].includes(reqResponseKind) ? reqResponseKind : 'image';
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
      mode = (reqMode && VALID_MODES.includes(reqMode)) ? reqMode : 'illustration';
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
        const focusedComposite = responseKind === 'markdown' || responseKind === 'chart'
          ? await compositeFocusImage(parentPath, parentClick.x, parentClick.y)
          : null;
        const contextTrail = buildContextTrail(parentId);
        if (responseKind === 'markdown') {
          const textPath = pageJsonPath(folder, id);
          if (fs.existsSync(textPath)) {
            const cached = loadPageContent(folder, id);
            if (cached) return cached;
          }
          const result = await generateTextDrillPage(focusedComposite, mode, intent, contextTrail, responseDepth, language);
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
          const result = await generateChartDrillPage(focusedComposite, mode, intent, contextTrail, responseDepth, language);
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
        imageBuffer = await generateChildPage(composited, mode, intent, contextTrail);
      }

      fs.writeFileSync(imagePath, imageBuffer);
      pageMeta[id] = { folder, query: initialQuery || pageMeta[parentId]?.query, mode, type: 'image', parentId: parentPageId, parentClick: parentClickData, intent };
      saveMetadata(pageMeta);
      console.log(`Saved: ${imagePath} (${imageBuffer.length} bytes)`);

      return { id, type: 'image', imageUrl, parentId: parentPageId, parentClick: parentClickData, initialQuery, mode };
    });

    res.json({ page });
  } catch (err) {
    console.error('Generation error:', err.message);
    res.status(500).json({ error: 'Generation failed. Try again or click elsewhere.' });
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
  if (!providerConfig?.chatUrl) throw new Error(`No chat URL for provider: ${provider}`);

  const headers = { 'Content-Type': 'application/json' };
  const auth = providerConfig.authHeader();
  if (auth) headers['Authorization'] = auth;

  const response = await fetch(providerConfig.chatUrl, {
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

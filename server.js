import { spawn, execFileSync } from 'node:child_process';
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

async function generateChildPage(compositedImageBuffer, mode, intent = '', contextTrail = []) {
  const modeConfig = MODES[mode];
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const intentPrompt = intent
    ? `\n\nLearner intent for this drill-down: ${intent}\nUse that intent to decide what details to reveal while still focusing on the marked region.`
    : '';
  const sourcePrompt = uploadContextPrompt(uploadContext, 2500);
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

function uploadContextPrompt(uploadContext, maxChars = 5000) {
  if (!uploadContext?.content) return '';
  return `\n\nSource upload OCR/transcription and analysis:
${uploadContext.content.slice(0, maxChars)}

Use this source context as primary evidence for uploaded images, especially visible text, labels, equations, axis names, legends, and table/chart structure. If the focused crop is ambiguous, prefer values and labels supported by this context and explicitly mark uncertainty.`;
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
  const uploadContextText = uploadContextPrompt(uploadContext);

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

async function generateChartDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '') {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContextPrompt(uploadContext);

  const systemPrompt = `You convert only the exact selected image region into a simple educational chart specification. Return JSON only. Use inferred or approximate values only when the selected region, OCR/transcription context, and surrounding page context support them. Describe uncertainty and cite evidence for every plotted point.`;
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

  const cfg = modelConfig.analysis;
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

async function generateTableDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '') {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContextPrompt(uploadContext);

  const systemPrompt = `You convert only the exact selected image region into a locally renderable educational table. Return JSON only. Use uploaded OCR/transcription context as primary evidence for visible text, labels, equations, and values. Preserve visible wording when extraction is requested.`;
  const userPrompt = `The image has two panels. The left panel shows the full page with the click marked. The right panel is a zoomed crop centered on the clicked location and labeled "ZOOMED CLICK REGION - ANSWER ABOUT THIS".

Focus on the feature at the center of the right zoomed panel. Use the full left panel only for context.

The learner clicked this region in a ${modeLabel}.

Learner intent: ${intent || 'Organize the marked region as a useful table.'}

${responseDepthInstruction(responseDepth, language)}

Drill-down context:
${contextText || 'No prior context available.'}${uploadContextText}

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

  const cfg = modelConfig.analysis;
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

async function generateDiagramDrillPage(compositedImageBuffer, mode, intent, contextTrail, responseDepth = 'explain', language = '') {
  const imageBase64 = compositedImageBuffer.toString('base64');
  const modeLabel = MODES[mode]?.modeLabelForPrompt || mode.replace(/_/g, ' ');
  const uploadContext = contextTrail.length ? findNearestUploadContext(contextTrail[contextTrail.length - 1].id) : null;
  const contextText = contextTrail.map((item, index) => {
    const step = index + 1;
    const intentText = item.intent ? ` Intent: ${item.intent}` : '';
    return `${step}. ${item.type} page about "${item.query}" (${item.mode}).${intentText}`;
  }).join('\n');
  const uploadContextText = uploadContextPrompt(uploadContext);

  const systemPrompt = `You convert only the exact selected image region into a locally renderable educational diagram specification. Return JSON only. Use uploaded OCR/transcription context as primary evidence for visible text, labels, equations, and relationships. The app will render the diagram itself as SVG; do not request image generation.`;
  const userPrompt = `The image has two panels. The left panel shows the full page with the click marked. The right panel is a zoomed crop centered on the clicked location and labeled "ZOOMED CLICK REGION - ANSWER ABOUT THIS".

Focus on the feature at the center of the right zoomed panel. Use the full left panel only for context.

The learner clicked this region in a ${modeLabel}.

Learner intent: ${intent || 'Turn the marked region into a concept or process diagram.'}

${responseDepthInstruction(responseDepth, language)}

Drill-down context:
${contextText || 'No prior context available.'}${uploadContextText}

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

  const cfg = modelConfig.analysis;
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

// --- PDF upload (Phase 1 MVP — tag detection + render-as-images) ---
// Detects whether the PDF is tagged. Renders every page to PNG via poppler's
// pdftoppm and registers each as a regular image-type page so the existing
// drill-in / analysis / entities pipelines work unchanged. Tagged-tree text
// extraction (Phase 2) is a follow-up — for now we record `pdfTagged` in
// metadata so the future flow can route from it.
function detectPdfMetadata(pdfPath) {
  let stdout = '';
  try {
    stdout = execFileSync('pdfinfo', [pdfPath], { encoding: 'utf8', timeout: 15000 });
  } catch (err) {
    throw new Error(`pdfinfo failed: ${err.message} (is poppler installed? brew install poppler)`);
  }
  const lines = stdout.split('\n');
  let pages = 0;
  let tagged = false;
  let title = '';
  for (const line of lines) {
    const m = line.match(/^([^:]+):\s*(.+)$/);
    if (!m) continue;
    const key = m[1].trim().toLowerCase();
    const val = m[2].trim();
    if (key === 'pages') pages = parseInt(val, 10) || 0;
    else if (key === 'tagged') tagged = /yes/i.test(val);
    else if (key === 'title') title = val;
  }
  return { pages, tagged, title };
}

// docling whole-document extraction. Returns the Markdown string on success,
// or null if docling is unavailable or fails. Used for high-quality structured
// extraction (preserves headings/lists/tables) before we fall back to per-page
// pdftotext concatenation.
function extractPdfWithDocling(pdfPath, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  try {
    execFileSync('docling', ['--to', 'md', '--output', outDir, pdfPath], {
      timeout: 600000,
      stdio: ['ignore', 'ignore', 'pipe'],
    });
  } catch (err) {
    console.warn(`[docling] failed: ${err.message}`);
    return null;
  }
  // docling writes <pdf-basename>.md alongside our outDir
  const base = path.basename(pdfPath, path.extname(pdfPath));
  const candidates = [
    path.join(outDir, `${base}.md`),
    ...fs.readdirSync(outDir).filter(f => f.endsWith('.md')).map(f => path.join(outDir, f)),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return fs.readFileSync(c, 'utf8');
  }
  return null;
}

function extractPdfPageText(pdfPath, pageNumber) {
  try {
    return execFileSync(
      'pdftotext',
      ['-layout', '-enc', 'UTF-8', '-f', String(pageNumber), '-l', String(pageNumber), pdfPath, '-'],
      { encoding: 'utf8', timeout: 15000, maxBuffer: 4 * 1024 * 1024 }
    );
  } catch (err) {
    console.warn(`[upload-pdf] pdftotext failed on page ${pageNumber}: ${err.message}`);
    return '';
  }
}

function synthesizePdfPageContext({ text, pageNumber, pageCount, tagged, title, label }) {
  const trimmed = text.trim();
  const taggedLine = tagged ? 'tagged' : 'untagged';
  const titleLine = title ? `**Document title**: ${title}\n` : '';
  return `# Source Context

## Image Type
PDF page (text extracted directly via \`pdftotext -layout\` — no vision model call). ${titleLine}**PDF page**: ${pageNumber} of ${pageCount}; **status**: ${taggedLine}; **upload label**: ${label}.

## OCR / Transcription
The following text was extracted from the PDF page in reading order. For tagged PDFs this comes from the structure tree; for untagged-but-text-bearing PDFs it is pdftotext's layout-aware approximation.

\`\`\`
${trimmed}
\`\`\`

## Structure
This context was generated without a vision call, so it does not yet describe spatial layout, figures, or visual emphasis. Open the **Learn** panel on this page to add a vision-derived description that complements the extracted text above.

## Key Concepts
Open the **Entities** panel to extract a structured list of components, concepts, flow steps, and other typed entities from the page text.
`;
}

function renderPdfPage(pdfPath, pageNumber, outPrefix, dpi = 150) {
  const args = [
    '-r', String(dpi),
    '-f', String(pageNumber),
    '-l', String(pageNumber),
    '-png',
    '-singlefile',
    pdfPath,
    outPrefix,
  ];
  try {
    execFileSync('pdftoppm', args, { timeout: 60000, stdio: ['ignore', 'ignore', 'pipe'] });
  } catch (err) {
    throw new Error(`pdftoppm failed on page ${pageNumber}: ${err.message}`);
  }
  const outPath = outPrefix + '.png';
  if (!fs.existsSync(outPath)) {
    throw new Error(`pdftoppm produced no output for page ${pageNumber}`);
  }
  return outPath;
}

app.post('/api/upload-pdf', async (req, res) => {
  const { pdfData, label, mode: reqMode, dpi: reqDpi } = req.body;
  if (!pdfData || !label) {
    return res.status(400).json({ error: 'pdfData (base64) and label are required' });
  }
  const trimmedLabel = String(label).trim();
  if (trimmedLabel.length < 1 || trimmedLabel.length > 300) {
    return res.status(400).json({ error: 'Label must be 1–300 characters' });
  }
  const mode = (reqMode && VALID_MODES.includes(reqMode)) ? reqMode : 'illustration';
  const dpi = Math.max(72, Math.min(300, Number(reqDpi) || 150));

  // Decode base64 (strip data URI prefix if present)
  const base64Clean = String(pdfData).replace(/^data:application\/pdf;base64,/, '').replace(/^data:[^;]+;base64,/, '');
  let pdfBuffer;
  try {
    pdfBuffer = Buffer.from(base64Clean, 'base64');
  } catch (err) {
    return res.status(400).json({ error: 'pdfData is not valid base64' });
  }
  if (pdfBuffer.length < 100 || pdfBuffer.slice(0, 5).toString() !== '%PDF-') {
    return res.status(400).json({ error: 'Uploaded file is not a valid PDF' });
  }

  const folder = `upload-pdf-${slugify(trimmedLabel)}`;
  const folderPath = path.join(GENERATED_DIR, folder);
  fs.mkdirSync(folderPath, { recursive: true });

  const pdfHashHex = crypto.createHash('sha256').update(pdfBuffer).digest('hex');
  const pdfDocId = pdfHashHex.slice(0, 16);
  const pdfPath = path.join(folderPath, `${pdfDocId}.pdf`);
  fs.writeFileSync(pdfPath, pdfBuffer);

  let info;
  try {
    info = detectPdfMetadata(pdfPath);
  } catch (err) {
    console.error('[upload-pdf] pdfinfo error:', err.message);
    return res.status(500).json({ error: err.message });
  }
  if (info.pages < 1) {
    return res.status(400).json({ error: 'PDF has no pages' });
  }

  console.log(`[upload-pdf] ${trimmedLabel}: ${info.pages} pages, tagged=${info.tagged}, dpi=${dpi}`);

  const pages = [];
  try {
    for (let n = 1; n <= info.pages; n++) {
      const pageId = hash(`pdfpage${CACHE_VERSION}${pdfHashHex}${n}`);
      const pngPath = path.join(folderPath, `${pageId}.png`);
      // Skip render if PNG already exists (cache hit on re-upload of identical PDF)
      if (!fs.existsSync(pngPath)) {
        const outPrefix = path.join(folderPath, pageId);
        const produced = renderPdfPage(pdfPath, n, outPrefix, dpi);
        if (produced !== pngPath) {
          // pdftoppm appends .png — produced path should equal pngPath
          if (fs.existsSync(produced) && produced !== pngPath) {
            fs.renameSync(produced, pngPath);
          }
        }
      }
      // Phase 2: per-page text extraction. For pages with meaningful embedded
      // text (tagged or simply text-bearing PDFs), persist a synthetic
      // upload_context page so the existing three-column source-layout
      // renders the real page text on first view, no vision call needed.
      const pageText = extractPdfPageText(pdfPath, n);
      const pageTextTrimmed = pageText.trim();
      const hasMeaningfulText = pageTextTrimmed.length >= 30;
      let contextPageId = null;
      if (hasMeaningfulText) {
        const ctxId = uploadContextPageId(pageId);
        const ctxContent = synthesizePdfPageContext({
          text: pageText,
          pageNumber: n,
          pageCount: info.pages,
          tagged: info.tagged,
          title: info.title,
          label: trimmedLabel,
        });
        const ctxPage = {
          id: ctxId,
          type: 'markdown',
          title: 'Source context',
          content: ctxContent,
          source: 'pdftotext-layout',
          parentId: pageId,
          parentClick: null,
          initialQuery: trimmedLabel,
          mode,
          intent: 'PDF page text extraction',
          generatedAt: new Date().toISOString(),
        };
        savePageContent(folder, ctxId, ctxPage);
        pageMeta[ctxId] = {
          folder,
          query: trimmedLabel,
          mode,
          type: 'upload_context',
          parentId: pageId,
          parentClick: null,
          intent: ctxPage.intent,
        };
        contextPageId = ctxId;

        // Also surface the extracted text in the analysis JSON so the
        // Entities/langextract pipeline can prefer it over vision-derived
        // description+explanation when present.
        const existingAnalysis = loadAnalysis(pageId) || {};
        existingAnalysis.extracted = {
          text: pageText,
          source: 'pdftotext-layout',
          wordCount: pageTextTrimmed.split(/\s+/).length,
          generatedAt: new Date().toISOString(),
        };
        saveAnalysis(pageId, existingAnalysis);
      }

      pageMeta[pageId] = {
        folder,
        query: trimmedLabel,
        mode,
        type: 'image',
        parentId: null,
        pdfDocId,
        pdfPageNumber: n,
        pdfPageCount: info.pages,
        pdfTagged: info.tagged,
        pdfTitle: info.title || null,
        contextPageId,
        pdfHasText: hasMeaningfulText,
      };
      pages.push({
        id: pageId,
        type: 'image',
        imageUrl: `/generated/${folder}/${pageId}.png`,
        parentId: null,
        parentClick: null,
        initialQuery: trimmedLabel,
        mode,
        pdfDocId,
        pdfPageNumber: n,
        pdfPageCount: info.pages,
        pdfTagged: info.tagged,
        pdfTitle: info.title || null,
        contextPageId,
        pdfHasText: hasMeaningfulText,
      });
    }
    saveMetadata(pageMeta);
  } catch (err) {
    console.error('[upload-pdf] render failed:', err.message);
    return res.status(500).json({ error: 'PDF render failed: ' + err.message });
  }

  // Run docling once over the whole PDF for high-quality structured Markdown
  // (preserves headings, lists, tables) when at least one page has text. This
  // is used by /api/entities/doc/:pdfDocId. If docling fails or is unavailable,
  // the doc-level endpoint falls back to per-page pdftotext concatenation.
  const anyTextBearing = pages.some(p => p.pdfHasText);
  if (anyTextBearing) {
    const doclingMdPath = path.join(folderPath, `doc-${pdfDocId}.md`);
    if (!fs.existsSync(doclingMdPath)) {
      console.log(`[upload-pdf] running docling on ${trimmedLabel} (${info.pages} pages)...`);
      const startedDocling = Date.now();
      const docText = extractPdfWithDocling(pdfPath, path.join(folderPath, `_docling-${pdfDocId}`));
      if (docText) {
        fs.writeFileSync(doclingMdPath, docText, 'utf8');
        console.log(`[upload-pdf] docling produced ${docText.length} chars in ${Date.now() - startedDocling}ms`);
      } else {
        console.log('[upload-pdf] docling unavailable or failed; doc-level entities will fall back to pdftotext concat');
      }
    }
  }

  console.log(`[upload-pdf] done: ${pages.length} pages registered for doc ${pdfDocId}`);
  res.json({
    pdfDocId,
    pdfPageCount: info.pages,
    pdfTagged: info.tagged,
    pdfTitle: info.title || null,
    pages,
  });
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
    // PDF fields, present only on PDF-derived pages
    pdfDocId: meta.pdfDocId || null,
    pdfPageNumber: meta.pdfPageNumber || null,
    pdfPageCount: meta.pdfPageCount || null,
    pdfTagged: typeof meta.pdfTagged === 'boolean' ? meta.pdfTagged : null,
    pdfTitle: meta.pdfTitle || null,
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
          const result = await generateTableDrillPage(focusedComposite, mode, intent, contextTrail, responseDepth, language);
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
          const result = await generateDiagramDrillPage(focusedComposite, mode, intent, contextTrail, responseDepth, language);
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

// --- LangExtract structured-entities API (experiment/langextract) ---
// Spawns tools/langextract/helper.py via its venv-installed Python; returns
// grounded entities with char_intervals back into the analysis Markdown.
// If the helper or local Gemma server is unavailable, returns an error so the
// frontend can degrade gracefully without blocking the existing flow.

const LANGEXTRACT_DIR = path.join(__dirname, 'tools', 'langextract');
const LANGEXTRACT_PY = path.join(LANGEXTRACT_DIR, '.venv', 'bin', 'python');
const LANGEXTRACT_SCRIPT = path.join(LANGEXTRACT_DIR, 'helper.py');

function runLangextractHelper(text, opts = {}) {
  return new Promise((resolve) => {
    if (!fs.existsSync(LANGEXTRACT_PY) || !fs.existsSync(LANGEXTRACT_SCRIPT)) {
      resolve({
        extractions: [],
        stats: {},
        error: 'helper not installed — run tools/langextract/setup.sh',
      });
      return;
    }

    const proc = spawn(LANGEXTRACT_PY, [LANGEXTRACT_SCRIPT], {
      cwd: LANGEXTRACT_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (chunk) => { stdout += chunk; });
    proc.stderr.on('data', (chunk) => { stderr += chunk; });

    const timeoutMs = Number.isFinite(opts.timeoutMs) ? opts.timeoutMs : 240000;
    const timer = setTimeout(() => {
      try { proc.kill('SIGKILL'); } catch {}
      resolve({ extractions: [], stats: {}, error: `helper timeout after ${timeoutMs}ms` });
    }, timeoutMs);

    proc.on('error', (err) => {
      clearTimeout(timer);
      resolve({ extractions: [], stats: {}, error: `spawn failed: ${err.message}` });
    });

    proc.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        resolve({
          extractions: [],
          stats: {},
          error: `helper exited ${code}: ${stderr.trim().slice(-400)}`,
        });
        return;
      }
      try {
        const parsed = JSON.parse(stdout.trim().split('\n').pop() || '{}');
        resolve(parsed);
      } catch (err) {
        resolve({ extractions: [], stats: {}, error: `bad helper output: ${err.message}` });
      }
    });

    const payload = {
      text,
      model_id: opts.modelId || undefined,
      base_url: opts.baseUrl || undefined,
      max_char_buffer: opts.maxCharBuffer || undefined,
      extraction_passes: opts.extractionPasses || undefined,
    };
    proc.stdin.end(JSON.stringify(payload));
  });
}

app.get('/api/entities/:pageId', (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }
  const cached = loadAnalysis(pageId);
  if (cached && cached.entities) {
    return res.json({ entities: cached.entities, source: cached.entities.source || null });
  }
  res.json({ entities: null });
});

app.post('/api/entities/:pageId', async (req, res) => {
  const { pageId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }
  const analysis = loadAnalysis(pageId);
  const extractedText = analysis?.extracted?.text?.trim() || '';
  const hasVision = Boolean(analysis?.description || analysis?.explanation);
  if (!analysis || (!extractedText && !hasVision)) {
    return res.status(400).json({
      error: 'No source text available — for image pages, run Learn analysis first; for PDF pages, this means the PDF had no extractable text.',
    });
  }

  // Phase 2: prefer the directly-extracted PDF text when present. For
  // image pages and image-PDF pages with no embedded text, fall back to the
  // existing vision-derived description+explanation.
  let text;
  let sourceKind;
  if (extractedText) {
    text = extractedText;
    sourceKind = 'pdf-extracted';
  } else {
    text = [analysis.description, analysis.explanation].filter(Boolean).join('\n\n');
    sourceKind = 'vision-analysis';
  }

  console.log(`[entities] Running langextract for ${pageId} (${text.length} chars)...`);
  const result = await runLangextractHelper(text);

  if (result.error) {
    console.error(`[entities] ${result.error}`);
    return res.status(502).json({ error: result.error });
  }

  analysis.entities = {
    extractions: result.extractions,
    stats: result.stats,
    sourceText: text,
    sourceKind,
    scope: 'page',
    generatedAt: new Date().toISOString(),
  };
  saveAnalysis(pageId, analysis);

  console.log(
    `[entities] ${result.stats.grounded || 0}/${result.stats.total || 0} grounded ` +
    `in ${result.stats.elapsed_ms || '?'}ms (source: ${sourceKind}, scope: page)`
  );
  res.json({ entities: analysis.entities });
});

// --- Document-level entity extraction for PDFs ---
// Whole-document langextract pass; cached at _analysis/pdf-<pdfDocId>.json so
// every page of the PDF shares the same entity result set.

const DOC_ANALYSIS_PREFIX = 'pdf-';

function loadDocAnalysis(pdfDocId) {
  const docPath = path.join(ANALYSIS_DIR, `${DOC_ANALYSIS_PREFIX}${pdfDocId}.json`);
  if (!fs.existsSync(docPath)) return null;
  try { return JSON.parse(fs.readFileSync(docPath, 'utf8')); } catch { return null; }
}
function saveDocAnalysis(pdfDocId, data) {
  fs.mkdirSync(ANALYSIS_DIR, { recursive: true });
  const docPath = path.join(ANALYSIS_DIR, `${DOC_ANALYSIS_PREFIX}${pdfDocId}.json`);
  fs.writeFileSync(docPath, JSON.stringify(data, null, 2));
}

function findPdfDocSourceText(pdfDocId) {
  // For entity extraction we always build the source as a per-page pdftotext
  // concatenation with `## Page N` markers. This costs us some structural
  // fidelity vs. docling Markdown, but in exchange every extraction's
  // char_interval falls inside a known page range, so we can attribute each
  // entity to a specific PDF page (Phase 3 step 1).
  //
  // The richer docling Markdown is still produced at upload time and remains
  // available at generated/upload-pdf-<slug>/doc-<pdfDocId>.md for display in
  // the source-content panels — entity *input* and source-content *display*
  // serve different purposes.
  const pages = [];
  for (const [id, m] of Object.entries(pageMeta)) {
    if (m.pdfDocId === pdfDocId) pages.push({ id, m });
  }
  if (!pages.length) return null;
  pages.sort((a, b) => (a.m.pdfPageNumber || 0) - (b.m.pdfPageNumber || 0));

  const parts = [];
  const pageOffsets = []; // [{ page, start, end }] — char ranges in the joined text
  let cursor = 0;
  for (const { id, m } of pages) {
    const a = loadAnalysis(id);
    const t = (a?.extracted?.text || '').trim();
    if (!t) continue;
    const header = `## Page ${m.pdfPageNumber}\n\n`;
    const block = header + t + '\n\n';
    pageOffsets.push({ page: m.pdfPageNumber, start: cursor, end: cursor + block.length });
    parts.push(block);
    cursor += block.length;
  }
  if (!parts.length) return null;
  return { text: parts.join(''), source: 'pdf-doc-page-concat', pageOffsets };
}

// Mutate each extraction in place to add `attributes.page` based on which
// `## Page N` block its char_interval falls in.
function attributePagesToExtractions(extractions, pageOffsets) {
  if (!Array.isArray(pageOffsets) || !pageOffsets.length) return 0;
  let attributed = 0;
  for (const e of extractions) {
    const start = e?.char_interval?.start_pos;
    if (typeof start !== 'number') continue;
    for (const off of pageOffsets) {
      if (start >= off.start && start < off.end) {
        e.attributes = Object.assign({}, e.attributes || {}, { page: off.page });
        attributed++;
        break;
      }
    }
  }
  return attributed;
}

// Parse docling Markdown headings (## H2 and ### H3) into a navigable TOC.
// Maps each heading to a source PDF page by finding which per-page pdftotext
// the heading text appears in. Cached at doc level so it's free on re-open.
function buildPdfOutline(pdfDocId) {
  const pages = [];
  for (const [id, m] of Object.entries(pageMeta)) {
    if (m.pdfDocId === pdfDocId) pages.push({ id, m });
  }
  if (!pages.length) return null;
  pages.sort((a, b) => (a.m.pdfPageNumber || 0) - (b.m.pdfPageNumber || 0));

  const sample = pages[0].m;
  const docMdPath = path.join(GENERATED_DIR, sample.folder, `doc-${pdfDocId}.md`);
  const haveDocling = fs.existsSync(docMdPath);

  // Build per-page text map for heading → page lookup
  const pageTexts = pages.map(({ id, m }) => ({
    page: m.pdfPageNumber,
    text: ((loadAnalysis(id)?.extracted?.text) || '').toLowerCase(),
  }));

  function findPageForHeading(headingText) {
    const needle = headingText.toLowerCase().trim();
    if (!needle) return null;
    // Prefer a page whose text starts with the heading (most likely the heading source)
    for (const p of pageTexts) {
      const startsWith = p.text.trimStart().startsWith(needle);
      if (startsWith) return p.page;
    }
    // Otherwise any page that contains it
    for (const p of pageTexts) {
      if (p.text.includes(needle)) return p.page;
    }
    return null;
  }

  const items = [];
  if (haveDocling) {
    const md = fs.readFileSync(docMdPath, 'utf8');
    // Pull H1/H2/H3 (leading ##, ###; first H1 is rare in docling output)
    const lines = md.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const m = line.match(/^(#{1,3})\s+(.+?)\s*$/);
      if (!m) continue;
      const level = m[1].length;
      const text = m[2].replace(/\s+/g, ' ').trim();
      if (!text || /^[\d\s.,-]+$/.test(text)) continue;  // skip number-only headings
      items.push({ level, text, page: findPageForHeading(text) });
    }
  }

  // Fallback: synthesize a flat per-page outline using each page's first
  // non-empty line. Always useful as a "Pages" list even when docling is absent
  // or produced a heading-poor document.
  const pageList = pageTexts.map(p => {
    const firstLine = (loadAnalysis(pages.find(x => x.m.pdfPageNumber === p.page).id)?.extracted?.text || '')
      .split('\n').map(s => s.trim()).find(Boolean) || `Page ${p.page}`;
    return { page: p.page, firstLine: firstLine.slice(0, 120) };
  });

  return { items, pages: pageList, sourceKind: haveDocling ? 'docling-headings' : 'pdftotext-firstline' };
}

app.get('/api/outline/:pdfDocId', (req, res) => {
  const { pdfDocId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pdfDocId))) {
    return res.status(400).json({ error: 'Invalid pdfDocId' });
  }
  const cached = loadDocAnalysis(pdfDocId);
  if (cached?.outline) return res.json({ outline: cached.outline });
  const outline = buildPdfOutline(pdfDocId);
  if (!outline) return res.status(404).json({ error: 'Document not found or has no pages' });
  // Persist for cheap re-fetch
  const docData = cached || {};
  docData.outline = { ...outline, generatedAt: new Date().toISOString() };
  saveDocAnalysis(pdfDocId, docData);
  res.json({ outline: docData.outline });
});

app.get('/api/entities/doc/:pdfDocId', (req, res) => {
  const { pdfDocId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pdfDocId))) {
    return res.status(400).json({ error: 'Invalid pdfDocId' });
  }
  const cached = loadDocAnalysis(pdfDocId);
  if (cached?.entities) return res.json({ entities: cached.entities });
  res.json({ entities: null });
});

app.post('/api/entities/doc/:pdfDocId', async (req, res) => {
  const { pdfDocId } = req.params;
  if (!(/^[a-f0-9]{16}$/.test(pdfDocId))) {
    return res.status(400).json({ error: 'Invalid pdfDocId' });
  }
  // Verify the doc exists
  const anyPage = Object.entries(pageMeta).find(([_id, m]) => m.pdfDocId === pdfDocId);
  if (!anyPage) {
    return res.status(404).json({ error: 'PDF document not found' });
  }
  const pdfPageCount = anyPage[1].pdfPageCount || null;

  const sourceObj = findPdfDocSourceText(pdfDocId);
  if (!sourceObj) {
    return res.status(400).json({
      error: 'No extractable text in this PDF — entities only run on text-bearing PDFs. Run Learn on individual pages to use the per-page vision-derived path.',
    });
  }

  // Cap the doc-level source to keep extraction tractable on local hardware.
  // 30_000 chars ≈ 12 chunks at 2_500 chars/chunk on the small Gemma-4-E4B
  // running with -np 8 (4096 token slots). Each chunk takes ~10–20s on the
  // small model, so a typical run completes in 2–5 minutes. PDFs longer than
  // this are truncated at character count; the UI surfaces the cap.
  const DOC_SOURCE_CAP = 30_000;
  const fullLen = sourceObj.text.length;
  const truncated = fullLen > DOC_SOURCE_CAP;
  const inputText = truncated ? sourceObj.text.slice(0, DOC_SOURCE_CAP) : sourceObj.text;
  console.log(`[entities-doc] ${pdfDocId}: running langextract on ${inputText.length}/${fullLen} chars (${sourceObj.source}${truncated ? '; TRUNCATED' : ''})...`);

  const result = await runLangextractHelper(inputText, {
    // 2_500 chars per chunk leaves headroom for the few-shot examples + prompt
    // scaffolding inside the small model's per-slot 4096-token budget.
    maxCharBuffer: 2_500,
    timeoutMs: 1_500_000, // 25 min
  });
  if (result.error) {
    console.error(`[entities-doc] ${result.error}`);
    return res.status(502).json({ error: result.error });
  }
  // Attribute each extraction to its source page. Truncated runs use a
  // pageOffsets list shortened at the cap so we don't claim an entity came
  // from a page that wasn't actually fed to the model.
  const sourcePageOffsets = sourceObj.pageOffsets || [];
  const truncatedPageOffsets = sourcePageOffsets.filter(o => o.start < inputText.length).map(o => ({
    page: o.page,
    start: o.start,
    end: Math.min(o.end, inputText.length),
  }));
  const attributed = attributePagesToExtractions(result.extractions, truncatedPageOffsets);
  console.log(`[entities-doc] attributed ${attributed}/${result.extractions.length} extractions to pages`);

  const docData = loadDocAnalysis(pdfDocId) || {};
  docData.entities = {
    extractions: result.extractions,
    stats: result.stats,
    sourceText: inputText,
    sourceFullLength: fullLen,
    sourceTruncated: truncated,
    sourceKind: sourceObj.source,
    scope: 'document',
    pdfPageCount,
    pageOffsets: truncatedPageOffsets,
    pagesCovered: truncatedPageOffsets.map(o => o.page),
    generatedAt: new Date().toISOString(),
  };
  saveDocAnalysis(pdfDocId, docData);
  console.log(`[entities-doc] ${result.stats.grounded || 0}/${result.stats.total || 0} grounded in ${result.stats.elapsed_ms || '?'}ms (${sourceObj.source})`);
  res.json({ entities: docData.entities });
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

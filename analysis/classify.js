// Classification pipeline: one structured-output vision call per image,
// returning a payload that drives drill-down behavior, side-panel rendering,
// and downstream image-generation prompts.
//
// Schema lives in ./classify-schema.json. The model is instructed to return
// JSON conforming to that schema. We validate with ajv and degrade gracefully
// when the model returns malformed output.
//
// Local model only by default. Cloud fallback is gated by an explicit setting
// (modelConfig.allowClassifyCloudFallback) — this is distinct from
// modelConfig.localOnly so existing description/explanation cloud-fallback
// behavior is unaffected.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv from 'ajv';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCHEMA = JSON.parse(fs.readFileSync(path.join(__dirname, 'classify-schema.json'), 'utf8'));

const ajv = new Ajv({ allErrors: true, removeAdditional: 'failing', useDefaults: true, allowUnionTypes: true });
const validate = ajv.compile(SCHEMA);

export const SCHEMA_VERSION = 1;
export const VALID_CATEGORIES = ['chart', 'event_flyer', 'diagram', 'general'];
export const VALID_DRILL_DEFAULTS = ['image', 'markdown', 'chart', 'table', 'diagram'];

const CATEGORY_DEFAULT_DRILL = {
  chart: { kind: 'table', label: 'Get the data as a table' },
  event_flyer: { kind: 'markdown', label: 'Explain this in text' },
  diagram: { kind: 'image', label: 'Drill into image' },
  general: { kind: 'image', label: 'Drill into image' },
};

const CATEGORY_DEFAULT_PRESET = {
  chart: 'schematic',
  event_flyer: 'poster',
  diagram: 'schematic',
  general: 'illustrated',
};

export const CLASSIFY_SYSTEM_PROMPT = `You classify and analyze images for accessibility, learning, and downstream content generation. You return ONLY a JSON object that conforms to the schema in the user message. Never include markdown code fences, explanations, or any text outside the JSON object. Be precise and conservative — when uncertain, mark uncertainty in the appropriate field rather than guessing.`;

// Compact TypeScript-style shape spec. Much more token-efficient than dumping
// the full JSON Schema (which is the source of truth for validation but stays
// server-side). On gemma-4-E4B at -np 8 (4096-token slots), the full schema
// blew the slot at ~4400 tokens; this compact form lands closer to ~2800.
const COMPACT_SHAPE = `{
  "category": "chart" | "event_flyer" | "diagram" | "general",
  "category_confidence": 0.0-1.0,
  "category_rationale": string (one sentence),
  "style": {
    "preset": "schematic" | "poster" | "illustrated" | "photographic" | "manuscript" | "match",
    "descriptor": string (one sentence — colors, line weight, typography),
    "palette_hint": string[] | null   // up to 5 color names or hex codes
  },
  "transcript": string,                  // verbatim visible text in reading order, "" if none
  "visual_summary": string,              // 2-3 sentences
  "drill_hints": {
    "suggested_default": "image" | "markdown" | "chart" | "table" | "diagram",
    "suggested_default_label": string,   // short button label
    "suggested_intents": string[],        // 2-4 specific questions about THIS image
    "regions_of_interest": [             // 2-5 named clickable regions
      { "label": string, "rationale": string }
    ]
  },
  // Populate ONE of these matching "category"; set the others to null
  "chart": {
    "chart_type": "bar"|"line"|"scatter"|"pie"|"area"|"other",
    "title": string|null,
    "x_axis": { "label": string|null, "unit": string|null } | null,
    "y_axis": { "label": string|null, "unit": string|null } | null,
    "series": [ { "name": string, "points": [ { "x": string|number, "y": string|number } ] } ],
    "takeaway": string|null,             // one-sentence story of the chart
    "uncertainty_notes": string[]
  } | null,
  "event_flyer": {
    "title": string|null,
    "tagline": string|null,
    "dates": [ { "starts": string|null, "ends": string|null, "label_raw": string } ],
    "location": { "venue": string|null, "address": string|null } | null,
    "cost": string|null,
    "audience": string|null,
    "register_by": string|null,
    "contact": { "email": string|null, "phone": string|null, "website": string|null } | null,
    "organizer": string|null,
    "key_calls_to_action": string[]
  } | null,
  "diagram": {
    "diagram_type": "flowchart"|"schematic"|"anatomy"|"organizational"|"process"|"other",
    "subject": string|null,
    "components": [ { "label": string, "role": "input"|"process"|"output"|"container"|"annotation"|"other"|null, "description": string|null } ],
    "relationships": [ { "from": string, "to": string, "kind": "flows-to"|"contains"|"depends-on"|"labels"|"other"|null, "description": string|null } ],
    "process_steps": [ { "step": int, "summary": string } ] | null
  } | null,
  "general": {
    "subjects": string[],
    "setting": string|null,
    "notable_details": string[],
    "likely_purpose": string|null
  } | null
}`;

export function buildClassifyUserPrompt() {
  return `Analyze this image. Classify it into exactly ONE category, then fill the matching category-specific block.

Categories:
- chart: data visualization with axes/series/data points. Pick when primary content is quantitative.
- event_flyer: flyer/poster/promotional graphic. Pick when designed to inform someone about WHEN/WHERE/HOW-TO-ATTEND.
- diagram: flowchart/schematic/anatomical drawing/process diagram. Pick when primary purpose is showing HOW PARTS RELATE.
- general: anything else (photos, artwork, scenery, casual screenshots).

Return a single JSON object with this exact shape (other category blocks set to null):

${COMPACT_SHAPE}

Field rules:
- transcript: every visible text character in the image, in reading order. Empty string if none.
- visual_summary: 2-3 sentences. Always populate.
- drill_hints.suggested_default: chart→"table", event_flyer→"markdown", diagram→"image", general→"image".
- drill_hints.suggested_default_label: short button label like "Get the data as a table", "Explain this in text", "Drill into image".
- drill_hints.suggested_intents: 2-4 questions a viewer of THIS image might ask. Reference actual content.
- drill_hints.regions_of_interest: 2-5 named clickable regions with spatial language ("top-right corner", "the rightmost bar").
- category_confidence: be honest; 0.5-0.7 if two categories plausible.
- style.preset defaults: chart→"schematic", event_flyer→"poster", diagram→"schematic", general→"illustrated". Override only if clearly photographic or hand-drawn.
- style.descriptor: ONE sentence on actual visual style — used downstream to match style in generated images, so be concrete.

If you cannot read part of the image clearly, fill with your best guess and note uncertainty in the relevant uncertainty array.

Return ONLY the JSON object. No prose. No markdown fences.`;
}

// Strip accidental code fences and surrounding whitespace
function stripFences(raw) {
  const s = String(raw || '').trim();
  const fenced = s.match(/^```(?:json)?\s*([\s\S]*?)\s*```\s*$/i);
  return fenced ? fenced[1].trim() : s;
}

// Try to extract a JSON object from text that may contain prose around it.
function extractJsonObject(raw) {
  const s = stripFences(raw);
  // Fast path: the whole string parses
  try { return { ok: true, value: JSON.parse(s) }; } catch {}
  // Slow path: find the first balanced {...} block
  let depth = 0, start = -1;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === '{') { if (depth === 0) start = i; depth++; }
    else if (ch === '}') {
      depth--;
      if (depth === 0 && start !== -1) {
        const candidate = s.slice(start, i + 1);
        try { return { ok: true, value: JSON.parse(candidate) }; }
        catch { /* keep scanning */ }
      }
    }
  }
  return { ok: false, error: 'No parseable JSON object found in model output.' };
}

// Build a minimal-but-valid fallback payload for when the model's output is
// unusable. We never throw — the caller always gets something that conforms
// to the schema, with fallback_used=true so the UI can flag it.
export function buildFallbackPayload(modelName, reason, rawText = '') {
  const summary = stripFences(rawText).slice(0, 600);
  return {
    schema_version: SCHEMA_VERSION,
    classified_at: new Date().toISOString(),
    model: modelName || 'unknown',
    fallback_used: true,
    fallback_reason: reason,
    category: 'general',
    category_confidence: 0,
    category_rationale: 'Fallback: classifier output did not validate.',
    style: {
      preset: 'illustrated',
      descriptor: 'No style information available; classification fell back.',
      palette_hint: null,
    },
    transcript: '',
    visual_summary: summary || 'Image could not be classified; the analyzer returned an unparseable response.',
    drill_hints: {
      suggested_default: 'image',
      suggested_default_label: 'Drill into image',
      suggested_intents: [],
      regions_of_interest: [],
    },
    chart: null,
    event_flyer: null,
    diagram: null,
    general: {
      subjects: [],
      setting: null,
      notable_details: [],
      likely_purpose: null,
    },
  };
}

// Coerce a parsed object into a valid payload. Fills missing fields with
// defaults derived from category, normalizes the category-only payloads,
// and runs the full ajv schema check at the end.
export function normalizeAndValidate(parsed, modelName) {
  if (!parsed || typeof parsed !== 'object') {
    return { ok: false, payload: buildFallbackPayload(modelName, 'Parsed value is not an object.') };
  }

  const category = VALID_CATEGORIES.includes(parsed.category) ? parsed.category : 'general';
  const defaults = CATEGORY_DEFAULT_DRILL[category];

  const drillHints = parsed.drill_hints && typeof parsed.drill_hints === 'object' ? parsed.drill_hints : {};
  const drillDefault = VALID_DRILL_DEFAULTS.includes(drillHints.suggested_default)
    ? drillHints.suggested_default
    : defaults.kind;
  const drillLabel = typeof drillHints.suggested_default_label === 'string' && drillHints.suggested_default_label.trim()
    ? drillHints.suggested_default_label.trim().slice(0, 80)
    : defaults.label;

  const styleIn = parsed.style && typeof parsed.style === 'object' ? parsed.style : {};
  const allowedPresets = ['schematic', 'poster', 'illustrated', 'photographic', 'manuscript', 'match'];
  const stylePreset = allowedPresets.includes(styleIn.preset) ? styleIn.preset : CATEGORY_DEFAULT_PRESET[category];
  const styleDescriptor = typeof styleIn.descriptor === 'string' && styleIn.descriptor.trim()
    ? styleIn.descriptor.trim().slice(0, 400)
    : `Default ${stylePreset} style.`;
  const palette = Array.isArray(styleIn.palette_hint)
    ? styleIn.palette_hint.filter(s => typeof s === 'string').map(s => s.slice(0, 32)).slice(0, 5)
    : null;

  const cleanRegions = Array.isArray(drillHints.regions_of_interest)
    ? drillHints.regions_of_interest
        .filter(r => r && typeof r.label === 'string' && typeof r.rationale === 'string')
        .map(r => ({ label: r.label.slice(0, 120), rationale: r.rationale.slice(0, 200) }))
        .slice(0, 8)
    : [];

  const cleanIntents = Array.isArray(drillHints.suggested_intents)
    ? drillHints.suggested_intents
        .filter(s => typeof s === 'string' && s.trim())
        .map(s => s.trim().slice(0, 200))
        .slice(0, 6)
    : [];

  const payload = {
    schema_version: SCHEMA_VERSION,
    classified_at: new Date().toISOString(),
    model: modelName || 'unknown',
    fallback_used: false,
    fallback_reason: null,
    category,
    category_confidence: clamp01(parsed.category_confidence),
    category_rationale: typeof parsed.category_rationale === 'string'
      ? parsed.category_rationale.slice(0, 600)
      : '',
    style: { preset: stylePreset, descriptor: styleDescriptor, palette_hint: palette },
    transcript: typeof parsed.transcript === 'string' ? parsed.transcript : '',
    visual_summary: typeof parsed.visual_summary === 'string'
      ? parsed.visual_summary.slice(0, 1200)
      : '',
    drill_hints: {
      suggested_default: drillDefault,
      suggested_default_label: drillLabel,
      suggested_intents: cleanIntents,
      regions_of_interest: cleanRegions,
    },
    chart: category === 'chart' ? normalizeChart(parsed.chart) : null,
    event_flyer: category === 'event_flyer' ? normalizeFlyer(parsed.event_flyer) : null,
    diagram: category === 'diagram' ? normalizeDiagram(parsed.diagram) : null,
    general: category === 'general' ? normalizeGeneral(parsed.general) : null,
  };

  if (!validate(payload)) {
    const errs = (validate.errors || []).slice(0, 3).map(e => `${e.instancePath} ${e.message}`).join('; ');
    return { ok: false, payload: buildFallbackPayload(modelName, `Schema validation failed: ${errs}`) };
  }
  return { ok: true, payload };
}

function clamp01(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return 0.5;
  return Math.max(0, Math.min(1, v));
}

function s(x, max = 200) { return typeof x === 'string' ? x.slice(0, max) : null; }
function arr(x, fn, max = 10) { return Array.isArray(x) ? x.map(fn).filter(Boolean).slice(0, max) : []; }

function normalizeChart(c) {
  if (!c || typeof c !== 'object') return { chart_type: 'other', title: null, x_axis: null, y_axis: null, series: [], takeaway: null, uncertainty_notes: [] };
  const types = ['bar', 'line', 'scatter', 'pie', 'area', 'other'];
  return {
    chart_type: types.includes(c.chart_type) ? c.chart_type : 'other',
    title: s(c.title, 200),
    x_axis: c.x_axis && typeof c.x_axis === 'object' ? { label: s(c.x_axis.label, 100), unit: s(c.x_axis.unit, 60) } : null,
    y_axis: c.y_axis && typeof c.y_axis === 'object' ? { label: s(c.y_axis.label, 100), unit: s(c.y_axis.unit, 60) } : null,
    series: arr(c.series, ser => {
      if (!ser || typeof ser !== 'object' || typeof ser.name !== 'string') return null;
      return {
        name: ser.name.slice(0, 100),
        points: arr(ser.points, pt => {
          if (!pt || typeof pt !== 'object' || !('x' in pt) || !('y' in pt)) return null;
          return { x: coercePoint(pt.x), y: coercePoint(pt.y) };
        }, 200),
      };
    }, 20),
    takeaway: s(c.takeaway, 400),
    uncertainty_notes: arr(c.uncertainty_notes, x => typeof x === 'string' ? x.slice(0, 200) : null, 10),
  };
}

function coercePoint(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string') return v.slice(0, 100);
  return null;
}

function normalizeFlyer(f) {
  if (!f || typeof f !== 'object') return { title: null, tagline: null, dates: [], location: null, cost: null, audience: null, register_by: null, contact: null, organizer: null, key_calls_to_action: [] };
  return {
    title: s(f.title, 200),
    tagline: s(f.tagline, 300),
    dates: arr(f.dates, d => {
      if (!d || typeof d !== 'object' || typeof d.label_raw !== 'string') return null;
      return {
        starts: typeof d.starts === 'string' ? d.starts.slice(0, 50) : null,
        ends: typeof d.ends === 'string' ? d.ends.slice(0, 50) : null,
        label_raw: d.label_raw.slice(0, 200),
      };
    }, 10),
    location: f.location && typeof f.location === 'object' ? { venue: s(f.location.venue, 200), address: s(f.location.address, 300) } : null,
    cost: s(f.cost, 200),
    audience: s(f.audience, 200),
    register_by: typeof f.register_by === 'string' ? f.register_by.slice(0, 50) : null,
    contact: f.contact && typeof f.contact === 'object' ? { email: s(f.contact.email, 200), phone: s(f.contact.phone, 60), website: s(f.contact.website, 300) } : null,
    organizer: s(f.organizer, 200),
    key_calls_to_action: arr(f.key_calls_to_action, x => typeof x === 'string' ? x.slice(0, 200) : null, 6),
  };
}

function normalizeDiagram(d) {
  if (!d || typeof d !== 'object') return { diagram_type: 'other', subject: null, components: [], relationships: [], process_steps: null };
  const types = ['flowchart', 'schematic', 'anatomy', 'organizational', 'process', 'other'];
  const roles = ['input', 'process', 'output', 'container', 'annotation', 'other'];
  const kinds = ['flows-to', 'contains', 'depends-on', 'labels', 'other'];
  return {
    diagram_type: types.includes(d.diagram_type) ? d.diagram_type : 'other',
    subject: s(d.subject, 300),
    components: arr(d.components, c => {
      if (!c || typeof c !== 'object' || typeof c.label !== 'string') return null;
      return {
        label: c.label.slice(0, 120),
        role: roles.includes(c.role) ? c.role : null,
        description: s(c.description, 300),
      };
    }, 30),
    relationships: arr(d.relationships, r => {
      if (!r || typeof r !== 'object' || typeof r.from !== 'string' || typeof r.to !== 'string') return null;
      return {
        from: r.from.slice(0, 120),
        to: r.to.slice(0, 120),
        kind: kinds.includes(r.kind) ? r.kind : null,
        description: s(r.description, 200),
      };
    }, 50),
    process_steps: Array.isArray(d.process_steps) ? d.process_steps
      .filter(p => p && typeof p === 'object' && Number.isFinite(p.step) && typeof p.summary === 'string')
      .map(p => ({ step: Math.max(1, Math.floor(p.step)), summary: p.summary.slice(0, 300) }))
      .slice(0, 30) : null,
  };
}

function normalizeGeneral(g) {
  if (!g || typeof g !== 'object') return { subjects: [], setting: null, notable_details: [], likely_purpose: null };
  return {
    subjects: arr(g.subjects, x => typeof x === 'string' ? x.slice(0, 120) : null, 10),
    setting: s(g.setting, 300),
    notable_details: arr(g.notable_details, x => typeof x === 'string' ? x.slice(0, 200) : null, 10),
    likely_purpose: s(g.likely_purpose, 300),
  };
}

// Main entry point. Caller passes an `analyzeFn` that takes (systemPrompt, userPrompt)
// and returns { text, source }. We keep the analyzer pluggable so the server can
// pass its existing local-only Gemma helper without us having to import all
// the fetch/auth logic here.
export async function classifyImage(imageBase64, analyzeFn, modelName = 'gemma-4-E4B') {
  let raw;
  try {
    const result = await analyzeFn(CLASSIFY_SYSTEM_PROMPT, buildClassifyUserPrompt(), imageBase64);
    raw = result?.text || '';
  } catch (err) {
    return buildFallbackPayload(modelName, `Vision call failed: ${err.message}`);
  }

  const parsed = extractJsonObject(raw);
  if (!parsed.ok) {
    return buildFallbackPayload(modelName, parsed.error, raw);
  }

  const norm = normalizeAndValidate(parsed.value, modelName);
  return norm.payload;
}

// Build a context block for downstream image-generation prompts. The caller
// (image-gen flow) decides whether to prepend this to the user prompt.
// Returns an empty string if the classification is missing or fallback.
export function buildGenerationContext(classified) {
  if (!classified || classified.fallback_used) return '';
  const parts = [];
  parts.push(`PARENT IMAGE CONTEXT (from upstream classification):`);
  parts.push(`- Category: ${classified.category} (confidence ${classified.category_confidence.toFixed(2)})`);
  if (classified.visual_summary) parts.push(`- Visual summary: ${classified.visual_summary}`);
  if (classified.style?.descriptor) parts.push(`- Visual style: ${classified.style.descriptor}`);
  if (classified.style?.palette_hint?.length) parts.push(`- Palette: ${classified.style.palette_hint.join(', ')}`);

  // Category-specific facts most useful for image generation
  if (classified.category === 'chart' && classified.chart) {
    const c = classified.chart;
    if (c.title) parts.push(`- Chart title: ${c.title}`);
    if (c.takeaway) parts.push(`- Chart takeaway: ${c.takeaway}`);
    if (c.x_axis?.label || c.y_axis?.label) parts.push(`- Axes: x=${c.x_axis?.label || '?'}, y=${c.y_axis?.label || '?'}${c.y_axis?.unit ? ` (${c.y_axis.unit})` : ''}`);
    if (c.series?.length) {
      const summary = c.series.slice(0, 3).map(s => `${s.name}: ${s.points.slice(0, 8).map(p => `${p.x}=${p.y}`).join(', ')}`).join(' | ');
      parts.push(`- Series data: ${summary}`);
    }
  } else if (classified.category === 'event_flyer' && classified.event_flyer) {
    const f = classified.event_flyer;
    if (f.title) parts.push(`- Event: ${f.title}`);
    if (f.tagline) parts.push(`- Tagline: ${f.tagline}`);
    if (f.dates?.length) parts.push(`- When: ${f.dates.map(d => d.label_raw).join('; ')}`);
    if (f.location?.venue || f.location?.address) parts.push(`- Where: ${[f.location.venue, f.location.address].filter(Boolean).join(', ')}`);
    if (f.cost) parts.push(`- Cost: ${f.cost}`);
  } else if (classified.category === 'diagram' && classified.diagram) {
    const d = classified.diagram;
    if (d.subject) parts.push(`- Subject: ${d.subject}`);
    if (d.diagram_type) parts.push(`- Diagram type: ${d.diagram_type}`);
    if (d.components?.length) {
      parts.push(`- Components: ${d.components.slice(0, 8).map(c => c.label).join(', ')}`);
    }
  } else if (classified.category === 'general' && classified.general) {
    const g = classified.general;
    if (g.subjects?.length) parts.push(`- Subjects: ${g.subjects.join(', ')}`);
    if (g.setting) parts.push(`- Setting: ${g.setting}`);
  }

  return parts.join('\n');
}

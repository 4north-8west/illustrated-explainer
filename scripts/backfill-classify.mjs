#!/usr/bin/env node
// Backfill classification on existing image pages — DIRECT to llama-server.
//
// Walks generated/metadata.json, finds image-type pages whose _analysis/<id>.json
// is missing the `classified` field, and runs the classify pipeline directly
// against llama-server on port 8080 (or whatever LLAMA_SERVER_URL points at).
// Writes the result to generated/_analysis/<id>.json without touching the
// running illustrated-explainer Node server — so the user's main app on port
// 3000 can stay running undisturbed.
//
// The classifications written here are picked up by any illustrated-explainer
// server on next page-load, since pageFromMetadata reads them from disk.
//
// Usage:
//   node scripts/backfill-classify.mjs
//   node scripts/backfill-classify.mjs --dry-run
//   node scripts/backfill-classify.mjs --force --limit 5
//   LLAMA_SERVER_URL=http://localhost:8082 node scripts/backfill-classify.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { classifyImage } from '../analysis/classify.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const GENERATED = path.join(ROOT, 'generated');
const METADATA = path.join(GENERATED, 'metadata.json');
const ANALYSIS = path.join(GENERATED, '_analysis');

const args = process.argv.slice(2);
const DRY_RUN = args.includes('--dry-run');
const FORCE = args.includes('--force');
const LIMIT = (() => {
  const i = args.indexOf('--limit');
  if (i !== -1 && args[i + 1]) return Number(args[i + 1]);
  return Infinity;
})();
const CONCURRENCY = (() => {
  const i = args.indexOf('--concurrency');
  if (i !== -1 && args[i + 1]) return Math.max(1, Number(args[i + 1]));
  return 4;  // matches default llama-server -np 4
})();
const MODEL_NAME = process.env.CLASSIFY_MODEL || 'gemma-4-E4B-it-Q4_K_S';
const LLAMA_URL = (process.env.LLAMA_SERVER_URL || 'http://localhost:8080').replace(/\/$/, '');
const CHAT_ENDPOINT = `${LLAMA_URL}/v1/chat/completions`;

function loadMetadata() {
  try { return JSON.parse(fs.readFileSync(METADATA, 'utf8')); }
  catch (err) { console.error(`Could not read ${METADATA}: ${err.message}`); process.exit(1); }
}

function loadAnalysis(pageId) {
  const p = path.join(ANALYSIS, `${pageId}.json`);
  if (!fs.existsSync(p)) return null;
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return null; }
}

function saveAnalysis(pageId, data) {
  fs.mkdirSync(ANALYSIS, { recursive: true });
  fs.writeFileSync(path.join(ANALYSIS, `${pageId}.json`), JSON.stringify(data, null, 2));
}

function isAlreadyClassified(pageId) {
  const data = loadAnalysis(pageId);
  return Boolean(data?.classified && !data.classified.fallback_used);
}

// Direct llama-server caller — matches the OpenAI-compatible chat-completions
// shape the in-app callVisionChat() builds, minus the cloud-fallback layer.
// Two key differences from the in-app helper:
//   1. response_format: { type: "json_object" } constrains the model's output
//      to valid JSON. Without this, the small Gemma intermittently produces a
//      good JSON payload then runs on with literal "\n" characters until it
//      hits max_tokens (causing a truncation-driven parse failure).
//   2. max_tokens lowered to 2048 — the largest realistic structured payload
//      for our schema is well under 1500 tokens. Tighter ceiling = fewer
//      runaway generations stealing GPU time.
async function callLlamaServer(systemPrompt, userPrompt, imageBase64) {
  const body = {
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
    temperature: 0.2,
    max_tokens: 2048,
    response_format: { type: 'json_object' },
  };
  const res = await fetch(CHAT_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(300_000), // 5 min ceiling per call
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`llama-server ${res.status}: ${text.slice(0, 240)}`);
  }
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content || '';
  return { text, source: `local (${MODEL_NAME})` };
}

async function classifyOnePage(pageId) {
  const meta = loadMetadata()[pageId];
  if (!meta) throw new Error('page not in metadata');
  const imagePath = path.join(GENERATED, meta.folder, `${pageId}.png`);
  if (!fs.existsSync(imagePath)) throw new Error(`image file not found: ${imagePath}`);
  const imageBase64 = fs.readFileSync(imagePath).toString('base64');

  const payload = await classifyImage(imageBase64, callLlamaServer, MODEL_NAME);

  const existing = loadAnalysis(pageId) || {};
  existing.classified = payload;
  existing.classifiedAt = payload.classified_at;
  saveAnalysis(pageId, existing);
  return payload;
}

async function main() {
  // Quick reachability check on the llama-server
  try {
    const r = await fetch(`${LLAMA_URL}/v1/models`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const { models } = await r.json();
    const loaded = models?.[0]?.name || '(unknown)';
    console.log(`llama-server reachable at ${LLAMA_URL} — currently loaded: ${loaded}`);
  } catch (err) {
    console.error(`✗ Cannot reach llama-server at ${LLAMA_URL}: ${err.message}`);
    console.error(`  Start it first, or set LLAMA_SERVER_URL.`);
    process.exit(1);
  }

  const meta = loadMetadata();
  const candidates = [];
  for (const [id, m] of Object.entries(meta)) {
    if (m?.type !== 'image') continue;
    if (!FORCE && isAlreadyClassified(id)) continue;
    candidates.push({ id, query: m.query || '(unnamed)', folder: m.folder || '?' });
  }

  console.log(`Found ${Object.keys(meta).length} pages total, ${candidates.length} image pages need${FORCE ? ' (force)' : ''} classification.`);
  if (LIMIT < candidates.length) console.log(`Will process the first ${LIMIT} (--limit).`);
  if (DRY_RUN) {
    console.log('\n--dry-run; would classify:');
    for (const c of candidates.slice(0, LIMIT)) console.log(`  ${c.id}  ${c.folder}  "${c.query}"`);
    return;
  }

  const target = candidates.slice(0, LIMIT);
  console.log(`Concurrency: ${CONCURRENCY} parallel calls (matches llama-server -np)`);
  let done = 0, failed = 0;
  let cursor = 0;
  const start = Date.now();

  async function worker(workerId) {
    while (cursor < target.length) {
      const i = cursor++;
      const c = target[i];
      const t0 = Date.now();
      try {
        const cls = await classifyOnePage(c.id);
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        const fb = cls.fallback_used ? ' [FALLBACK]' : '';
        console.log(`[${i + 1}/${target.length}] w${workerId} ${c.id} "${c.query.slice(0, 50)}" — ${elapsed}s → ${cls.category} (${(cls.category_confidence * 100).toFixed(0)}%)${fb}`);
        done++;
      } catch (err) {
        console.log(`[${i + 1}/${target.length}] w${workerId} ${c.id} FAILED: ${err.message}`);
        failed++;
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, target.length) }, (_, i) => worker(i + 1)));
  const totalElapsed = ((Date.now() - start) / 60_000).toFixed(1);
  console.log(`\nDone. ${done} succeeded, ${failed} failed in ${totalElapsed} min.`);
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });

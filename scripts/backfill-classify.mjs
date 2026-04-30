#!/usr/bin/env node
// Backfill classification on existing image pages that don't have it yet.
//
// Walks generated/metadata.json, finds image-type pages whose corresponding
// _analysis/<id>.json is missing the `classified` field, and POSTs to the
// running server's /api/classify/<id> endpoint for each — serially, since
// llama-server has one slot.
//
// Usage:
//   1. Make sure the server is running (default http://localhost:3000;
//      override with PORT env var or pass --port).
//   2. node scripts/backfill-classify.mjs
//      Optional flags:
//        --dry-run        Show what would be classified, don't make calls
//        --force          Re-classify even pages that already have a payload
//        --limit N        Stop after N successful classifications
//        --port N         Use http://localhost:N instead of $PORT or 3000

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const GENERATED = path.join(ROOT, 'generated');
const METADATA = path.join(GENERATED, 'metadata.json');
const ANALYSIS = path.join(GENERATED, '_analysis');

const args = process.argv.slice(2);
const DRY_RUN = args.includes('--dry-run');
const FORCE = args.includes('--force');
const PORT = (() => {
  const i = args.indexOf('--port');
  if (i !== -1 && args[i + 1]) return Number(args[i + 1]);
  return Number(process.env.PORT) || 3000;
})();
const LIMIT = (() => {
  const i = args.indexOf('--limit');
  if (i !== -1 && args[i + 1]) return Number(args[i + 1]);
  return Infinity;
})();

const SERVER = `http://localhost:${PORT}`;

function loadMetadata() {
  try { return JSON.parse(fs.readFileSync(METADATA, 'utf8')); }
  catch (err) {
    console.error(`Could not read ${METADATA}: ${err.message}`);
    process.exit(1);
  }
}

function isAlreadyClassified(pageId) {
  const p = path.join(ANALYSIS, `${pageId}.json`);
  if (!fs.existsSync(p)) return false;
  try {
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const c = data?.classified;
    return Boolean(c && !c.fallback_used);
  } catch { return false; }
}

async function main() {
  // Sanity check: server up?
  try {
    const r = await fetch(`${SERVER}/api/modes`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) throw new Error(`Server replied ${r.status}`);
  } catch (err) {
    console.error(`✗ Cannot reach server at ${SERVER}: ${err.message}`);
    console.error(`  Start it first: PORT=${PORT} node server.js`);
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

  let done = 0;
  let failed = 0;
  const start = Date.now();
  for (const c of candidates.slice(0, LIMIT)) {
    const t0 = Date.now();
    process.stdout.write(`[${done + failed + 1}/${Math.min(LIMIT, candidates.length)}] ${c.id} "${c.query}" ... `);
    try {
      const r = await fetch(`${SERVER}/api/classify/${c.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: FORCE }),
        signal: AbortSignal.timeout(300_000), // 5 min ceiling per call
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      const cls = body?.classified;
      if (!cls) throw new Error('no classified payload returned');
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      const fb = cls.fallback_used ? ' [FALLBACK]' : '';
      console.log(`${elapsed}s  →  ${cls.category} (${(cls.category_confidence * 100).toFixed(0)}%)${fb}`);
      done++;
    } catch (err) {
      console.log(`FAILED: ${err.message}`);
      failed++;
    }
  }
  const totalElapsed = ((Date.now() - start) / 60_000).toFixed(1);
  console.log(`\nDone. ${done} succeeded, ${failed} failed in ${totalElapsed} min.`);
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });

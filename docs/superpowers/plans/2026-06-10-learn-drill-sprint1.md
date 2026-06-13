# Learn & Drill Sprint 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix silent analysis failures (C1/C2), add Ollama Cloud fallback, repair right-column layout (H1), add Learn panel polling (H2), and surface vault search errors (H3).

**Architecture:** All server changes are in `server.js` (single Express file). All frontend changes are in `public/index.html` (single-page vanilla JS + inline CSS). No new files are required. Changes are organized in dependency order: server fallback chain first, then the analysis endpoint refactor, then the frontend that consumes the new response shape, then layout, then vault.

**Tech Stack:** Node.js/Express, vanilla JS, Sharp (image processing), no test framework — manual browser verification after each task.

---

## File map

| File | Tasks that touch it |
|------|---------------------|
| `server.js` | T1 (Ollama registry + fallback helpers), T2 (POST /api/analysis fire-and-forget + C2 gate fix) |
| `public/index.html` | T3 (Learn panel error + polling), T4 (right-column CSS + drill dismiss), T5 (vault error) |
| `vault.js` | T5 (return error field) |

---

## Task 1: Ollama Cloud Fallback — `server.js`

**Files:**
- Modify: `server.js:15-23` (env constants)
- Modify: `server.js:83-139` (MODEL_REGISTRY)
- Modify: `server.js:2038-2054` (`analyzeImage`)
- Modify: `server.js:2061-2077` (`analyzeImageForClassify`)
- Modify: `server.js:809-818` (`generateTextDrillPage` fallback)
- Modify: `server.js:1033-1041` (`generateChartDrillPage` fallback)
- Modify: `server.js:1119-1127` (`generateTableDrillPage` fallback)
- Modify: `server.js:1183-1193` (`generateDiagramDrillPage` fallback)
- Modify: `server.js:1489-1497` (`translatePageContent` fallback)

---

- [ ] **Step 1.1: Add OLLAMA_API_KEY constant next to the other API key declarations (around line 15)**

```javascript
// existing lines (shown for context):
const XAI_API_KEY = process.env.XAI_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
// ADD after GEMINI_API_KEY:
const OLLAMA_API_KEY = process.env.OLLAMA_API_KEY;
const OLLAMA_FALLBACK_MODEL = 'gemma4:31b-cloud';
```

- [ ] **Step 1.2: Add `ollama` provider to MODEL_REGISTRY (after the `local` entry, around line 139)**

```javascript
  ollama: {
    name: 'Ollama Cloud',
    models: {
      [OLLAMA_FALLBACK_MODEL]: {
        name: 'Gemma 4 31B Cloud',
        capabilities: ['analysis', 'classify', 'drillText'],
      },
    },
    chatUrl: 'https://api.ollama.com/v1/chat/completions',
    authHeader: () => (OLLAMA_API_KEY ? `Bearer ${OLLAMA_API_KEY}` : null),
  },
```

---

- [ ] **Step 1.3: Add two small helpers after the MODEL_REGISTRY block (after `resolveChatUrl`, around line 150). These replace every per-function inline fallback choice.**

```javascript
function isLocalOutage(err) {
  return (
    err.message === 'VISION_NOT_SUPPORTED' ||
    err.message.includes('ECONNREFUSED') ||
    err.message.includes('not reachable')
  );
}

// Returns ['ollama', OLLAMA_FALLBACK_MODEL] if Ollama key is set,
// ['xai', 'grok-4-1-fast-non-reasoning'] otherwise.
// Returns null if neither key is configured.
function cloudFallback() {
  if (OLLAMA_API_KEY) return ['ollama', OLLAMA_FALLBACK_MODEL];
  if (XAI_API_KEY)    return ['xai', 'grok-4-1-fast-non-reasoning'];
  return null;
}
```

---

- [ ] **Step 1.4: Update `analyzeImage` to use `cloudFallback()` (currently around line 2044–2053)**

Replace:
```javascript
// OLD — only falls back to Grok
if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
  console.log(`[analysis] Local model unavailable (${err.message}), falling back to Grok...`);
  const text = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
  return { text, source: 'grok (fallback)' };
}
```

With:
```javascript
// NEW — Ollama first, Grok second
if (cfg.provider === 'local' && isLocalOutage(err)) {
  const fb = cloudFallback();
  if (!modelConfig.localOnly && fb) {
    const [fbProvider, fbModel] = fb;
    console.log(`[analysis] Local outage → ${fbProvider}/${fbModel}`);
    const text = await callVisionChat(fbProvider, fbModel, imageBase64, systemPrompt, userPrompt);
    return { text, source: `${fbProvider} (fallback)` };
  }
}
```

---

- [ ] **Step 1.5: Update `analyzeImageForClassify` fallback (currently around line 2068–2076)**

Replace:
```javascript
// OLD
if (isLocalOutage && modelConfig.allowClassifyCloudFallback === true) {
  console.log(`[classify] Local model unavailable (${err.message}); cloud fallback authorized — calling Grok.`);
  const text = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
  return { text, source: 'grok (fallback)' };
}
```

With:
```javascript
// NEW — Ollama first (no gate), Grok gated by allowClassifyCloudFallback
if (cfg.provider === 'local' && isLocalOutage(err)) {
  if (OLLAMA_API_KEY) {
    console.log(`[classify] Local outage → ollama/${OLLAMA_FALLBACK_MODEL}`);
    const text = await callVisionChat('ollama', OLLAMA_FALLBACK_MODEL, imageBase64, systemPrompt, userPrompt);
    return { text, source: 'ollama (fallback)' };
  }
  if (modelConfig.allowClassifyCloudFallback === true && XAI_API_KEY) {
    console.log(`[classify] Local outage → xai/grok (fallback authorized)`);
    const text = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
    return { text, source: 'grok (fallback)' };
  }
}
```

---

- [ ] **Step 1.6: Update `generateTextDrillPage` fallback (currently around line 813–818)**

Replace:
```javascript
if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
  const text = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
  return { text, source: 'grok (fallback)' };
}
```

With:
```javascript
if (cfg.provider === 'local' && isLocalOutage(err)) {
  const fb = cloudFallback();
  if (!modelConfig.localOnly && fb) {
    const [fbProvider, fbModel] = fb;
    const text = await callVisionChat(fbProvider, fbModel, imageBase64, systemPrompt, userPrompt);
    return { text, source: `${fbProvider} (fallback)` };
  }
}
```

---

- [ ] **Step 1.7: Update `generateChartDrillPage` fallback (around line 1033–1040)**

Replace:
```javascript
if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
  result = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
  usedFallback = true;
}
```

With:
```javascript
if (cfg.provider === 'local' && isLocalOutage(err)) {
  const fb = cloudFallback();
  if (!modelConfig.localOnly && fb) {
    const [fbProvider, fbModel] = fb;
    result = await callVisionChat(fbProvider, fbModel, imageBase64, systemPrompt, userPrompt);
    usedFallback = true;
  } else { throw err; }
} else { throw err; }
```

Also update the retry path in the same function (~line 1055–1062):
```javascript
// Was: usedFallback ? 'xai' : cfg.provider, usedFallback ? 'grok-4-1-fast-non-reasoning' : cfg.model
// Becomes:
const retryProvider = usedFallback ? (cloudFallback()?.[0] ?? 'xai') : cfg.provider;
const retryModel    = usedFallback ? (cloudFallback()?.[1] ?? 'grok-4-1-fast-non-reasoning') : cfg.model;
const retryResult = await callVisionChat(retryProvider, retryModel, imageBase64, systemPrompt, retryPrompt);
```

---

- [ ] **Step 1.8: Update `generateTableDrillPage` fallback (around line 1119–1127)**

Replace:
```javascript
if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
  result = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
  const table = normalizeTableSpec(parseJsonResponse(result));
  return { table, source: 'grok (fallback)' };
}
```

With:
```javascript
if (cfg.provider === 'local' && isLocalOutage(err)) {
  const fb = cloudFallback();
  if (!modelConfig.localOnly && fb) {
    const [fbProvider, fbModel] = fb;
    result = await callVisionChat(fbProvider, fbModel, imageBase64, systemPrompt, userPrompt);
    const table = normalizeTableSpec(parseJsonResponse(result));
    return { table, source: `${fbProvider} (fallback)` };
  }
}
throw err;
```

---

- [ ] **Step 1.9: Update `generateDiagramDrillPage` fallback (around line 1183–1193)**

Replace:
```javascript
if (!modelConfig.localOnly && cfg.provider === 'local' && (err.message === 'VISION_NOT_SUPPORTED' || err.message.includes('ECONNREFUSED'))) {
  result = await callVisionChat('xai', 'grok-4-1-fast-non-reasoning', imageBase64, systemPrompt, userPrompt);
  const diagram = normalizeDiagramSpec(parseJsonResponse(result));
  return { diagram, source: 'grok (fallback)' };
}
```

With:
```javascript
if (cfg.provider === 'local' && isLocalOutage(err)) {
  const fb = cloudFallback();
  if (!modelConfig.localOnly && fb) {
    const [fbProvider, fbModel] = fb;
    result = await callVisionChat(fbProvider, fbModel, imageBase64, systemPrompt, userPrompt);
    const diagram = normalizeDiagramSpec(parseJsonResponse(result));
    return { diagram, source: `${fbProvider} (fallback)` };
  }
}
throw err;
```

---

- [ ] **Step 1.10: Update `translatePageContent` fallback (around line 1492–1496)**

Replace:
```javascript
if (!modelConfig.localOnly && cfg.provider === 'local' && err.message.includes('ECONNREFUSED')) {
  const text = await callTextChat('xai', 'grok-4-1-fast-non-reasoning', systemPrompt, userPrompt);
  return { text, source: 'grok (fallback)' };
}
```

With:
```javascript
if (cfg.provider === 'local' && isLocalOutage(err)) {
  const fb = cloudFallback();
  if (!modelConfig.localOnly && fb) {
    const [fbProvider, fbModel] = fb;
    const text = await callTextChat(fbProvider, fbModel, systemPrompt, userPrompt);
    return { text, source: `${fbProvider} (fallback)` };
  }
}
```

---

- [ ] **Step 1.11: Manual verification — start the server without XAI_API_KEY but with OLLAMA_API_KEY**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
OLLAMA_API_KEY=test node server.js
```

Open http://localhost:3000. Upload an image. Check server logs: when the local model is unavailable, logs should show `[analysis] Local outage → ollama/gemma4:31b-cloud`. (The test key will fail with 401 from Ollama, but the routing is confirmed.)

- [ ] **Step 1.12: Commit**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
git add server.js
git commit -m "feat: add Ollama Cloud fallback (gemma4:31b-cloud) before Grok on local outage"
```

---

## Task 2: C1+C2 — Fire-and-forget POST /api/analysis + synthesis gate fix — `server.js`

**Files:**
- Modify: `server.js:2652-2700` (`POST /api/analysis/:pageId` handler)

The current handler blocks the HTTP response waiting for 30-90 second model calls. This task changes it to fire work off the intake/synthesis queues and return immediately with current cache state + an `analyzing` flag. The frontend (Task 3) polls for completion.

C2 fix: the current code skips `enqueueSynthesis` if `classified.fallback_used`. When a user *explicitly* requests analysis, we should attempt synthesis regardless — the classify fallback might have been a transient model outage.

---

- [ ] **Step 2.1: Replace the entire `POST /api/analysis/:pageId` handler body**

Find the handler starting at `app.post('/api/analysis/:pageId', async (req, res) => {` (around line 2652). Replace its body with:

```javascript
app.post('/api/analysis/:pageId', async (req, res) => {
  const { pageId } = req.params;
  const { type } = req.body; // 'description', 'explanation', or 'both'

  if (!(/^[a-f0-9]{16}$/.test(pageId))) {
    return res.status(400).json({ error: 'Invalid page ID' });
  }

  const meta = pageMeta[pageId];
  if (!meta) return res.status(404).json({ error: 'Page not found in metadata' });

  const imagePath = path.join(GENERATED_DIR, meta.folder, `${pageId}.png`);
  if (!fs.existsSync(imagePath)) {
    return res.status(404).json({ error: 'Image file not found' });
  }

  // v2 pipeline: fire-and-forget — do not block the HTTP response.
  // The client reads `analyzing: true` and polls GET /api/intake/:pageId.
  const existing = loadAnalysis(pageId) || {};
  const wantDescription = type === 'description' || type === 'both';
  const wantExplanation = type === 'explanation' || type === 'both';
  let analyzing = false;

  if (wantDescription && !existing.description) {
    enqueueImageIntake(pageId);   // fire-and-forget
    analyzing = true;
  }

  if (wantExplanation && !existing.explanation) {
    if (!existing.classified) {
      enqueueImageIntake(pageId); // fire-and-forget; classify runs first, then synthesis is queued downstream
      analyzing = true;
    } else {
      // C2 fix: attempt synthesis regardless of fallback_used when user explicitly triggers.
      // The prior fallback may have been a transient outage; synthesis costs no vision call.
      enqueueSynthesis(pageId);   // fire-and-forget
      analyzing = true;
    }
  }

  res.json({ ...existing, analyzing });
});
```

---

- [ ] **Step 2.2: Ensure `enqueueImageIntake` automatically queues synthesis after classify completes**

Open `server.js` and find `runImageIntake` (the function called by `enqueueIntakeWork` inside `enqueueImageIntake`). Confirm it already queues synthesis for the "both" path. Search for `enqueueSynthesis` inside `runImageIntake`; if it's called there conditionally, verify the condition does not block on `fallback_used`:

```bash
grep -n "enqueueSynthesis\|runImageIntake\|enqueueImageIntake" /Users/timhohne/Documents/my-agent-team/illustrated-explainer/server.js
```

If `runImageIntake` calls `enqueueSynthesis` only when `!payload.fallback_used`, add a secondary attempt after the primary:

Find the block in `runImageIntake` that looks like:
```javascript
if (!payload.fallback_used) {
  enqueueSynthesis(pageId);
}
```

Leave it as-is (background intake keeps the condition). The explicit user-trigger path in the POST handler (Step 2.1) already calls `enqueueSynthesis` unconditionally.

---

- [ ] **Step 2.3: Manual verification**

Start the server and open http://localhost:3000. Generate an image. Open the Learn panel. Click "Analyze description & concepts". The server console should show no blocking — the POST should return in under 200ms. Browser network tab: POST /api/analysis should complete quickly with `{ analyzing: true }`. The Learn panel should show the loading spinner (wired in Task 3 — if Task 3 is not done yet, the spinner won't appear but the fast response is verifiable in DevTools).

- [ ] **Step 2.4: Commit**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
git add server.js
git commit -m "fix(C1,C2): POST /api/analysis returns immediately; synthesis no longer gated on fallback_used"
```

---

## Task 3: C1+H2 — Learn panel inline error state + intake polling — `public/index.html`

**Files:**
- Modify: `public/index.html:2677-2679` (module-level Learn state variables)
- Modify: `public/index.html:2715-2769` (`renderLearnTab`)
- Modify: `public/index.html:2771-2792` (`runAnalysis`)

Add `learnAnalysisError` module-level variable, update `renderLearnTab` to show a persistent inline error state, update `runAnalysis` to handle the `analyzing: true` response, and add a `pollLearnStatus` function that polls `GET /api/intake/:pageId` until work completes or times out.

---

- [ ] **Step 3.1: Add `learnAnalysisError` next to the existing Learn state variables**

Find the block at line 2677:
```javascript
let learnCache = {}; // pageId -> { description, explanation }
let learnActiveTab = 'description';
let learnAnalyzing = false;
```

Add one line:
```javascript
let learnCache = {}; // pageId -> { description, explanation }
let learnActiveTab = 'description';
let learnAnalyzing = false;
let learnAnalysisError = null; // string | null — shown persistently inside the Learn panel body
let learnPollTimer = null;     // timer handle for active polling
```

---

- [ ] **Step 3.2: Update `renderLearnTab` to show the inline error state**

Find the block (around line 2760) that renders the "No analysis yet" empty state and buttons:
```javascript
  // Not yet analyzed — primary action is "Analyze both" (faster than two round-trips)
  const label = learnActiveTab === 'description' ? 'description' : 'concept explanation';
  body.innerHTML = `<div class="empty-msg">No analysis yet for this page.<br>Uses your local Gemma model if vision is available, otherwise falls back to Grok API.</div>` +
    `<button class="analyze-btn" id="analyzeBothBtn">Analyze description &amp; concepts</button>` +
    `<button class="analyze-btn secondary" id="analyzeBtn">Just analyze ${escapeHtml(label)}</button>`;

  $('analyzeBothBtn').onclick = () => runAnalysis(page.id, 'both');
  $('analyzeBtn').onclick = () => runAnalysis(page.id, learnActiveTab);
```

Replace with:
```javascript
  // Not yet analyzed
  const label = learnActiveTab === 'description' ? 'description' : 'concept explanation';
  const errorBlock = learnAnalysisError
    ? `<div class="learn-error-msg">${escapeHtml(learnAnalysisError)}</div>`
    : '';
  const helpText = learnAnalysisError
    ? '' // error block already explains the situation
    : `<div class="empty-msg">No analysis yet for this page.<br>Uses local Gemma if running on port 8080, Ollama Cloud, or Grok as fallback.</div>`;
  body.innerHTML = helpText + errorBlock +
    `<button class="analyze-btn" id="analyzeBothBtn">Analyze description &amp; concepts</button>` +
    `<button class="analyze-btn secondary" id="analyzeBtn">Just analyze ${escapeHtml(label)}</button>`;

  $('analyzeBothBtn').onclick = () => { learnAnalysisError = null; runAnalysis(page.id, 'both'); };
  $('analyzeBtn').onclick    = () => { learnAnalysisError = null; runAnalysis(page.id, learnActiveTab); };
```

---

- [ ] **Step 3.3: Add the `learn-error-msg` CSS style**

Find the `.learn-body .analyze-btn` style block (around line 336) and add directly below it:

```css
  .learn-body .learn-error-msg { background: #fdf0ef; border: 1px solid #e8b4b0; border-radius: 6px; padding: 10px 14px; font-size: 13px; color: #8b3a33; line-height: 1.5; margin: 12px 0 8px; }
```

---

- [ ] **Step 3.4: Rewrite `runAnalysis` to handle `analyzing: true` and launch polling**

Find `runAnalysis` starting at line 2771. Replace the entire function:

```javascript
async function runAnalysis(pageId, type) {
  learnAnalyzing = true;
  learnAnalysisError = null;
  renderLearnTab();

  try {
    const res = await fetch(`/api/analysis/${pageId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type }),
    });
    const data = await res.json();
    if (!res.ok) {
      learnAnalysisError = data.error || `Server error ${res.status}`;
      learnAnalyzing = false;
      renderLearnTab();
      return;
    }
    learnCache[pageId] = data;
    if (data.analyzing) {
      // Work kicked off; keep spinner, start polling
      renderLearnTab();
      pollLearnStatus(pageId);
    } else {
      learnAnalyzing = false;
      renderLearnTab();
    }
  } catch (err) {
    learnAnalysisError = 'Could not reach the server: ' + err.message;
    learnAnalyzing = false;
    renderLearnTab();
  }
}
```

---

- [ ] **Step 3.5: Add `pollLearnStatus` after `runAnalysis`**

Insert this function immediately after `runAnalysis`:

```javascript
function pollLearnStatus(pageId) {
  if (learnPollTimer) clearTimeout(learnPollTimer);
  const MAX_POLLS = 36;  // 36 × 5s = 3 minutes max
  let polls = 0;

  async function poll() {
    if (!$('learnPanel').classList.contains('visible')) return; // panel closed

    polls++;
    try {
      const statusRes = await fetch(`/api/intake/${pageId}`);
      const { analysisStatus } = await statusRes.json();
      const stillWorking = analysisStatus.inFlight?.classify ||
                           analysisStatus.inFlight?.intake  ||
                           analysisStatus.inFlight?.synthesis;
      const hasData = analysisStatus.description || analysisStatus.explanation;

      if (hasData) {
        // Fetch the actual content now that something is ready
        const freshRes = await fetch(`/api/analysis/${pageId}`);
        learnCache[pageId] = await freshRes.json();
        learnAnalyzing = false;
        learnAnalysisError = null;
        renderLearnTab();
        return; // done
      }

      if (!stillWorking) {
        // All in-flight work stopped but nothing appeared — failure
        learnAnalyzing = false;
        learnAnalysisError =
          'Analysis could not be completed. Check that llama-server is running on port 8080, ' +
          'or set OLLAMA_API_KEY / XAI_API_KEY for a cloud fallback.';
        renderLearnTab();
        return; // done
      }

      if (polls >= MAX_POLLS) {
        learnAnalyzing = false;
        learnAnalysisError = 'Analysis timed out after 3 minutes. The model server may be overloaded — try again.';
        renderLearnTab();
        return;
      }

      learnPollTimer = setTimeout(poll, 5000);
    } catch {
      learnAnalysisError = 'Lost connection to server while waiting for analysis.';
      learnAnalyzing = false;
      renderLearnTab();
    }
  }

  learnPollTimer = setTimeout(poll, 5000);
}
```

---

- [ ] **Step 3.6: Reset `learnAnalysisError` when the Learn panel is closed or a new page is loaded**

Find `closePanel` (line 2794) and update it:
```javascript
function closePanel(name) {
  $(name + 'Overlay').classList.remove('visible');
  $(name + 'Panel').classList.remove('visible');
  if (name === 'learn') {
    if (learnPollTimer) { clearTimeout(learnPollTimer); learnPollTimer = null; }
    learnAnalyzing = false;
    learnAnalysisError = null;
  }
}
```

Also clear on page navigation — find `renderCurrentPage` (line 1720) and add at the top:
```javascript
function renderCurrentPage() {
  // Reset learn-panel error when the user navigates to a different page
  if (!$('learnPanel').classList.contains('visible')) {
    learnAnalysisError = null;
  }
  // ... rest of function unchanged
```

---

- [ ] **Step 3.7: Manual verification**

1. Start the server without any API keys: `node server.js`
2. Open http://localhost:3000. Generate an image.
3. Open Learn panel → click "Analyze description & concepts".
4. **Expected:** spinner shows immediately (loading state), then after ~3 min timeout (or sooner if analysis fails), a red error box appears inside the Learn panel explaining the failure. Buttons reappear below it.
5. Now set `XAI_API_KEY` or `OLLAMA_API_KEY` and restart.
6. Click "Analyze description & concepts".
7. **Expected:** spinner shows, updates appear as analysis completes, final content renders without error.

- [ ] **Step 3.8: Commit**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
git add public/index.html
git commit -m "fix(C1,H2): Learn panel shows persistent inline error; polls intake status until analysis completes"
```

---

## Task 4: H1 — Right-column layout + drill dismiss — `public/index.html`

**Files:**
- Modify: `public/index.html:81` (`.source-right` CSS)
- Modify: `public/index.html:~130-138` (add `.source-right` panel rules)
- Modify: `public/index.html:1630-1653` (`drillResultPanel`)
- Modify: `public/index.html:2195-2219` (`bindClassifiedPanelActions`)

The right column uses `display: grid` (inherited from `.source-side`) with `grid-template-rows: minmax(0, 1fr)`. A second child (drill result panel) goes into an implicit auto row and is clipped. Fix by switching the right column to flexbox so both panels share the space predictably.

---

- [ ] **Step 4.1: Replace the `.source-right` CSS rule (line 81)**

Find:
```css
  .source-right { grid-template-rows: minmax(0, 1fr); }
```

Replace with:
```css
  .source-right { display: flex; flex-direction: column; gap: 12px; min-height: 0; overflow: hidden; }
```

---

- [ ] **Step 4.2: Add flex sizing rules for children of `.source-right`**

Find the comment block around line 130 (`/* Phase B classified panels */`) and insert these rules just before it:

```css
  /* Right-column panel flex sizing */
  .source-right .source-panel:not(.drill-result) { flex: 0 0 auto; max-height: 260px; overflow-y: auto; }
  .source-right .source-panel.drill-result { flex: 1 1 0; min-height: 120px; overflow-y: auto; }
```

---

- [ ] **Step 4.3: Add a dismiss button to `drillResultPanel` when there is an active drill**

Find `drillResultPanel` starting at line 1630. Replace the `return` for the non-empty case (around line 1649):

```javascript
  // OLD:
  return `<section class="source-panel drill-result">
    <h2>${escapeHtml(title)}</h2>
    <div class="drill-result-meta">${escapeHtml(meta)}</div>
    <div class="source-content">${body}</div>
  </section>`;

  // NEW:
  return `<section class="source-panel drill-result">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:6px;margin-bottom:4px;">
      <h2 style="margin:0;flex:1;min-width:0;">${escapeHtml(title)}</h2>
      <button type="button" data-clear-drill="1" title="Dismiss drill result"
        style="flex:0 0 auto;background:transparent;border:none;font-size:18px;cursor:pointer;color:#aaa;padding:0 0 0 4px;line-height:1;">&times;</button>
    </div>
    <div class="drill-result-meta">${escapeHtml(meta)}</div>
    <div class="source-content">${body}</div>
  </section>`;
```

---

- [ ] **Step 4.4: Wire the dismiss button in `bindClassifiedPanelActions`**

Find `bindClassifiedPanelActions` (around line 2195). Add inside the function, after the existing `root.querySelectorAll('[data-retry-intake]')` block:

```javascript
  // Drill result dismiss button
  root.querySelectorAll('[data-clear-drill]').forEach(btn => {
    btn.addEventListener('click', () => {
      delete state.activeDrillByParent[page.id];
      saveSession();
      renderCurrentPage();
    });
  });
```

---

- [ ] **Step 4.5: Manual verification**

1. Open http://localhost:3000. Generate an image.
2. Wait for classification to complete (left column populates).
3. Click "Drill" button → click a region → confirm the drill. Wait for the drill result.
4. **Expected:** classified info panel appears in upper portion of the right column (~260px max height, scrollable if long). Drill result appears below it, filling remaining height. Both are visible simultaneously without scrolling the outer page.
5. Click the `×` button on the drill result panel.
6. **Expected:** drill result panel reverts to the "Click a red drill-in circle" empty message. Classified info panel expands to fill the full right column.

- [ ] **Step 4.6: Commit**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
git add public/index.html
git commit -m "fix(H1): right column uses flexbox so classified panel and drill result are both visible; add drill dismiss button"
```

---

## Task 5: H3 — Vault search error distinction — `vault.js` + `server.js` + `public/index.html`

**Files:**
- Modify: `vault.js:44-69` (`searchVault`)
- Modify: `server.js:1624-1633` (`POST /api/vault-search`)
- Modify: `server.js:1735-1744` (vault search inside `POST /api/page`)
- Modify: `public/index.html:~519` (add `vaultErrorCache`)
- Modify: `public/index.html:2134-2149` (vault panel render in `renderCurrentPage`)

Currently a CLI error and zero results look identical. This adds an `error` field from `searchVault` through to the frontend.

---

- [ ] **Step 5.1: Update `searchVault` in `vault.js` to return the error field**

Find the entire `searchVault` function. Replace the `catch` block:

```javascript
  } catch (err) {
    console.error('[vault] searchVault failed:', err.message);
    return { hits: [], error: err.message };
  }
```

And update the success path to include `error: null`:
```javascript
    return { hits: applyFilters(hits, filters), error: null };
```

---

- [ ] **Step 5.2: Update `POST /api/vault-search` to forward the error flag**

Find the handler at line 1624:
```javascript
app.post('/api/vault-search', async (req, res) => {
  const { parentId, intent, vaultFilters: reqVaultFilters } = req.body;
  if (!intent?.trim()) return res.json({ hits: [] });
  const filters = ...;
  const parentClassified = parentId ? loadParentClassified(parentId) : null;
  const query = buildVaultQuery(intent, parentClassified).trim();
  if (!query) return res.json({ hits: [] });
  const result = await searchVault(query, { filters });
  res.json({ hits: result.hits });
});
```

Replace the last two lines:
```javascript
  const result = await searchVault(query, { filters });
  res.json({ hits: result.hits, vaultError: result.error ?? null });
});
```

---

- [ ] **Step 5.3: Update vault search inside `POST /api/page` to log the error**

Find the vault search block inside `POST /api/page` around line 1735:
```javascript
const vaultResult = await searchVault(vaultQuery, { filters: vaultFilters });
vaultHits = vaultResult.hits;
vaultContext = buildVaultContextBlock(vaultHits);
```

Replace with:
```javascript
const vaultResult = await searchVault(vaultQuery, { filters: vaultFilters });
if (vaultResult.error) console.warn('[vault] page-gen search failed:', vaultResult.error);
vaultHits = vaultResult.hits;
vaultContext = buildVaultContextBlock(vaultHits);
```

Also update the response to include `vaultError` when generating from `POST /api/page`:
```javascript
// Find (line ~1763): return { page, vaultHits };
// Replace with:
return { page, vaultHits, vaultError: vaultResult.error ?? null };
```

And in the outer handler where `genVaultHits` is destructured (around line 1883):
```javascript
// Find:
const { page: genPage, vaultHits: genVaultHits } = genResult?.page ? genResult : { page: genResult, vaultHits: [] };
res.json({ page: genPage, vaultHits: genVaultHits ?? [] });
// Replace with:
const { page: genPage, vaultHits: genVaultHits, vaultError: genVaultError } = genResult?.page
  ? genResult
  : { page: genResult, vaultHits: [], vaultError: null };
res.json({ page: genPage, vaultHits: genVaultHits ?? [], vaultError: genVaultError ?? null });
```

---

- [ ] **Step 5.4: Add `vaultErrorCache` module-level variable in `public/index.html`**

Find the cache variables block around line 519:
```javascript
const vaultHitsCache = new Map(); // pageId -> hits[]
const vaultSearchInFlight = new Set(); // pageIds with active vault search
```

Add:
```javascript
const vaultErrorCache = new Map(); // pageId -> error string | null
```

---

- [ ] **Step 5.5: Capture vault error when a drill-in page is generated**

Find in `public/index.html` where `POST /api/page` response is handled (around line 2257 in `generateChildPage`). Locate:
```javascript
const { page, vaultHits } = await res.json();
```

Replace with:
```javascript
const { page, vaultHits, vaultError } = await res.json();
if (vaultError) vaultErrorCache.set(page.id, vaultError);
```

---

- [ ] **Step 5.6: Capture vault error from the lazy vault search path**

Find the lazy vault search block in `renderCurrentPage` (around line 2142):
```javascript
}).then(r => r.json()).then(({ hits }) => {
  vaultSearchInFlight.delete(page.id);
  if (hits?.length) { vaultHitsCache.set(page.id, hits); renderCurrentPage(); }
  else vaultHitsCache.set(page.id, []); // mark searched, no results
}).catch(() => vaultSearchInFlight.delete(page.id));
```

Replace with:
```javascript
}).then(r => r.json()).then(({ hits, vaultError }) => {
  vaultSearchInFlight.delete(page.id);
  if (vaultError) vaultErrorCache.set(page.id, vaultError);
  vaultHitsCache.set(page.id, hits?.length ? hits : []);
  if (hits?.length || vaultError) renderCurrentPage();
}).catch(() => vaultSearchInFlight.delete(page.id));
```

---

- [ ] **Step 5.7: Update the vault panel render to show errors**

Find where `renderVaultPanel(currentVaultHits)` is called (around line 2138):
```javascript
if (currentVaultHits?.length) {
  leftParts.push(renderVaultPanel(currentVaultHits));
} else if (!vaultHitsCache.has(page.id) && !vaultSearchInFlight.has(page.id)) {
  // Lazy search for cached drill pages ...
```

Replace:
```javascript
const currentVaultError = vaultErrorCache.get(page.id) || null;
if (currentVaultHits?.length) {
  leftParts.push(renderVaultPanel(currentVaultHits));
} else if (currentVaultError) {
  leftParts.push(`<section class="source-panel">
    <h2>Vault Sources</h2>
    <div class="empty-drill" style="color:#8b3a33;font-style:normal;">
      Vault search unavailable — is <code>qmd</code> on PATH?<br>
      <span style="font-size:11px;color:#aaa;">${escapeHtml(currentVaultError.slice(0, 120))}</span>
    </div>
  </section>`);
} else if (!vaultHitsCache.has(page.id) && !vaultSearchInFlight.has(page.id)) {
  // Lazy search for cached drill pages ...
```

---

- [ ] **Step 5.8: Manual verification**

1. Stop `qmd` from being reachable: `export PATH_BACKUP=$PATH && export PATH=$(echo $PATH | tr ':' '\n' | grep -v qmd | tr '\n' ':')`
2. Open http://localhost:3000. Enable Vault toggle. Generate an image. Click a region to drill.
3. **Expected:** Left column shows "Vault Sources" panel with "Vault search unavailable — is `qmd` on PATH?" message in red.
4. Restore PATH. Retry.
5. **Expected:** Vault Sources panel shows hits or an empty state (no error).

- [ ] **Step 5.9: Commit**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
git add vault.js server.js public/index.html
git commit -m "fix(H3): distinguish vault search errors from empty results; surface qmd failures in left panel"
```

---

## Self-Review

**Spec coverage:**
- C1 (silent error cycle): covered by T3 (inline error) + T2 (fast POST) ✓
- C2 (synthesis gate): covered by T2 step 2.1 ✓
- Ollama fallback (gemma4:31b-cloud): covered by T1 ✓
- H1 (right-column layout + dismiss): covered by T4 ✓
- H2 (Learn panel polling): covered by T3 (pollLearnStatus) ✓
- H3 (vault error distinction): covered by T5 ✓

**Placeholder scan:** All code blocks are complete. No "TBD" or "similar to" language.

**Type consistency:**
- `learnAnalysisError` is `string | null` throughout (set, checked, cleared consistently)
- `learnPollTimer` is `ReturnType<typeof setTimeout> | null`
- `cloudFallback()` returns `[string, string] | null` — all callers destructure with `const [fbProvider, fbModel] = fb`
- `vaultErrorCache` is `Map<string, string>` — get returns `string | undefined`, coerced to `null` with `|| null`
- `data-clear-drill` attribute is used in `drillResultPanel` (T4.3) and queried in `bindClassifiedPanelActions` (T4.4) ✓

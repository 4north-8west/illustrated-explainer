# Mode Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Phase 3 mode editor — runtime CRUD over `modes/<id>.json` files plus a structured-form UI in the Settings panel — so users can create, edit, duplicate, delete, and reset modes without touching files or restarting the server.

**Architecture:** New API surface (`GET /api/modes/raw`, `POST /api/modes/:id`, `DELETE /api/modes/:id`, `POST /api/modes/:id/reset`). All writes confined to `modes/`. `FALLBACK_MODES` is the always-available floor — baked-in modes can be overlaid via on-disk files but are never destroyable. Validation via ajv (already a dep), with `additionalProperties: false`, length caps on every string, regex-pinned id, and a positive `{{query}}` check on `firstPageTemplate`. After each mutation: `reloadModes()` reassigns `MODES` and `VALID_MODES` — all call sites already read the `let`s on every request so changes take effect immediately. Frontend grows a "Modes" subsection inside the existing Settings panel; no new HTML/JS files.

**Tech Stack:** Node.js/Express (`server.js`), vanilla JS + inline CSS (`public/index.html`), ajv 8 (already in `package.json`).

**Design spec:** `docs/superpowers/specs/2026-06-13-mode-editor-design.md`

---

## File map

| File | What changes |
|------|-------------|
| `server.js` | `MODE_SCHEMA` constant + ajv `validateMode`; `sanitizeModeId`, `isBakedInMode`, `reloadModes`, `editableMode` helpers; `inferModeFromQuery` extended for `inferKeywords`; four new routes (`GET /api/modes/raw`, `POST /api/modes/:id`, `DELETE /api/modes/:id`, `POST /api/modes/:id/reset`) — all mutating routes mount their own `express.json({ limit: '64kb' })` because the global limit is `50mb`. |
| `public/index.html` | New `state.editableModes`; `renderModeList`, `openModeEditor`, `submitModeForm`, `deleteMode`, `resetMode`, `reconcileActiveMode` functions; Modes subsection markup + form panel inside the existing Settings panel; CSS for the new subsection. |

---

## Task 1: Server — ajv schema + helpers + `reloadModes` — `server.js`

**Files:**
- Modify: `server.js:1-15` (imports — add `ajv`)
- Modify: `server.js:362-420` (add helpers after `loadModes` block)

---

- [ ] **Step 1.1: Add the ajv import**

Find the imports near the top of `server.js` (lines 1–15). Add an ajv import after the existing `express`/`fs` imports:

```javascript
import Ajv from 'ajv';
```

If the file uses CommonJS `require`, use:

```javascript
const Ajv = require('ajv').default || require('ajv');
```

Match whichever style the file already uses (the file is ESM per `package.json: "type": "module"`, so prefer the `import` form).

---

- [ ] **Step 1.2: Add `MODE_SCHEMA` and the compiled `validateMode` near `FALLBACK_MODES`**

Find `const FALLBACK_MODES = {` (around line 252). Immediately **before** that constant, insert:

```javascript
const MODE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'label', 'tagLabel', 'style', 'firstPageTemplate', 'childPageTemplate', 'modeLabelForPrompt'],
  properties: {
    id:                  { type: 'string', pattern: '^[a-z][a-z0-9_]{0,30}$' },
    label:               { type: 'string', minLength: 1, maxLength: 60 },
    tagLabel:            { type: 'string', minLength: 1, maxLength: 30 },
    placeholder:         { type: 'string', maxLength: 200 },
    description:         { type: 'string', maxLength: 500 },
    style:               { type: 'string', minLength: 1, maxLength: 8000 },
    firstPageTemplate:   { type: 'string', minLength: 1, maxLength: 8000 },
    childPageTemplate:   { type: 'string', minLength: 1, maxLength: 8000 },
    modeLabelForPrompt:  { type: 'string', minLength: 1, maxLength: 60 },
    inferKeywords: {
      type: 'array',
      maxItems: 40,
      items: { type: 'string', minLength: 2, maxLength: 40 },
    },
  },
};

const ajv = new Ajv({ allErrors: true });
const validateMode = ajv.compile(MODE_SCHEMA);
```

---

- [ ] **Step 1.3: Add `sanitizeModeId`, `isBakedInMode`, `reloadModes`, `editableMode` helpers**

Find `let MODES = loadModes();` (around line 416). Immediately **after** that line and its sibling `let VALID_MODES = Object.keys(MODES);`, insert:

```javascript
function sanitizeModeId(id) {
  return typeof id === 'string' && /^[a-z][a-z0-9_]{0,30}$/.test(id);
}

function isBakedInMode(id) {
  return Object.prototype.hasOwnProperty.call(FALLBACK_MODES, id);
}

function modeFilePath(id) {
  const filePath = path.join(MODES_DIR, `${id}.json`);
  const resolved = path.resolve(filePath);
  if (path.dirname(resolved) !== path.resolve(MODES_DIR)) {
    throw new Error(`Refusing to operate outside modes/ for id: ${id}`);
  }
  return resolved;
}

function modeHasOverlay(id) {
  try {
    return fs.existsSync(modeFilePath(id));
  } catch {
    return false;
  }
}

function reloadModes() {
  MODES = loadModes();
  VALID_MODES = Object.keys(MODES);
}

function editableMode(mode) {
  return {
    id: mode.id,
    label: mode.label,
    tagLabel: mode.tagLabel,
    placeholder: mode.placeholder,
    description: mode.description,
    style: mode.style,
    firstPageTemplate: mode.firstPageTemplate,
    childPageTemplate: mode.childPageTemplate,
    modeLabelForPrompt: mode.modeLabelForPrompt,
    inferKeywords: Array.isArray(mode.inferKeywords) ? mode.inferKeywords : [],
    bakedIn: isBakedInMode(mode.id),
    hasOverlay: modeHasOverlay(mode.id),
  };
}
```

---

- [ ] **Step 1.4: Confirm `normalizeModeConfig` preserves `inferKeywords`**

Find `function normalizeModeConfig(mode, fallbackId = null) {` (around line 368). Look at the returned object (the `return {` block ending around line 388). Verify it includes `inferKeywords`. If it does NOT, add this line just before the closing `};`:

```javascript
    inferKeywords: Array.isArray(mode.inferKeywords) ? mode.inferKeywords : [],
```

If you can't find the return block, run:

```bash
grep -n "inferKeywords" server.js
```

Expected after this step: `normalizeModeConfig` returns an object whose `inferKeywords` is either the original array or `[]`.

---

- [ ] **Step 1.5: Smoke test — server still starts**

Run:

```bash
cd ~/Documents/my-agent-team/illustrated-explainer && node -e "import('./server.js').then(()=>console.log('loaded'),e=>{console.error(e);process.exit(1)})" &
sleep 2
curl -s http://localhost:3000/api/modes | head -c 200
kill %1 2>/dev/null
```

Expected: JSON response listing the four baked-in modes. If the import fails because the file calls `app.listen` at import time, replace the test with starting the server normally:

```bash
cd ~/Documents/my-agent-team/illustrated-explainer && (npm start &) && sleep 3 && curl -s http://localhost:3000/api/modes | head -c 200 && pkill -f "node server.js"
```

---

- [ ] **Step 1.6: Commit**

```bash
cd ~/Documents/my-agent-team/illustrated-explainer
git add server.js
git commit -m "feat(modes): MODE_SCHEMA + ajv validator + sanitize/reload/editable helpers"
```

---

## Task 2: Server — `GET /api/modes/raw` — `server.js`

**Files:**
- Modify: `server.js:1280-1290` (insert new route immediately after `GET /api/modes`)

---

- [ ] **Step 2.1: Add the route**

Find `app.get('/api/modes', (_req, res) => {` (around line 1282). Immediately **after** the closing `});` of that handler, insert:

```javascript
app.get('/api/modes/raw', (_req, res) => {
  res.json({
    modes: VALID_MODES.map(id => editableMode(MODES[id])),
    bakedInIds: Object.keys(FALLBACK_MODES),
  });
});
```

---

- [ ] **Step 2.2: Manual smoke test**

Start the server (`npm start`), then:

```bash
curl -s http://localhost:3000/api/modes/raw | python3 -m json.tool | head -40
```

Expected: array of mode objects each with `bakedIn`, `hasOverlay`, `inferKeywords`, full templates. `bakedInIds` should list the four baked-in mode ids.

---

- [ ] **Step 2.3: Commit**

```bash
git add server.js
git commit -m "feat(modes): GET /api/modes/raw — editable shape for the editor UI"
```

---

## Task 3: Server — `POST /api/modes/:id` (create or update) — `server.js`

**Files:**
- Modify: `server.js` (insert after the `GET /api/modes/raw` route from Task 2)

---

- [ ] **Step 3.1: Add the route with a route-local body-parser limit**

Immediately after the `GET /api/modes/raw` route, insert:

```javascript
const modeJsonParser = express.json({ limit: '64kb' });

app.post('/api/modes/:id', modeJsonParser, (req, res) => {
  const { id } = req.params;
  if (!sanitizeModeId(id)) {
    return res.status(400).json({ error: 'Invalid mode id' });
  }
  const payload = req.body && typeof req.body === 'object' ? req.body : {};
  if (payload.id && payload.id !== id) {
    return res.status(400).json({ error: 'Path id and body id must match' });
  }
  payload.id = id;

  if (!validateMode(payload)) {
    return res.status(400).json({
      error: 'Validation failed',
      details: validateMode.errors,
    });
  }
  if (!payload.firstPageTemplate.includes('{{query}}')) {
    return res.status(400).json({
      error: 'firstPageTemplate must contain {{query}}',
    });
  }

  let filePath;
  try {
    filePath = modeFilePath(id);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }

  try {
    fs.writeFileSync(filePath, JSON.stringify(payload, null, 2) + '\n', 'utf8');
    reloadModes();
  } catch (err) {
    return res.status(500).json({ error: `Failed to write mode: ${err.message}` });
  }

  res.json({
    modes: VALID_MODES.map(id => publicMode(MODES[id])),
    defaultMode: VALID_MODES.includes('illustration') ? 'illustration' : VALID_MODES[0],
    saved: editableMode(MODES[id]),
  });
});
```

---

- [ ] **Step 3.2: Manual happy-path test — create a custom mode**

Start the server. Then:

```bash
curl -s -X POST http://localhost:3000/api/modes/lab_diagrams \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "lab_diagrams",
    "label": "Lab Diagrams",
    "tagLabel": "lab",
    "placeholder": "Type a setup...",
    "description": "Bench-top lab apparatus diagrams.",
    "style": "Crisp ink line drawing, clean labels.",
    "firstPageTemplate": "{{style}}\n\nSubject: {{query}}\n\nA single labeled lab apparatus diagram.",
    "childPageTemplate": "{{style}}\n\nZoom into the marked region.",
    "modeLabelForPrompt": "lab diagram",
    "inferKeywords": ["microscope", "petri", "titration"]
  }' | python3 -m json.tool | head -30
```

Expected: 200 response with the new mode in `modes` and a `saved` object. Verify the file exists:

```bash
ls modes/lab_diagrams.json && head -5 modes/lab_diagrams.json
```

---

- [ ] **Step 3.3: Negative tests**

Run these and confirm each returns 400:

```bash
# bad id
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:3000/api/modes/bad-id \
  -H 'Content-Type: application/json' -d '{}'

# id mismatch
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:3000/api/modes/lab_diagrams \
  -H 'Content-Type: application/json' -d '{"id":"other","label":"x"}'

# missing required field
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:3000/api/modes/lab_diagrams \
  -H 'Content-Type: application/json' -d '{}'

# additionalProperties
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:3000/api/modes/lab_diagrams \
  -H 'Content-Type: application/json' \
  -d '{"id":"lab_diagrams","label":"x","tagLabel":"x","style":"x","firstPageTemplate":"{{query}}","childPageTemplate":"x","modeLabelForPrompt":"x","evil":true}'

# firstPageTemplate missing {{query}}
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:3000/api/modes/lab_diagrams \
  -H 'Content-Type: application/json' \
  -d '{"id":"lab_diagrams","label":"x","tagLabel":"x","style":"x","firstPageTemplate":"no placeholder","childPageTemplate":"x","modeLabelForPrompt":"x"}'
```

Expected: all five print `400`.

---

- [ ] **Step 3.4: Path traversal smoke test**

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST 'http://localhost:3000/api/modes/..%2Fetc%2Fpasswd' \
  -H 'Content-Type: application/json' -d '{}'
```

Expected: `400`. Also verify `modes/` doesn't contain anything unexpected:

```bash
ls modes/
```

Should list only `*.json` files matching the sanitize regex.

---

- [ ] **Step 3.5: Commit**

```bash
git add server.js
git commit -m "feat(modes): POST /api/modes/:id with ajv validation + 64kb route-local limit"
```

---

## Task 4: Server — `DELETE /api/modes/:id` and `POST /api/modes/:id/reset` — `server.js`

**Files:**
- Modify: `server.js` (insert after the `POST /api/modes/:id` route from Task 3)

---

- [ ] **Step 4.1: Add the DELETE route**

Immediately after the `POST /api/modes/:id` route, insert:

```javascript
app.delete('/api/modes/:id', (req, res) => {
  const { id } = req.params;
  if (!sanitizeModeId(id)) {
    return res.status(400).json({ error: 'Invalid mode id' });
  }
  if (isBakedInMode(id)) {
    return res.status(400).json({ error: 'Cannot delete a baked-in mode; use reset instead' });
  }
  let filePath;
  try {
    filePath = modeFilePath(id);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'Mode not found' });
  }
  try {
    fs.unlinkSync(filePath);
    reloadModes();
  } catch (err) {
    return res.status(500).json({ error: `Failed to delete mode: ${err.message}` });
  }
  res.json({
    modes: VALID_MODES.map(id => publicMode(MODES[id])),
    defaultMode: VALID_MODES.includes('illustration') ? 'illustration' : VALID_MODES[0],
  });
});
```

---

- [ ] **Step 4.2: Add the reset route**

Immediately after the DELETE route, insert:

```javascript
app.post('/api/modes/:id/reset', (req, res) => {
  const { id } = req.params;
  if (!sanitizeModeId(id)) {
    return res.status(400).json({ error: 'Invalid mode id' });
  }
  if (!isBakedInMode(id)) {
    return res.status(400).json({ error: 'Only baked-in modes can be reset' });
  }
  let filePath;
  try {
    filePath = modeFilePath(id);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }
  if (fs.existsSync(filePath)) {
    try {
      fs.unlinkSync(filePath);
    } catch (err) {
      return res.status(500).json({ error: `Failed to reset mode: ${err.message}` });
    }
  }
  reloadModes();
  res.json({
    modes: VALID_MODES.map(id => publicMode(MODES[id])),
    defaultMode: VALID_MODES.includes('illustration') ? 'illustration' : VALID_MODES[0],
    saved: editableMode(MODES[id]),
  });
});
```

---

- [ ] **Step 4.3: Manual tests — delete the custom mode + reset a baked-in**

Start the server. First create a fresh custom mode (see Task 3 Step 3.2). Then:

```bash
# delete the custom mode
curl -s -X DELETE http://localhost:3000/api/modes/lab_diagrams | python3 -m json.tool | head -10
ls modes/lab_diagrams.json 2>&1   # expected: No such file

# try deleting a baked-in mode -> 400
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE http://localhost:3000/api/modes/illustration

# overlay then reset a baked-in mode
curl -s -X POST http://localhost:3000/api/modes/illustration \
  -H 'Content-Type: application/json' \
  -d "$(cat modes/illustration.json | python3 -c 'import json,sys; m=json.load(sys.stdin); m["label"]="Watercolor Test"; print(json.dumps(m))')" >/dev/null
test -f modes/illustration.json && echo "overlay present"
curl -s -X POST http://localhost:3000/api/modes/illustration/reset >/dev/null
test -f modes/illustration.json || echo "overlay removed"

# try resetting a non-baked-in mode -> 400
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:3000/api/modes/lab_diagrams/reset
```

Expected:
- DELETE on custom: 200 + file removed
- DELETE on baked-in: 400
- POST overlay → file appears, label is "Watercolor Test"
- POST reset → file removed, `GET /api/modes` shows "Illustration" again
- POST reset on non-baked-in: 400

---

- [ ] **Step 4.4: Commit**

```bash
git add server.js
git commit -m "feat(modes): DELETE /api/modes/:id and POST /api/modes/:id/reset"
```

---

## Task 5: Server — extend `inferModeFromQuery` for user `inferKeywords` — `server.js`

**Files:**
- Modify: `server.js:419-436` (`inferModeFromQuery`)

---

- [ ] **Step 5.1: Add a regex-escape helper**

Find `function inferModeFromQuery(query) {` (around line 419). Immediately **before** that function, insert:

```javascript
function escapeRegExp(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
```

---

- [ ] **Step 5.2: Append the user-keyword loop**

Inside `inferModeFromQuery`, find the final `return hasMode('illustration') ? 'illustration' : VALID_MODES[0];` line. Immediately **before** that return, insert:

```javascript
  for (const id of VALID_MODES) {
    if (isBakedInMode(id)) continue;
    const kw = MODES[id]?.inferKeywords;
    if (!Array.isArray(kw) || kw.length === 0) continue;
    const alternation = kw.map(escapeRegExp).join('|');
    try {
      const re = new RegExp(`\\b(${alternation})\\b`, 'i');
      if (re.test(text)) return id;
    } catch {
      // skip a mode whose keywords produce an invalid regex
    }
  }
```

The `text` variable comes from `const text = normalize(query);` at the top of the function — leave that line alone.

---

- [ ] **Step 5.3: Smoke test**

Start the server. With Mode = `auto`:

```bash
curl -s -X POST http://localhost:3000/api/modes/lab_diagrams \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "lab_diagrams",
    "label": "Lab Diagrams",
    "tagLabel": "lab",
    "style": "Crisp ink lines.",
    "firstPageTemplate": "{{style}} {{query}}",
    "childPageTemplate": "{{style}} zoom",
    "modeLabelForPrompt": "lab diagram",
    "inferKeywords": ["microscope", "petri", "titration"]
  }' >/dev/null

# Now hit the intake route with a query that should match
curl -s -X POST http://localhost:3000/api/intake \
  -H 'Content-Type: application/json' \
  -d '{"query":"microscope dish bacteria","mode":"auto"}' | python3 -m json.tool | grep -E '"mode"|"resolvedMode"' | head
```

Expected: response shows `mode: "lab_diagrams"` (or whatever field intake uses). If you're not sure which field carries the resolved mode, add a temporary `console.log` at the `inferModeFromQuery` call site instead and watch the server logs.

Clean up:

```bash
curl -s -X DELETE http://localhost:3000/api/modes/lab_diagrams >/dev/null
```

---

- [ ] **Step 5.4: Commit**

```bash
git add server.js
git commit -m "feat(modes): inferModeFromQuery picks up user-defined inferKeywords"
```

---

## Task 6: Frontend — Mode list inside Settings panel — `public/index.html`

**Files:**
- Modify: `public/index.html` (state init around line 507, settings panel markup, new render function)

---

- [ ] **Step 6.1: Extend state with `editableModes`**

Find `state = { mode: 'auto', modes: [], ... }` (around line 507). Add `editableModes: [],` immediately after `modes: [],`:

```javascript
state = {
  mode: 'auto',
  modes: [],
  editableModes: [],
  // ...other existing keys...
};
```

---

- [ ] **Step 6.2: Add a `loadEditableModes` fetcher and a `renderModeList` function**

Find the function that currently fetches `/api/modes` on startup (search for `'/api/modes'` in `public/index.html`). Near it, add two new functions:

```javascript
async function loadEditableModes() {
  try {
    const res = await fetch('/api/modes/raw');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.editableModes = data.modes || [];
    renderModeList();
  } catch (err) {
    console.warn('loadEditableModes failed:', err);
  }
}

function renderModeList() {
  const container = document.getElementById('modeListContainer');
  if (!container) return;
  container.innerHTML = '';
  for (const m of state.editableModes) {
    const row = document.createElement('div');
    row.className = 'mode-row';
    const label = document.createElement('span');
    label.className = 'mode-row-label';
    label.textContent = m.label;
    const tag = document.createElement('span');
    tag.className = 'mode-row-tag';
    tag.textContent = m.bakedIn ? 'baked-in' : 'custom';
    const editBtn = document.createElement('button');
    editBtn.className = 'mode-row-btn';
    editBtn.textContent = 'Edit';
    editBtn.addEventListener('click', () => openModeEditor(m.id));
    row.append(label, tag, editBtn);
    if (m.bakedIn && m.hasOverlay) {
      const resetBtn = document.createElement('button');
      resetBtn.className = 'mode-row-btn';
      resetBtn.textContent = 'Reset';
      resetBtn.addEventListener('click', () => resetMode(m.id));
      row.append(resetBtn);
    }
    if (!m.bakedIn) {
      const delBtn = document.createElement('button');
      delBtn.className = 'mode-row-btn danger';
      delBtn.textContent = 'Delete';
      delBtn.addEventListener('click', () => deleteMode(m.id));
      row.append(delBtn);
    }
    container.append(row);
  }
}
```

---

- [ ] **Step 6.3: Add the Modes subsection markup inside the Settings panel**

Find the Settings panel markup (search for `id="settingsPanel"` or `Settings` heading; should be near the same area as the model dropdowns). At the **bottom** of the Settings panel, before its closing tag, insert:

```html
<section class="settings-section" id="modesSection">
  <header class="settings-section-header">
    <h3>Modes</h3>
    <button type="button" id="newModeBtn" class="settings-action-btn">+ New Mode</button>
  </header>
  <div id="modeListContainer" class="mode-list"></div>
  <div id="modeEditorPanel" class="mode-editor-panel" hidden></div>
</section>
```

Then add this CSS in the existing `<style>` block (anywhere near the other settings-panel CSS):

```css
.settings-section { margin-top: 18px; padding-top: 14px; border-top: 1px solid #e8e0d0; }
.settings-section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.settings-section-header h3 { font-size: 14px; margin: 0; color: #6b5a3e; }
.settings-action-btn { padding: 4px 12px; background: #6b5a3e; color: #fff; border: none; border-radius: 6px; font-size: 12px; cursor: pointer; font-family: inherit; }
.settings-action-btn:hover { background: #564a32; }
.mode-list { display: flex; flex-direction: column; gap: 6px; }
.mode-row { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: #faf3e3; border-radius: 6px; }
.mode-row-label { flex: 1; font-size: 13px; color: #4a3f2a; }
.mode-row-tag { font-size: 10px; padding: 1px 7px; border-radius: 3px; background: #e8e0d0; color: #6b5a3e; }
.mode-row-btn { padding: 3px 10px; background: #fff; border: 1px solid #d6cfc7; border-radius: 4px; font-size: 11px; cursor: pointer; font-family: inherit; color: #555; }
.mode-row-btn:hover { background: #f5ebd3; border-color: #b8a06f; }
.mode-row-btn.danger { color: #a02020; border-color: #d8b0b0; }
.mode-row-btn.danger:hover { background: #fbe6e6; }
.mode-editor-panel { margin-top: 12px; padding: 12px; background: #fff; border: 1px solid #d6cfc7; border-radius: 6px; }
.mode-editor-panel label { display: block; font-size: 11px; color: #6b5a3e; margin-top: 8px; margin-bottom: 3px; }
.mode-editor-panel input[type="text"],
.mode-editor-panel textarea { width: 100%; padding: 6px 8px; border: 1px solid #d6cfc7; border-radius: 4px; font-family: inherit; font-size: 12px; box-sizing: border-box; }
.mode-editor-panel textarea { font-family: ui-monospace, monospace; min-height: 80px; resize: vertical; }
.mode-editor-panel .editor-actions { margin-top: 12px; display: flex; gap: 8px; justify-content: flex-end; }
.mode-editor-panel .field-error { color: #a02020; font-size: 11px; margin-top: 2px; }
```

---

- [ ] **Step 6.4: Wire the `+ New Mode` button and the initial load**

Find the function that opens the Settings panel (search for `openSettings` or similar). At the end of that function, add:

```javascript
  loadEditableModes();
  const newBtn = document.getElementById('newModeBtn');
  if (newBtn && !newBtn.dataset.bound) {
    newBtn.dataset.bound = '1';
    newBtn.addEventListener('click', () => openModeEditor());
  }
```

---

- [ ] **Step 6.5: Smoke test**

Start the server. Open `http://localhost:3000`. Click the Settings button. Confirm a "Modes" subsection appears with the four baked-in modes listed and `+ New Mode` in the header. Edit and Delete buttons should render per the bakedIn rules (no Delete on baked-in, no Reset unless overlay exists).

---

- [ ] **Step 6.6: Commit**

```bash
git add public/index.html
git commit -m "feat(modes): Modes subsection + list rendering in Settings panel"
```

---

## Task 7: Frontend — Mode editor form, save, delete, reset, reconcile — `public/index.html`

**Files:**
- Modify: `public/index.html` (add `openModeEditor`, `submitModeForm`, `deleteMode`, `resetMode`, `reconcileActiveMode` near the functions added in Task 6)

---

- [ ] **Step 7.1: Add `openModeEditor`**

Add this function near `renderModeList` from Task 6:

```javascript
function openModeEditor(id) {
  const panel = document.getElementById('modeEditorPanel');
  if (!panel) return;
  const existing = id ? state.editableModes.find(m => m.id === id) : null;
  const isNew = !existing;
  const m = existing || {
    id: '', label: '', tagLabel: '', placeholder: '', description: '',
    style: '', firstPageTemplate: '', childPageTemplate: '',
    modeLabelForPrompt: '', inferKeywords: [], bakedIn: false, hasOverlay: false,
  };
  panel.hidden = false;
  panel.innerHTML = `
    <label>id <input type="text" id="me-id" value="${escapeHtml(m.id)}" ${isNew ? '' : 'readonly'} placeholder="lower_snake_case"></label>
    <label>label <input type="text" id="me-label" value="${escapeHtml(m.label)}"></label>
    <label>tagLabel <input type="text" id="me-tagLabel" value="${escapeHtml(m.tagLabel)}"></label>
    <label>placeholder <input type="text" id="me-placeholder" value="${escapeHtml(m.placeholder || '')}"></label>
    <label>description <input type="text" id="me-description" value="${escapeHtml(m.description || '')}"></label>
    <label>style <textarea id="me-style">${escapeHtml(m.style)}</textarea></label>
    <label>firstPageTemplate (must contain {{query}}) <textarea id="me-firstPageTemplate">${escapeHtml(m.firstPageTemplate)}</textarea></label>
    <label>childPageTemplate <textarea id="me-childPageTemplate">${escapeHtml(m.childPageTemplate)}</textarea></label>
    <label>modeLabelForPrompt <input type="text" id="me-modeLabelForPrompt" value="${escapeHtml(m.modeLabelForPrompt)}"></label>
    <label>inferKeywords (comma-separated, optional) <input type="text" id="me-inferKeywords" value="${escapeHtml((m.inferKeywords || []).join(', '))}"></label>
    <div class="field-error" id="me-error" hidden></div>
    <div class="editor-actions">
      <button type="button" id="me-cancel" class="mode-row-btn">Cancel</button>
      <button type="button" id="me-save" class="settings-action-btn">Save</button>
    </div>
  `;
  document.getElementById('me-cancel').addEventListener('click', () => { panel.hidden = true; panel.innerHTML = ''; });
  document.getElementById('me-save').addEventListener('click', submitModeForm);
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
}
```

---

- [ ] **Step 7.2: Add `submitModeForm`**

```javascript
async function submitModeForm() {
  const errorEl = document.getElementById('me-error');
  errorEl.hidden = true;
  errorEl.textContent = '';

  const idEl = document.getElementById('me-id');
  const id = (idEl.value || '').trim();
  if (!/^[a-z][a-z0-9_]{0,30}$/.test(id)) {
    errorEl.textContent = 'id must be lower_snake_case, start with a letter, max 31 chars';
    errorEl.hidden = false;
    return;
  }

  const payload = {
    id,
    label: document.getElementById('me-label').value.trim(),
    tagLabel: document.getElementById('me-tagLabel').value.trim(),
    placeholder: document.getElementById('me-placeholder').value,
    description: document.getElementById('me-description').value,
    style: document.getElementById('me-style').value,
    firstPageTemplate: document.getElementById('me-firstPageTemplate').value,
    childPageTemplate: document.getElementById('me-childPageTemplate').value,
    modeLabelForPrompt: document.getElementById('me-modeLabelForPrompt').value.trim(),
    inferKeywords: document.getElementById('me-inferKeywords').value
      .split(',')
      .map(s => s.trim())
      .filter(Boolean),
  };
  if (payload.inferKeywords.length === 0) delete payload.inferKeywords;

  try {
    const res = await fetch(`/api/modes/${encodeURIComponent(id)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      errorEl.textContent = data.error || `HTTP ${res.status}`;
      errorEl.hidden = false;
      return;
    }
    const data = await res.json();
    state.modes = data.modes || [];
    await loadEditableModes();
    refreshModeDropdown();
    reconcileActiveMode();
    document.getElementById('modeEditorPanel').hidden = true;
    document.getElementById('modeEditorPanel').innerHTML = '';
  } catch (err) {
    errorEl.textContent = String(err.message || err);
    errorEl.hidden = false;
  }
}

function refreshModeDropdown() {
  const select = document.getElementById('modeSelect');
  if (!select) return;
  const previous = state.mode;
  select.innerHTML = '<option value="auto">Auto</option>';
  for (const m of state.modes) {
    if (m.id === 'auto') continue;
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    select.append(opt);
  }
  select.value = state.modes.find(m => m.id === previous) ? previous : 'auto';
}

function reconcileActiveMode() {
  if (state.mode === 'auto') return;
  if (!state.modes.find(m => m.id === state.mode)) {
    state.mode = 'auto';
    const select = document.getElementById('modeSelect');
    if (select) select.value = 'auto';
  }
}
```

If your codebase already has a `refreshModeDropdown` / `populateModeSelect` function, reuse that and skip the duplicate definition above. Search first:

```bash
grep -n "modeSelect" public/index.html | head
```

---

- [ ] **Step 7.3: Add `deleteMode` and `resetMode`**

```javascript
async function deleteMode(id) {
  if (!confirm(`Delete mode "${id}"? This cannot be undone.`)) return;
  try {
    const res = await fetch(`/api/modes/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      alert(`Delete failed: ${data.error || res.status}`);
      return;
    }
    const data = await res.json();
    state.modes = data.modes || [];
    await loadEditableModes();
    refreshModeDropdown();
    reconcileActiveMode();
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
}

async function resetMode(id) {
  if (!confirm(`Reset "${id}" will discard your edits and restore the baked-in default. Continue?`)) return;
  try {
    const res = await fetch(`/api/modes/${encodeURIComponent(id)}/reset`, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      alert(`Reset failed: ${data.error || res.status}`);
      return;
    }
    const data = await res.json();
    state.modes = data.modes || [];
    await loadEditableModes();
    refreshModeDropdown();
    reconcileActiveMode();
  } catch (err) {
    alert(`Reset failed: ${err.message}`);
  }
}
```

---

- [ ] **Step 7.4: Manual end-to-end test**

Start the server. Then run through this sequence in the browser (do not skip steps):

1. Open Settings → Modes. All 4 baked-in modes listed. None show "Reset".
2. Edit "Illustration": change label to "Watercolor". Save. Toolbar dropdown now shows "Watercolor". The row now has a "Reset" button.
3. Click Reset on Illustration. Confirm. Label returns to "Illustration".
4. Click + New Mode. Set id `bad-id!` and try Save. See inline error.
5. Fix id to `lab_diagrams`. Fill style, both templates (include `{{query}}` in the first), modeLabelForPrompt, and add inferKeywords. Save. Dropdown grows.
6. With Mode = `auto`, generate a page with prompt "microscope dish bacteria" — confirm it routes through `lab_diagrams`.
7. Delete `lab_diagrams` from Settings. If it was active, dropdown resets to Auto.
8. Restart the server (`pkill -f "node server.js" && npm start`). Reopen the page. State persists for any remaining custom modes.

If any step fails, log the inline error / network response and fix before continuing.

---

- [ ] **Step 7.5: Commit**

```bash
git add public/index.html
git commit -m "feat(modes): editor form, save/delete/reset, dropdown reconciliation"
```

---

## Task 8: Documentation + push

**Files:**
- Modify: `README.md` (one short paragraph noting the editor)

---

- [ ] **Step 8.1: Add a short README note**

Open `README.md`. Find the existing "Features" or "Settings" section. Add one paragraph:

```markdown
### Mode editor

Open Settings → Modes to create, edit, duplicate, delete, or reset modes from
the browser. Custom modes are saved as JSON files under `modes/` and are
git-trackable. Baked-in modes (Illustration, Historical Map, Math, Science) can
be overridden but never destroyed — use Reset to restore the baked-in default.
```

---

- [ ] **Step 8.2: Commit and push everything**

```bash
git add README.md
git commit -m "docs(readme): note Settings → Modes editor"
git push origin main
git tag -a v2.2.0-mode-editor -m "Phase 3: in-app mode editor"
git push origin v2.2.0-mode-editor
```

---

## Acceptance checklist

Before declaring done:

- [ ] All 4 baked-in modes still load on a fresh checkout with `modes/` deleted (FALLBACK_MODES floor works).
- [ ] `POST /api/modes/:id` rejects all five negative tests from Task 3 Step 3.3 with HTTP 400.
- [ ] `DELETE /api/modes/:id` rejects baked-in ids with HTTP 400.
- [ ] `POST /api/modes/:id/reset` rejects non-baked-in ids with HTTP 400.
- [ ] Path-traversal payload from Task 3 Step 3.4 returns HTTP 400 and creates no file outside `modes/`.
- [ ] Editing a baked-in mode in the UI persists across server restart.
- [ ] Reset on an overlaid baked-in mode removes the overlay file.
- [ ] Custom mode with `inferKeywords` gets picked by `auto` inference for matching queries.
- [ ] Dropdown reconciles to `auto` when the active mode is deleted.
- [ ] All commits pushed to `origin/main`; tag `v2.2.0-mode-editor` pushed.

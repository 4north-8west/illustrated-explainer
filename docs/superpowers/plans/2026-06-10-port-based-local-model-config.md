# Port-Based Local Model Configuration (M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the named local-model dropdown in the Settings panel with two port number inputs (small: 8080, large: 8082) so users configure which llama-server port to use rather than selecting cryptic model names.

**Architecture:** Two virtual model IDs (`auto` → small port, `large` → large port) are added to `MODEL_REGISTRY.local`. `resolveChatUrl` is updated to read `modelConfig.localPorts` dynamically for those two IDs so port changes take effect immediately without restarting. `POST /api/models` accepts a `localPorts` field. The Settings panel replaces all named local-model options with two labelled radio choices and adds a "Local Model Ports" section with editable number inputs. Named models (`gemma-4-E4B`, etc.) stay in the registry for backward compatibility and advanced use but are hidden from the UI.

**Note:** M2 (Ollama Cloud fallback with `gemma4:31b-cloud`) was fully completed in Sprint 1.

**Tech Stack:** Node.js/Express (`server.js`), vanilla JS + inline CSS (`public/index.html`)

---

## File map

| File | What changes |
|------|-------------|
| `server.js` | `DEFAULT_MODEL_CONFIG` (add `localPorts`), `MODEL_REGISTRY.local.models` (add `large`), `resolveChatUrl` (dynamic ports), `POST /api/models` (accept `localPorts`) |
| `public/index.html` | `openSettings` — `optionsFor` filter, new port inputs section, Save button payload |

---

## Task 1: Server — `localPorts` config + dynamic `resolveChatUrl` + `POST /api/models` — `server.js`

**Files:**
- Modify: `server.js:54-66` (`DEFAULT_MODEL_CONFIG`)
- Modify: `server.js:127-137` (`MODEL_REGISTRY.local.models`)
- Modify: `server.js:158-163` (`resolveChatUrl`)
- Modify: `server.js:1262-1282` (`POST /api/models`)

---

- [ ] **Step 1.1: Add `localPorts` to `DEFAULT_MODEL_CONFIG`**

Find `const DEFAULT_MODEL_CONFIG = {` (line 54). Add `localPorts` as the last property before the closing `};`:

```javascript
const DEFAULT_MODEL_CONFIG = {
  localOnly: false,
  // ── Initial run ─────────────────────────────────────────────────────
  generation: { provider: 'xai', model: 'grok-imagine-image' },
  classify:   { provider: 'local', model: 'auto' },
  // ── Subsequent information ──────────────────────────────────────────
  analysis:   { provider: 'local', model: 'auto' },
  // ── Drill in ────────────────────────────────────────────────────────
  editing:    { provider: 'xai', model: 'grok-imagine-image' },
  drillText:  { provider: 'local', model: 'auto' },
  // ── Cloud-fallback policy ───────────────────────────────────────────
  allowClassifyCloudFallback: false,
  // ── Local model ports ───────────────────────────────────────────────
  localPorts: { small: 8080, large: 8082 },
};
```

---

- [ ] **Step 1.2: Add `large` virtual model to `MODEL_REGISTRY.local.models`**

Find the `local:` entry in `MODEL_REGISTRY` (around line 121). The `models` object currently starts with `'auto': { ... }`. Add `'large'` immediately after `'auto'`:

```javascript
    models: {
      'auto':  { name: 'Auto-detect (whatever small port has loaded)', capabilities: ['analysis', 'classify', 'drillText'] },
      'large': { name: 'Large-port model', capabilities: ['analysis', 'classify', 'drillText'] },
      'gemma-4-E2B': { ... },   // keep all existing named models as-is
      // ...
    },
```

Only add the `'large'` line — do not change or remove any existing entries.

---

- [ ] **Step 1.3: Update `resolveChatUrl` to use `modelConfig.localPorts` for `auto` and `large`**

Find `function resolveChatUrl(provider, model) {` (line 158). Replace the entire function:

```javascript
function resolveChatUrl(provider, model) {
  const providerConfig = MODEL_REGISTRY[provider];
  if (!providerConfig) return null;
  // Dynamic port resolution: local/auto → small port, local/large → large port.
  // Named local models keep their hard-coded chatUrl.
  if (provider === 'local') {
    if (model === 'large') {
      const port = modelConfig.localPorts?.large ?? 8082;
      return `http://localhost:${port}/v1/chat/completions`;
    }
    if (model === 'auto') {
      const port = modelConfig.localPorts?.small ?? 8080;
      return `http://localhost:${port}/v1/chat/completions`;
    }
  }
  const modelConfigEntry = providerConfig.models?.[model];
  return modelConfigEntry?.chatUrl || providerConfig.chatUrl || null;
}
```

---

- [ ] **Step 1.4: Update `POST /api/models` to accept and validate `localPorts`**

Find `app.post('/api/models', (req, res) => {` (line 1262). The destructuring line currently is:
```javascript
const { generation, editing, analysis, classify, drillText, localOnly, allowClassifyCloudFallback } = req.body;
```
Add `localPorts` to it:
```javascript
const { generation, editing, analysis, classify, drillText, localOnly, allowClassifyCloudFallback, localPorts } = req.body;
```

Then, after the `if (typeof allowClassifyCloudFallback === 'boolean')` line and before the `for (const [key, val] of [...])` loop, add:
```javascript
  if (localPorts && typeof localPorts === 'object') {
    const small = parseInt(localPorts.small, 10);
    const large = parseInt(localPorts.large, 10);
    const current = modelConfig.localPorts ?? { small: 8080, large: 8082 };
    if (Number.isInteger(small) && small > 0 && small < 65536) current.small = small;
    if (Number.isInteger(large) && large > 0 && large < 65536) current.large = large;
    modelConfig.localPorts = current;
  }
```

---

- [ ] **Step 1.5: Syntax check and verification**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
node --check server.js
```
Expected: no output.

```bash
grep -n "localPorts\|'large'" server.js | head -20
```
Expected: `localPorts` in DEFAULT_MODEL_CONFIG, in `resolveChatUrl` (twice), and in `POST /api/models`. `'large'` in MODEL_REGISTRY.local.models.

- [ ] **Step 1.6: Commit**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
git add server.js
git commit -m "feat(M1): add localPorts config; resolveChatUrl uses dynamic ports for auto/large"
```

---

## Task 2: Frontend — Settings panel port inputs + simplified local model options — `public/index.html`

**Files:**
- Modify: `public/index.html:2587-2596` (`optionsFor` function inside `openSettings`)
- Modify: `public/index.html:2651-2653` (remove "edit server.js" note, add port inputs section)
- Modify: `public/index.html:2687-2701` (Save button handler — include `localPorts`)

---

- [ ] **Step 2.1: Update `optionsFor` to show only `auto`/`large` for the local provider, with port labels**

Find `function optionsFor(capability) {` inside `openSettings` (around line 2587). Replace the entire function with:

```javascript
function optionsFor(capability) {
  let html = '';
  for (const [pid, prov] of Object.entries(registry)) {
    for (const m of prov.models) {
      if (!m.capabilities.includes(capability)) continue;
      if (pid === 'local') {
        // Show only the two virtual port models; hide all named Gemma variants
        if (m.id === 'auto') {
          html += `<option value="local/auto">Local — Small port (${current.localPorts?.small ?? 8080})</option>`;
        } else if (m.id === 'large') {
          html += `<option value="local/large">Local — Large port (${current.localPorts?.large ?? 8082})</option>`;
        }
        continue; // skip all other named local models
      }
      html += `<option value="${escapeHtml(pid)}/${escapeHtml(m.id)}">${escapeHtml(prov.name)} — ${escapeHtml(m.name)}</option>`;
    }
  }
  return html;
}
```

Note: `escapeHtml` is a module-level function already defined in the file.

---

- [ ] **Step 2.2: Replace the "edit server.js" note with a Local Model Ports section**

Find the paragraph (around line 2651–2653):
```html
      <div class="setting-group">
        <p class="desc"><strong>Local models with port overrides:</strong> the local registry in server.js can pin individual models to specific llama-server ports (e.g. 8080 for the small Gemma, 8082 for the medium 26B, 8083 for the 31B). Edit MODEL_REGISTRY.local in server.js to add or change ports.</p>
      </div>
```

Replace it with:
```html
      <h3 class="settings-section-h">Local model ports</h3>
      <div class="setting-group">
        <label>Small model port
          <small>Port for all capabilities set to "Local — Small port". Typically your fast 4B model.</small>
        </label>
        <input type="number" class="setting-select" id="smallPortInput"
          min="1" max="65535" step="1"
          value="${current.localPorts?.small ?? 8080}"
          style="width:90px;font-family:monospace">
        <label style="margin-top:10px">Large model port
          <small>Port for all capabilities set to "Local — Large port". Typically your 26B/31B model.</small>
        </label>
        <input type="number" class="setting-select" id="largePortInput"
          min="1" max="65535" step="1"
          value="${current.localPorts?.large ?? 8082}"
          style="width:90px;font-family:monospace">
      </div>
```

---

- [ ] **Step 2.3: Add `localPorts` to the Save button payload**

Find the `$('saveSettingsBtn').onclick` handler (around line 2687). It currently sends:
```javascript
      body: JSON.stringify({
        generation: split('genModelSelect'),
        editing:    split('editModelSelect'),
        analysis:   split('analysisModelSelect'),
        classify:   split('classifyModelSelect'),
        drillText:  split('drillTextModelSelect'),
        localOnly:  $('localOnlyToggle').checked,
        allowClassifyCloudFallback: $('classifyCloudToggle').checked,
      }),
```

Add `localPorts` to the payload:
```javascript
      body: JSON.stringify({
        generation: split('genModelSelect'),
        editing:    split('editModelSelect'),
        analysis:   split('analysisModelSelect'),
        classify:   split('classifyModelSelect'),
        drillText:  split('drillTextModelSelect'),
        localOnly:  $('localOnlyToggle').checked,
        allowClassifyCloudFallback: $('classifyCloudToggle').checked,
        localPorts: {
          small: parseInt($('smallPortInput').value, 10) || 8080,
          large: parseInt($('largePortInput').value, 10) || 8082,
        },
      }),
```

---

- [ ] **Step 2.4: Handle `local/large` in the `split()` helper**

The `split` helper inside `saveSettingsBtn.onclick` is:
```javascript
const split = id => { const [p, m] = $(id).value.split('/'); return { provider: p, model: m }; };
```

The value `local/large` contains one `/` — `split('/')` returns `['local', 'large']`. This already works correctly. No change needed. Just verify by reading the function.

---

- [ ] **Step 2.5: Syntax check**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
node -e "
const fs = require('fs');
const content = fs.readFileSync('public/index.html', 'utf8');
const scripts = content.match(/<script[^>]*>([\s\S]*?)<\/script>/g) || [];
const vm = require('vm');
scripts.forEach((s, i) => {
  try {
    new vm.Script(s.replace(/<\/?script[^>]*>/g, ''));
    console.log('script', i, 'OK');
  } catch(e) {
    console.error('script', i, 'ERROR:', e.message);
  }
});
"
```
Expected: all OK (the pre-existing module-import false positive on one script is acceptable).

- [ ] **Step 2.6: Manual verification**

Start the server:
```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
node server.js
```
Open http://localhost:3000. Click the settings gear icon.

Verify:
1. Each local-capable capability (Classification, Learn-panel analysis, Drill In text) shows exactly two local options: "Local — Small port (8080)" and "Local — Large port (8082)".
2. No named Gemma model entries appear in those dropdowns.
3. A "Local model ports" section appears with two number inputs pre-filled with 8080 and 8082.
4. Changing ports to 8081/8083 and clicking Save: the panel closes without error.
5. Reopening settings: the port inputs show 8081/8083.
6. Check `model-config.json` on disk: `"localPorts": { "small": 8081, "large": 8083 }` should be written.

Kill the server when done.

- [ ] **Step 2.7: Commit**

```bash
cd /Users/timhohne/Documents/my-agent-team/illustrated-explainer
git add public/index.html
git commit -m "feat(M1): settings panel shows port inputs and simplified local model choices"
```

---

## Self-Review

**Spec coverage:**
- Add `localPorts` to config: T1 step 1.1 ✓
- `POST /api/models` accepts `localPorts`: T1 step 1.4 ✓
- `resolveChatUrl` uses dynamic ports for `auto`/`large`: T1 step 1.3 ✓
- Settings panel: two port number inputs: T2 step 2.2 ✓
- Settings panel: remove local model dropdown complexity: T2 step 2.1 ✓
- Named models kept in registry (hidden from UI): T1 step 1.2 adds `large` without removing others; `optionsFor` skips them ✓
- Port changes persisted to disk: T1 step 1.4 calls `saveModelConfig` (existing path) ✓

**Placeholder scan:** All steps contain complete code. No TBDs.

**Type consistency:**
- `localPorts.small` / `localPorts.large`: integers, validated with `parseInt` + `Number.isInteger` in server; `parseInt(...) || 8080` on client. Consistent.
- `'large'` model ID: added to registry in T1 step 1.2, used in `resolveChatUrl` T1 step 1.3, appears in dropdown T2 step 2.1, split by `/` in T2 step 2.4. Consistent throughout.
- `current.localPorts` read in `optionsFor` T2 step 2.1 — `current` comes from `GET /api/models` response which returns `modelConfig` directly (`res.json({ current: modelConfig, registry })`). `modelConfig.localPorts` is set in T1 step 1.1. Consistent.

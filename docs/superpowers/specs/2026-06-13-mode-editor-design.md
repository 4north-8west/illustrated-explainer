# Mode Editor — Design Spec

**Date:** 2026-06-13
**Status:** Approved, awaiting implementation plan
**Author:** Tim Hohne (with Claude Opus 4.7)
**Roadmap origin:** April 2026 Phase 3 — "Configurable mode templates: move MODES to editable JSON files, mode editor in UI." The "move to JSON" half is already shipped (`modes/illustration.json`, `historical_map.json`, `math_equation.json`, `science_process.json` exist; `loadModes()` overlays them onto `FALLBACK_MODES`). This spec covers the remaining half: a runtime CRUD API + a structured-form editor in the Settings panel, with hot reload and ajv-backed validation.

---

## Goals

1. Let the user create, edit, duplicate, delete, and reset modes from the Settings panel without restarting the server or hand-editing files.
2. Make baked-in modes overridable but never destroyable — `FALLBACK_MODES` is the always-available floor.
3. Keep the existing per-file storage in `modes/` so changes remain git-diffable.
4. Tighten input validation surfacing on `POST /api/modes/:id` so the upcoming security review has minimal grounds for findings.

## Non-Goals (YAGNI)

- No authentication / authorization — same localhost threat model as the existing `POST /api/models`.
- No mode versioning, undo history, or import/export — git is the version control.
- No raw-JSON editor tab — structured form is the sole authoring surface.
- No automated test framework — repo has none today; manual test pass is the bar.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  Browser (public/index.html)                                           │
│                                                                        │
│   Settings panel ──► new "Modes" subsection                            │
│     · list of modes  · Edit  · Duplicate  · Delete  · Reset-to-default │
│     · structured form (one input per mode property)                    │
│     · live validation against ajv schema (server-side authoritative)   │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │  fetch JSON
┌──────────────────────────▼─────────────────────────────────────────────┐
│  server.js (Express)                                                   │
│                                                                        │
│   GET    /api/modes              (existing — publicMode shape)         │
│   GET    /api/modes/raw          (NEW — editable shape, all fields)    │
│   POST   /api/modes/:id          (NEW — create or update; ajv-gated)   │
│   DELETE /api/modes/:id          (NEW — only for user-created modes)   │
│   POST   /api/modes/:id/reset    (NEW — restore baked-in default)      │
│                                                                        │
│   On every mutation: validate → write file → reloadModes() → respond   │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │  fs read/write (limited to modes/)
┌──────────────────────────▼─────────────────────────────────────────────┐
│  modes/<id>.json   (per-mode file, git-trackable, baked-in OR custom)  │
│  FALLBACK_MODES     (in-memory, immutable, always available)           │
└────────────────────────────────────────────────────────────────────────┘
```

Two invariants the design enforces:
1. **The set of baked-in modes is captured at boot.** At server start, `BAKED_IN_IDS = new Set(Object.keys(loadModes()))` snapshots the IDs and `BAKED_IN_SNAPSHOTS[id]` deep-copies each mode's normalized content. Baked-in modes are un-deletable and resettable for the lifetime of the process. This handles the asymmetry where some modes (`illustration`, `historical_map`) live in `FALLBACK_MODES` while others (`math_equation`, `science_process`) ship only as on-disk JSON — both are treated as baked-in.
2. **All file writes are confined to `modes/`.** Path is constructed as `path.join(MODES_DIR, sanitizeId(id) + '.json')` and the resolved path's parent must equal `MODES_DIR`.

`hasOverlay` therefore means "the current normalized mode content differs from the boot-time snapshot" — not merely "a file exists on disk." On a fresh boot all four baked-in modes report `hasOverlay: false`; the flag flips to `true` only after a user edits and saves a baked-in mode. Reset writes the snapshot back to `modes/<id>.json` and calls `reloadModes()`.

## Components

### Server-side (all in `server.js` — no new files)

| Symbol | Purpose | Notes |
|---|---|---|
| `MODE_SCHEMA` | ajv JSON schema for editable mode payload | New constant near `FALLBACK_MODES` (~line 252) |
| `validateMode` | compiled ajv validator from `MODE_SCHEMA` | Initialized once at startup; `additionalProperties: false` |
| `sanitizeModeId(id)` | regex check `^[a-z][a-z0-9_]{0,30}$`; rejects traversal | New helper |
| `isBakedInMode(id)` | `id in FALLBACK_MODES` | New helper |
| `reloadModes()` | wraps `loadModes()` + reassigns `MODES` / `VALID_MODES` | New helper |
| `editableMode(mode, { hasOverlay, bakedIn })` | counterpart of `publicMode` — returns full editable shape | New helper |
| `GET /api/modes/raw` handler | returns array of `editableMode` shapes, each with `bakedIn: boolean` and `hasOverlay: boolean` | New route |
| `POST /api/modes/:id` handler | validate → write → reload → return updated `publicMode` list | New route |
| `DELETE /api/modes/:id` handler | reject if baked-in → unlink file → reload | New route |
| `POST /api/modes/:id/reset` handler | reject if not baked-in → write `BAKED_IN_SNAPSHOTS[id]` to file → reload | New route |

### Frontend (all in `public/index.html` — no new files)

| Symbol | Purpose |
|---|---|
| `state.editableModes` | parallel to `state.modes`, populated from `/api/modes/raw` |
| `openModeEditor(id?)` | opens form panel; new mode if `id` undefined |
| `renderModeList()` | renders the Modes subsection inside Settings |
| `submitModeForm()` | POST `/api/modes/:id` with form values; on success refresh both `/api/modes` and `/api/modes/raw` |
| `deleteMode(id)` / `resetMode(id)` | wrap the DELETE / reset routes with a confirm dialog |
| `reconcileActiveMode()` | after refresh, if `state.mode` is no longer in `state.modes`, reset to `auto` |

## Data flow

### Create a custom mode

```
1. User clicks "+ New Mode" in Settings → openModeEditor() (no id)
2. User fills form → submitModeForm()
3. POST /api/modes/:id  { id, label, tagLabel, placeholder, description,
                          style, firstPageTemplate, childPageTemplate,
                          modeLabelForPrompt, inferKeywords? }
4. Server: validateMode(payload)
              → sanitizeModeId(id)
              → confirm no clash with existing modes (unless updating)
              → fs.writeFileSync(modes/<id>.json)
              → reloadModes()
              → 200 { modes: VALID_MODES.map(id => publicMode(MODES[id])) }
5. Frontend: replace state.modes with response, refetch /api/modes/raw,
   re-render mode <select>, reconcileActiveMode(), close editor,
   toast "Mode '<label>' saved"
```

### Edit a baked-in mode

```
1. User clicks Edit on a baked-in mode → openModeEditor('illustration')
2. Form pre-fills from /api/modes/raw entry (which reflects whatever
   the disk currently holds — baked-in or overlaid)
3. id field is rendered read-only and greyed out
4. Save → POST /api/modes/illustration with the modified payload
5. Server writes modes/illustration.json (overlay), reloads
6. UI shows "Reset" button next to the baked-in mode since hasOverlay=true
```

### Reset / delete

```
Reset (baked-in only):
  POST /api/modes/illustration/reset
  → if !isBakedInMode → 400
  → fs.writeFileSync(modes/illustration.json, JSON.stringify(BAKED_IN_SNAPSHOTS[id], null, 2))
  → reloadModes() → the snapshot content is now effective again
  (note: this approach also works for math_equation and science_process,
   which are baked-in by virtue of shipping in modes/ but have no
   FALLBACK_MODES inline entry to fall back to)

Delete (custom only):
  DELETE /api/modes/lab_diagrams
  → if isBakedInMode → 400
  → fs.unlinkSync(modes/lab_diagrams.json)
  → reloadModes()
```

## Validation & security

### `MODE_SCHEMA` (ajv, `additionalProperties: false`)

| Field | Type / constraint |
|---|---|
| `id` | string, `^[a-z][a-z0-9_]{0,30}$`, required |
| `label` | string, 1–60 chars, required |
| `tagLabel` | string, 1–30 chars, required |
| `placeholder` | string, 0–200 chars |
| `description` | string, 0–500 chars |
| `style` | string, 1–8000 chars, required |
| `firstPageTemplate` | string, 1–8000 chars, required, must contain `{{query}}` |
| `childPageTemplate` | string, 1–8000 chars, required |
| `modeLabelForPrompt` | string, 1–60 chars, required |
| `inferKeywords` | array of strings, 0–40 items, each 2–40 chars, **optional** |

The `{{query}}` requirement on `firstPageTemplate` is enforced via a custom ajv keyword or a post-validate check; it ensures the template will actually substitute the user's topic instead of silently producing an empty prompt.

### Path safety

- `sanitizeModeId(id)` rejects anything outside `^[a-z][a-z0-9_]{0,30}$`. No `.`, `/`, `\`, or unicode confusables.
- Server constructs the path via `path.join(MODES_DIR, sanitizeId + '.json')` and **then** verifies that `path.dirname(path.resolve(filePath)) === path.resolve(MODES_DIR)` (defence-in-depth against future regex regressions).
- DELETE only succeeds when `isBakedInMode(id)` is false. Reset only succeeds when `isBakedInMode(id)` is true.

### Request limits

- Verify the global `express.json({ limit: ... })` in `server.js`. If it's looser than ~64 KB, attach a route-local `express.json({ limit: '64 KB' })` to the mode routes.

### Inference integration

- `inferModeFromQuery` keeps its baked-in regex blocks unchanged. After those, a new loop iterates user-defined modes whose `inferKeywords` is non-empty, building a case-insensitive `\b(kw1|kw2|...)\b` regex and returning the first match.
- Baked-in modes win ties (they're checked first) — preserves current behaviour for existing users.
- Each user keyword is `RegExp`-escaped before being joined into the alternation. No user input ever reaches the regex engine raw.

### Concurrency

- Writes are short and infrequent (human-driven from a settings panel) so no lock is necessary. The pattern is `validate → fs.writeFileSync → reloadModes()` synchronously inside the request handler, which serializes against Express's single-threaded event loop. If two saves race, the second one wins — acceptable for a single-user app and consistent with how `model-config.json` is already handled.

## Hot reload

- `reloadModes()` reassigns the module-scoped `MODES` and `VALID_MODES` after every mutation. All existing call sites already read those `let`s on each request (verified at `server.js:706, 712, 841, 1048, 1147, 1212, 1282, 1283, 1346, 1771`), so the next inbound request sees the new modes immediately. No captured-snapshot bugs.
- The default-mode fallback (`hasMode('illustration') ? 'illustration' : VALID_MODES[0]`) survives mode deletions automatically.
- Frontend's `reconcileActiveMode()` resets `state.mode` to `'auto'` whenever the currently-selected mode disappears from the refreshed `state.modes`.

## UI behaviour

```
┌─ Settings ──────────────────────────────────────────────────┐
│  Generation, Editing, Analysis, ports …  (existing)         │
│  ─────────────────────────────────────────────────────────  │
│  Modes                                          [+ New]     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Illustration       (baked-in)   [Edit] [Reset]      │    │
│  │ Historical Map     (baked-in)   [Edit]              │    │
│  │ Math               (baked-in)   [Edit]              │    │
│  │ Science            (baked-in)   [Edit]              │    │
│  │ My Custom Mode     (custom)     [Edit] [Delete]     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  [ Editor slides in when Edit / + New clicked ]             │
│   id ⌸ label ⌸ tagLabel ⌸ placeholder ⌸ description …      │
│   style                    (textarea, monospace)            │
│   firstPageTemplate        (textarea, must contain {{query}})│
│   childPageTemplate        (textarea)                       │
│   modeLabelForPrompt       (text)                           │
│   inferKeywords            (chip input, optional)           │
│                                                             │
│  Validation messages appear inline beneath each field.      │
│  [Cancel]  [Save]                                           │
└─────────────────────────────────────────────────────────────┘
```

- `[Reset]` only renders when the baked-in mode has an overlay file (`hasOverlay: true` from `/api/modes/raw`).
- `[Delete]` only renders for custom modes (`bakedIn: false`).
- Editing a baked-in mode's `id` is disabled (greyed out). The slug is the mode's identity; changing it would orphan its file.
- After Save, the top toolbar's `Mode` dropdown refreshes from the new `/api/modes` response. The currently-selected mode is preserved if it still exists, else reset to `auto`.
- Confirm dialog before Delete and before Reset ("Reset will discard your edits to this mode; the original baked-in version will be restored.").

## Test pass (manual)

1. Start the app, open Settings → Modes. All 4 baked-in modes listed, none marked "has overlay".
2. Click Edit on Illustration. Change `label` to "Watercolor". Save. Toolbar dropdown shows "Watercolor". `modes/illustration.json` now exists on disk with the override.
3. Click Reset on Illustration. `modes/illustration.json` removed. Reload page — label is back to "Illustration".
4. Click + New. Try id `bad-id!`. See inline validation error. Fix to `lab_diagrams`. Fill fields, save. Toolbar dropdown grows.
5. Generate a page with the new `lab_diagrams` mode active. Confirm both `firstPageTemplate` and `childPageTemplate` substitute `{{query}}` and `{{style}}` correctly.
6. Add `inferKeywords: ["microscope", "petri", "titration"]` to the new mode. Save. With Mode = `auto`, type "microscope dish" — inference should select `lab_diagrams`.
7. Delete `lab_diagrams`. Confirm dialog. After delete, the toolbar dropdown shrinks; if `state.mode === 'lab_diagrams'` it resets to `auto`.
8. Negative tests — payload with `additionalProperties: { evil: 1 }` → 400. id `../etc/passwd` → 400. `firstPageTemplate` missing `{{query}}` → 400. `style` of 10,000 chars → 400.
9. Restart the app after step 6. All persisted modes load correctly from disk; `FALLBACK_MODES` keeps baked-in modes available even if a custom file fails to parse (existing behaviour — verify nothing regressed).

## Open questions / risks

- **ajv is already a dep** (`package.json` lists `"ajv": "^8.20.0"`), but the codebase doesn't currently import it. Implementation must verify it's wired up correctly and consider whether to centralize the validator instance.
- **Express body-parser limit** in `server.js` must be inspected during implementation; tighten if global is looser than route needs.
- **Mode dropdown state** when a custom mode is deleted while it's the active mode — `reconcileActiveMode()` covers it but worth verifying interaction with gallery filter pills (`.gallery-mode-pill`).

## Out of scope (future iterations)

- Raw-JSON editor tab (Approach B in brainstorming).
- Filesystem watcher for out-of-band edits (Approach C — useful but not required for Phase 3 acceptance).
- Mode-share / export bundle.
- Migration UI for renaming a mode's id (would require updating gallery records).

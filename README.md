# Illustrated Explainer

Illustrated Explainer is an experimental educational exploration tool for drilling into visual material. Start from a generated explainer image or an uploaded image, click a specific feature, and ask for the next layer of explanation in the format that best fits the content: text, math-aware Markdown, chart, table, diagram, or a follow-up image.

The core idea is simple: the click matters. For non-image responses, the server sends the model a focused two-panel reference image: the full source image with the click marked, plus a zoomed crop centered on the selected region. That gives the model both document context and local detail, while the browser renders the result locally whenever possible.

## What It Does

- Generates first-page educational images from a topic.
- Accepts uploaded images, including notes, diagrams, worksheets, and generated illustrations.
- Lets users click directly on a region of an image to request a drill-down.
- Supports multiple response depths:
  - `Extract`: transcribe or identify what is visibly present.
  - `Explain Here`: explain the selected region in context.
  - `Teach Concepts`: use the selected region as a starting point for broader teaching.
- Renders typed responses locally:
  - Markdown/text
  - LaTeX math via MathJax
  - SVG charts
  - HTML tables
  - SVG diagrams
- Keeps generated pages in a session timeline.
- Restores the previous session from browser local storage.
- Provides a gallery of saved generations.
- Supports per-page translation for generated text-based renderers.
- Includes a `Keep it local` mode to block cloud model calls.

## Current Status

This is a working prototype, not a packaged product. The app is intentionally small: one Express server, one browser page, JSON mode templates, and a local generated-content cache.

The system is useful for experimenting with:

- STEM explainers
- math notation and equation drill-downs
- science process diagrams
- historical maps
- uploaded handwritten or printed notes
- chart/table/diagram response routing
- local-first model workflows

## Quick Start

```bash
npm install
cp .env.example .env
npm start
```

Then open:

```text
http://localhost:3000/
```

To run on another port:

```bash
PORT=3001 npm start
```

Important: open the app through `http://localhost:PORT/`, not by opening `public/index.html` directly. The browser page depends on the Express API for generation, uploads, gallery loading, settings, and saved pages.

## Requirements

- Node.js 18 or newer.
- npm.
- Optional: a local OpenAI-compatible chat/vision server for local analysis.
- Optional: cloud API keys if you choose cloud-backed providers.

The app uses:

- `express` for the local web/API server.
- `sharp` for image composition, markers, crops, and upload processing.
- MathJax in the browser for LaTeX rendering.

## Environment Variables

Copy `.env.example` to `.env` and fill only the providers you plan to use.

```bash
cp .env.example .env
```

Supported variables:

| Variable | Purpose |
| --- | --- |
| `PORT` | Express server port. Defaults to `3000`. |
| `LLAMA_SERVER_URL` | Local OpenAI-compatible chat/vision endpoint. Defaults to `http://localhost:8080`. |
| `XAI_API_KEY` | Enables xAI/Grok cloud models when selected. |
| `OPENAI_API_KEY` | Enables OpenAI cloud models when selected. |
| `GEMINI_API_KEY` | Enables Gemini cloud models when selected. |

Cloud keys are optional. If `Keep it local` is enabled in the Models panel, the server blocks cloud model calls even if cloud providers are configured.

## Model Behavior

The app has three model roles:

| Role | Used For |
| --- | --- |
| Generation | First-page image generation from a text topic. |
| Editing | Image drill-downs that return another image. |
| Analysis | Upload context, Learn panel analysis, text drill-downs, chart specs, table specs, diagram specs, and translation. |

Model settings are stored in `model-config.json`.

### Keep It Local

The `Keep it local` checkbox is intended for privacy-sensitive workflows. When enabled:

- Cloud model calls are blocked server-side.
- Local analysis is allowed through the configured local provider.
- Cloud fallback from local analysis is disabled.
- Cloud image generation and image editing are blocked until a local image generation/editing provider is added.

This means image generation may intentionally fail while `Keep it local` is on. Typed renderers can still work if your local model supports the needed vision/chat task.

## Drill-Down Flow

1. The user starts with a generated or uploaded image.
2. The user enables `Drill In`.
3. The user clicks a feature in the image.
4. The app shows a confirmation bar with:
   - optional user intent
   - response depth
   - output kind: image, text, chart, table, or diagram
5. The server creates a marked reference image:
   - image drill-downs use the full image with a red marker
   - typed drill-downs use a two-panel image with full context and zoomed local detail
6. The selected model returns either:
   - image bytes for image drill-downs
   - Markdown for text
   - JSON specs for charts, tables, and diagrams
7. The browser renders typed responses locally.

## Response Types

### Text

Text drill-downs return Markdown. Math should be returned as LaTeX:

- inline math: `$...$`
- display math: `$$...$$`

The browser renders math with MathJax.

### Chart

Chart drill-downs ask the model for a small JSON chart spec. The browser renders the result as SVG. This is best for numeric trends, ordered sequences, comparisons, or conceptual progressions.

### Table

Table drill-downs ask the model for columns, rows, notes, and uncertainty. The browser renders an HTML table. This is useful for extraction, comparisons, vocabulary, equation parts, or structured notes.

### Diagram

Diagram drill-downs ask the model for nodes and edges. The browser renders an SVG diagram locally. This is useful for processes, dependencies, causal chains, equation relationships, and systems.

### Image

Image drill-downs use the image editing model to generate the next visual page. These are currently cloud-backed unless a local image provider is added.

## Content Modes

Modes live in `modes/*.json`. Each mode defines:

- `id`
- `label`
- `tagLabel`
- `placeholder`
- `description`
- `style`
- `firstPageTemplate`
- `childPageTemplate`
- `modeLabelForPrompt`

Included modes:

- `illustration`
- `historical_map`
- `math_equation`
- `science_process`

To add a mode, create a new JSON file in `modes/` using the same structure. The server loads modes at startup.

## Project Structure

```text
.
├── public/
│   └── index.html          # Single-page browser UI
├── modes/
│   ├── illustration.json
│   ├── historical_map.json
│   ├── math_equation.json
│   └── science_process.json
├── generated/              # Ignored local cache of images, JSON pages, uploads, analysis
├── server.js               # Express API, model calls, image composition, persistence
├── model-config.json       # Current model/provider settings
├── .env.example            # Environment variable template
└── package.json
```

## Generated Data

Generated and uploaded content is saved under `generated/`, which is intentionally ignored by git. It may contain:

- generated PNG files
- uploaded source images
- JSON pages for text/chart/table/diagram outputs
- metadata mappings
- cached image analysis

Do not commit `generated/` unless you intentionally want to publish sample outputs.

## Public Data Safety

This repository intentionally ignores:

- `.env`
- `generated/`
- `node_modules/`

Before publishing a fork or deployment, check that:

- `.env` is not tracked.
- generated user uploads are not tracked.
- cached generated content is not tracked unless intentionally included.
- model settings do not imply privacy guarantees your deployment does not enforce.

## Development Notes

Useful checks:

```bash
node --check server.js
node -e "const fs=require('fs'); const html=fs.readFileSync('public/index.html','utf8'); const scripts=[...html.matchAll(/<script[^>]*>([\\s\\S]*?)<\\/script>/g)].map(m=>m[1]); new Function(scripts.at(-1)); console.log('frontend script parses');"
git diff --check
```

Run the app:

```bash
npm start
```

Run on a specific port:

```bash
PORT=3001 npm start
```

## Known Limitations

- The local image generation/editing provider is not implemented yet.
- `Keep it local` blocks cloud image generation/editing instead of replacing it with a local image model.
- The frontend is a single HTML file, so larger UI changes may eventually benefit from a component framework.
- Chart/table/diagram quality depends heavily on the selected vision/chat model.
- Branching history is not yet implemented; the current UI uses a linear session strip.

## Roadmap Ideas

- Add local image generation/editing providers.
- Add richer MathML or structured equation renderers.
- Add more robust document upload analysis.
- Add selectable prompt templates for different disciplines and age levels.
- Add export/share flows for generated sessions.
- Add branching history once the linear workflow stabilizes.

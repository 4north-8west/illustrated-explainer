# Illustrated Explainer

Illustrated Explainer is a local-first educational exploration tool. Start from a generated or uploaded image, click a specific feature, and ask the app to drill into that location as text, a chart, a table, a diagram, or a follow-up image.

## Features

- Click-targeted drill-downs using a marked full image plus a zoomed detail crop.
- Multiple response depths: extraction, focused explanation, or broader concept teaching.
- Local renderers for Markdown, math, charts, tables, and diagrams.
- Per-page translation for generated text-based renderers.
- Configurable content modes in `modes/*.json`.
- Optional `Keep it local` setting that blocks cloud model calls.
- Gallery and session restore for saved explorations.

## Requirements

- Node.js 18 or newer.
- npm.
- Optional: a local OpenAI-compatible vision/chat server at `LLAMA_SERVER_URL` for local analysis.
- Optional: cloud API keys if you choose to enable cloud providers.

## Setup

```bash
npm install
cp .env.example .env
npm start
```

Then open:

```text
http://localhost:3000/
```

To use another port:

```bash
PORT=3001 npm start
```

## Environment

The app reads these optional variables from `.env`:

- `XAI_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `LLAMA_SERVER_URL`
- `PORT`

Cloud keys are only required when you select cloud-backed providers. With `Keep it local` enabled, cloud calls are blocked.

## Public Data Safety

This repository intentionally ignores:

- `.env`
- `generated/`
- `node_modules/`

Generated images, uploaded content, cached analysis, and local API keys should stay out of git.

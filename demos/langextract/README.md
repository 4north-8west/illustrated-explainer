# LangExtract demo

Experimental sandbox for evaluating [google/langextract](https://github.com/google/langextract)
against the Illustrated Explainer's existing OCR/analysis output.

This demo is **not** part of the shipped app. It lives on the
`experiment/langextract` branch only. To roll back, switch back to `main` and
delete the branch — see "Rollback" below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langextract openai
```

## Run

Requires the local Gemma server running on `http://127.0.0.1:8080`
(the same llama-server the main app uses).

```bash
python demo.py
```

Outputs (gitignored, regenerable):

- `extraction.jsonl` — annotated document with grounded extractions
- `visualization.html` — self-contained interactive viewer

## What it does

Loads the existing analysis Markdown the app already produced for the
"Obsidian-AI Knowledge Loop" diagram (in
`generated/_analysis/5262c3e616ae6657.json`), feeds it to langextract with a
small few-shot example, and emits typed extractions (`component`, `flow_step`,
`concept`, `text_label`, `caption`) with character-offset grounding back into
the source text.

## Rollback

This entire experiment lives on a feature branch. To discard:

```bash
git checkout main
git branch -D experiment/langextract
```

The `demos/` directory in the working tree is untracked on `main` (covered by
`.gitignore` on this branch only); delete it manually if desired:

```bash
rm -rf demos/langextract
```

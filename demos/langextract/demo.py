"""LangExtract demo on the Illustrated Explainer's analysis output.

Feeds the existing OCR/analysis Markdown for the Obsidian-AI Knowledge Loop
diagram to langextract, with a small few-shot example tuned for diagram
explainers. Uses the local Gemma model running on llama-server's
OpenAI-compatible API at http://127.0.0.1:8080/v1/chat/completions.

Outputs:
  - extraction.jsonl
  - visualization.html (self-contained, opens in a browser)
"""

import json
import textwrap
import webbrowser
from pathlib import Path

import langextract as lx
from langextract import factory


# ---- 1) Load the actual analysis the Illustrated Explainer produced ----
ANALYSIS_PATH = Path.home() / (
    "Documents/my-agent-team/illustrated-explainer/generated/_analysis/"
    "5262c3e616ae6657.json"
)
analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
source_text = analysis["description"] + "\n\n" + analysis["explanation"]
print(f"Source length: {len(source_text):,} chars")


# ---- 2) Define extraction prompt + few-shot example ----
prompt = textwrap.dedent("""\
    You are extracting structured entities from an automated analysis of an
    educational diagram. Use VERBATIM spans of the source text, in order of
    appearance, no paraphrasing, no overlap.

    Extraction classes:
      - component: a named labeled box / actor in the diagram (e.g. role names
        in ALL CAPS, anchored objects). Attribute: role (one of: actor, input,
        process, output, container, exclusion).
      - flow_step: a numbered or named arrow describing a transition between
        components. Attribute: step_number (string or empty).
      - concept: a high-level abstract idea named in the analysis (usually
        introduced with "concept", "principle", or in a Key Concepts list).
      - text_label: a verbatim string that appears as text inside the diagram
        image (folder names, file names, captions, signs).
      - caption: the explanatory caption text describing the whole image.

    Provide concise meaningful attributes. Do not invent text — every
    extraction_text must be a substring of the source.""")

# Few-shot example: a small fictional diagram analysis written in the same
# style the Illustrated Explainer produces. All extraction_text values are
# verbatim from the example text.
example_text = (
    "## Visual Description\nThis image is a flow diagram titled \"Photosynthesis "
    "Cycle.\" On the left is a SUN (Energy Source) icon, connected by an arrow "
    "labeled \"1. Provides Light\" to a LEAF (Reactor) icon.\n\n"
    "## Text Content\n**Title:**\nPhotosynthesis Cycle\n\n"
    "**Labels and Components:**\nSUN (Energy Source)\n1. Provides Light\nLEAF (Reactor)\n\n"
    "**Caption:**\nLight energy is absorbed and converted into chemical energy."
)
examples = [
    lx.data.ExampleData(
        text=example_text,
        extractions=[
            lx.data.Extraction(
                extraction_class="component",
                extraction_text="SUN (Energy Source)",
                attributes={"role": "input"},
            ),
            lx.data.Extraction(
                extraction_class="flow_step",
                extraction_text="1. Provides Light",
                attributes={"step_number": "1"},
            ),
            lx.data.Extraction(
                extraction_class="component",
                extraction_text="LEAF (Reactor)",
                attributes={"role": "process"},
            ),
            lx.data.Extraction(
                extraction_class="text_label",
                extraction_text="Photosynthesis Cycle",
                attributes={"context": "title"},
            ),
            lx.data.Extraction(
                extraction_class="caption",
                extraction_text="Light energy is absorbed and converted into chemical energy.",
                attributes={"position": "bottom"},
            ),
        ],
    )
]


# ---- 3) Configure langextract to use local Gemma via llama-server ----
config = factory.ModelConfig(
    model_id="gemma-4-E4B-it-Q4_K_S.gguf",  # llama-server uses whatever's loaded
    provider="OpenAILanguageModel",
    provider_kwargs={
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "local-llama-server",  # placeholder; llama-server doesn't check
        "format_type": lx.data.FormatType.JSON,
        "temperature": 0.0,
    },
)

# ---- 4) Run extraction ----
print("Running extraction (local Gemma; may take ~30-60s)...")
result = lx.extract(
    text_or_documents=source_text,
    prompt_description=prompt,
    examples=examples,
    config=config,
    fence_output=True,         # tell model to wrap output in ```json fences
    use_schema_constraints=False,  # OpenAI provider doesn't support schema constraints here
    max_char_buffer=4000,
    extraction_passes=1,
    max_workers=1,
    show_progress=True,
)

print()
print(f"Total extractions: {len(result.extractions)}")
grounded = [e for e in result.extractions if e.char_interval]
ungrounded = len(result.extractions) - len(grounded)
print(f"  Grounded (have char_interval): {len(grounded)}")
print(f"  Ungrounded (filtered out by `if e.char_interval`): {ungrounded}")

by_class: dict[str, int] = {}
for e in grounded:
    by_class[e.extraction_class] = by_class.get(e.extraction_class, 0) + 1
print(f"  By class: {by_class}")

print()
print("Sample extractions:")
for e in grounded[:8]:
    text = e.extraction_text.replace("\n", " ")
    if len(text) > 70:
        text = text[:67] + "..."
    print(
        f"  [{e.extraction_class:<11}] '{text}' "
        f"@ {e.char_interval.start_pos}–{e.char_interval.end_pos} "
        f"attrs={e.attributes}"
    )

# ---- 5) Save JSONL + render HTML viz ----
out_dir = Path(__file__).parent
lx.io.save_annotated_documents([result], output_name="extraction.jsonl", output_dir=str(out_dir))
print(f"\nWrote {out_dir/'extraction.jsonl'}")

html_content = lx.visualize(str(out_dir / "extraction.jsonl"))
html_str = html_content.data if hasattr(html_content, "data") else html_content
viz_path = out_dir / "visualization.html"
viz_path.write_text(html_str, encoding="utf-8")
print(f"Wrote {viz_path}")

webbrowser.open(f"file://{viz_path}")

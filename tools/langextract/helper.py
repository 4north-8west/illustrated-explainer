"""LangExtract helper: stdin JSON in, stdout JSON out.

Designed to be spawned as a subprocess from the Express server.

Stdin (one JSON object on a single read):
  {
    "text": "<source markdown / analysis text>",
    "model_id": "gemma-4-E4B-it-Q4_K_S.gguf",        # optional
    "base_url": "http://127.0.0.1:8080/v1",          # optional
    "max_char_buffer": 4000,                          # optional
    "extraction_passes": 1                            # optional
  }

Stdout (one JSON object):
  {
    "extractions": [
      {
        "extraction_class": "component",
        "extraction_text": "USER (Organizer)",
        "char_interval": {"start_pos": 424, "end_pos": 440},
        "attributes": {"role": "actor"}
      },
      ...
    ],
    "stats": {"total": 41, "grounded": 41, "by_class": {...}, "elapsed_ms": 63000},
    "error": null   # or string on failure
  }

The helper exits 0 with `error` set when extraction fails so the caller can
fall back gracefully without parsing stderr.
"""

import json
import sys
import time
import traceback
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            emit({"extractions": [], "stats": {}, "error": "empty stdin"})
            return 0
        req = json.loads(raw)
    except Exception as exc:
        emit({"extractions": [], "stats": {}, "error": f"bad stdin json: {exc}"})
        return 0

    text = (req.get("text") or "").strip()
    if not text:
        emit({"extractions": [], "stats": {}, "error": "no text"})
        return 0

    model_id = req.get("model_id") or "gemma-4-E4B-it-Q4_K_S.gguf"
    base_url = req.get("base_url") or "http://127.0.0.1:8080/v1"
    max_char_buffer = int(req.get("max_char_buffer") or 4000)
    extraction_passes = int(req.get("extraction_passes") or 1)

    try:
        import langextract as lx  # noqa: WPS433
        from langextract import factory  # noqa: WPS433
    except Exception as exc:
        emit(
            {
                "extractions": [],
                "stats": {},
                "error": f"langextract not installed: {exc}",
            }
        )
        return 0

    prompt = (
        "You are extracting structured entities from an automated analysis of an "
        "image (educational diagram, infographic, informational flyer, math notes, "
        "scientific process, historical map, or chart). Use VERBATIM spans from "
        "the source text, in order of appearance, with no paraphrasing or overlap.\n\n"
        "Extraction classes:\n"
        "  - component: a named labeled element / actor / box in the image (e.g. role "
        "names in ALL CAPS, anchored objects, headings of major sections of a flyer). "
        "Attribute: role (one of: actor, input, process, output, container, exclusion, "
        "section).\n"
        "  - flow_step: a numbered or named arrow / step describing a transition. "
        "Attribute: step_number (string or empty).\n"
        "  - concept: a high-level abstract idea introduced as a principle, concept, "
        "or key term.\n"
        "  - text_label: a verbatim string that appears as text inside the image "
        "(folder names, file names, captions, signs, dates, contact info on flyers).\n"
        "  - caption: explanatory caption text describing the whole image or a major "
        "region.\n"
        "  - data_point: a quantitative datum (value + label, axis tick, table cell) "
        "that came from a chart or table.\n"
        "  - cta: a call-to-action or actionable item on a flyer (e.g. 'Register at...', "
        "'RSVP by 5/15', 'Email contact@...').\n\n"
        "Provide concise meaningful attributes. Every extraction_text must be a "
        "substring of the source. Skip categories that don't apply to the image."
    )

    examples = [
        # Diagram example
        lx.data.ExampleData(
            text=(
                "## Visual Description\n"
                "This image is a flow diagram titled \"Photosynthesis Cycle.\" On the "
                "left is a SUN (Energy Source) icon, connected by an arrow labeled "
                "\"1. Provides Light\" to a LEAF (Reactor) icon.\n\n"
                "## Text Content\n"
                "**Title:**\nPhotosynthesis Cycle\n\n"
                "**Labels and Components:**\nSUN (Energy Source)\n"
                "1. Provides Light\nLEAF (Reactor)\n\n"
                "**Caption:**\nLight energy is absorbed and converted into chemical "
                "energy."
            ),
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
                    extraction_text=(
                        "Light energy is absorbed and converted into chemical energy."
                    ),
                    attributes={"position": "bottom"},
                ),
            ],
        ),
        # Flyer example
        lx.data.ExampleData(
            text=(
                "## Visual Description\n"
                "An informational flyer titled \"Spring Career Fair\" announcing an "
                "event on April 15, 2026 at 1:00 PM in the UC Merced Lantern. The "
                "flyer lists FEATURED EMPLOYERS and provides an RSVP link.\n\n"
                "## Text Content\n"
                "**Title:**\nSpring Career Fair\n\n"
                "**Labels and Components:**\nFEATURED EMPLOYERS\nApril 15, 2026\n"
                "1:00 PM\nUC Merced Lantern\n\n"
                "**Caption:**\nRSVP at career.ucmerced.edu/fair by April 10."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="text_label",
                    extraction_text="Spring Career Fair",
                    attributes={"context": "title"},
                ),
                lx.data.Extraction(
                    extraction_class="component",
                    extraction_text="FEATURED EMPLOYERS",
                    attributes={"role": "section"},
                ),
                lx.data.Extraction(
                    extraction_class="text_label",
                    extraction_text="April 15, 2026",
                    attributes={"context": "date"},
                ),
                lx.data.Extraction(
                    extraction_class="text_label",
                    extraction_text="1:00 PM",
                    attributes={"context": "time"},
                ),
                lx.data.Extraction(
                    extraction_class="text_label",
                    extraction_text="UC Merced Lantern",
                    attributes={"context": "location"},
                ),
                lx.data.Extraction(
                    extraction_class="cta",
                    extraction_text="RSVP at career.ucmerced.edu/fair by April 10.",
                    attributes={"deadline": "April 10"},
                ),
            ],
        ),
    ]

    config = factory.ModelConfig(
        model_id=model_id,
        provider="OpenAILanguageModel",
        provider_kwargs={
            "base_url": base_url,
            "api_key": "local-llama-server",
            "format_type": lx.data.FormatType.JSON,
            "temperature": 0.0,
        },
    )

    started = time.time()
    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            config=config,
            fence_output=True,
            use_schema_constraints=False,
            max_char_buffer=max_char_buffer,
            extraction_passes=extraction_passes,
            max_workers=1,
            show_progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        emit(
            {
                "extractions": [],
                "stats": {},
                "error": f"extract failed: {exc}",
                "trace": traceback.format_exc().splitlines()[-6:],
            }
        )
        return 0

    elapsed_ms = int((time.time() - started) * 1000)
    grounded = [e for e in result.extractions if e.char_interval]
    by_class: dict[str, int] = {}
    for e in grounded:
        by_class[e.extraction_class] = by_class.get(e.extraction_class, 0) + 1

    payload_extractions = []
    for e in grounded:
        ci = e.char_interval
        payload_extractions.append(
            {
                "extraction_class": e.extraction_class,
                "extraction_text": e.extraction_text,
                "char_interval": {
                    "start_pos": int(ci.start_pos),
                    "end_pos": int(ci.end_pos),
                },
                "attributes": dict(e.attributes or {}),
            }
        )

    emit(
        {
            "extractions": payload_extractions,
            "stats": {
                "total": len(result.extractions),
                "grounded": len(grounded),
                "by_class": by_class,
                "elapsed_ms": elapsed_ms,
                "model_id": model_id,
                "base_url": base_url,
            },
            "error": None,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

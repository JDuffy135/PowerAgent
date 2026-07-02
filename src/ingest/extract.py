"""LLM extraction node: raw log text -> `ParsedBatch` (ARCHITECTURE.md §5.3).

`extract_training_data` is pure with respect to the DB: it never writes.
`resolve_exercise` is used read-only (if a connection is supplied) to tag sets
with a known `exercise_id`; unresolved names are surfaced as
`NewExerciseCandidate`s for Step 3's HITL review to confirm, not auto-inserted.

`get_llm()` is the provider seam from ARCHITECTURE.md §6.3: every extraction
call goes through it, so flipping this node from local Ollama to a cloud
provider is a `config.yaml` edit, not a call-site change.
"""
from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import yaml
from pydantic import ValidationError

from src.ingest.models import NewExerciseCandidate, ParsedBatch
from src.tools.resolve import resolve_exercise

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_TIER = "accessory"

# Prompt text in, raw JSON string out (before Pydantic validation).
LLMCallable = Callable[[str], str]

EXTRACTION_SYSTEM_PROMPT = """You are a data-extraction engine for a powerlifting training log. \
Convert the raw log text you're given into JSON matching the provided schema exactly. \
Output ONLY the JSON object -- no prose, no markdown code fences.

Rules:
- Normalize every weight to pounds. If a value is given in kg (e.g. "170KG"), convert \
using 1 kg = 2.20462 lb and round to 1 decimal place, but keep the original substring \
verbatim in `raw_text` so nothing is lost.
- If a cell/line states both a planned/projected weight and a different actual weight \
used (e.g. "1x3 @ 170KG (actually used 375 pounds)"), put the plan in a \
`programmed_slots` entry (`prescription`, `target_weight_lb`) on the session and the \
real performance in a `sets` entry. Do not merge the two into one number.
- Pin/plate configuration strings (e.g. "143x1, 121x2", "35KGx2") describe machine \
settings, not just weight: put the setting text in `equipment_note` on the set row and \
still fill `weight_lb` with your best-effort numeric read (normalized to lb).
- A line with a top single followed by backoff sets (e.g. "385x1, 315x4") becomes \
multiple set rows in order; the heaviest/first row gets `is_top_set: true`.
- If an exercise was explicitly skipped (e.g. "Reps: N/A"), do not emit set rows for \
it, but keep the surrounding note text in the session's `raw_note` unchanged.
- Slang, Spanish, emoji, and other prose are irrelevant to the numeric fields -- leave \
them in `raw_note`/`raw_text` and do not let them affect weight/rep/RPE parsing.
- Set a field's `confidence` below 1.0 whenever you had to guess (ambiguous unit, \
illegible number, uncertain exercise identity, etc.).
- Always populate `exercise_raw` with the name exactly as it appears in the log; a \
separate step resolves it to a canonical exercise, so do not normalize or invent a name.
"""


def get_llm(
    node: str = "ingest_extract",
    *,
    system_prompt: str | None = None,
    schema: dict | None = None,
) -> LLMCallable:
    """Return a `prompt -> raw JSON string` callable for the given graph node.

    Reads `config.yaml`'s `nodes.<node>` section for `provider`/`model`/`host`;
    defaults to a local Ollama chat completion with structured output
    (the JSON schema of `ParsedBatch` passed as Ollama's `format` param).

    `system_prompt`/`schema` default to the extraction prompt and `ParsedBatch`
    schema; other structured-output nodes (e.g. the HITL correction pass) supply
    their own. `src.agent.llm_provider` re-exports this as the shared raw seam.
    """
    cfg = _node_config(node)
    provider = cfg.get("provider", "local")
    if provider != "local":
        raise NotImplementedError(f"Provider {provider!r} is not wired up yet (cloud lands in Stage 7)")

    model = cfg.get("model", DEFAULT_MODEL)
    host = cfg.get("host", DEFAULT_OLLAMA_HOST)
    if schema is None:
        schema = ParsedBatch.model_json_schema()
    if system_prompt is None:
        system_prompt = EXTRACTION_SYSTEM_PROMPT

    def _call(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
        }
        request = urllib.request.Request(
            f"{host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request to {host} failed (is `ollama serve` running?): {exc}"
            ) from exc
        return body["message"]["content"]

    return _call


def _node_config(node: str) -> dict:
    if not CONFIG_PATH.exists():
        return {}
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return cfg.get("nodes", {}).get(node, {}) or {}


def extract_training_data(
    text: str,
    conn: sqlite3.Connection | None = None,
    llm: LLMCallable | None = None,
) -> ParsedBatch:
    """Parse raw log text into a schema-validated `ParsedBatch`.

    Pure with respect to the DB: `conn`, if given, is only used read-only via
    `resolve_exercise` to fill in `exercise_id` on sets/slots and to decide
    which raw names become `new_exercise_candidates`. No rows are written here.

    `llm` defaults to `get_llm()`; tests inject a stub so extraction is
    deterministic without a live Ollama server.
    """
    call = llm if llm is not None else get_llm()

    raw_response = call(text)
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM extraction did not return valid JSON: {exc}\n{raw_response!r}") from exc

    try:
        batch = ParsedBatch.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"LLM extraction output failed schema validation: {exc}") from exc

    if conn is not None:
        _resolve_exercises(conn, batch)

    return batch


def _resolve_exercises(conn: sqlite3.Connection, batch: ParsedBatch) -> None:
    """Fill in `exercise_id` for every set/slot; unresolved raw names become
    `new_exercise_candidates` (deduplicated by normalized raw name)."""
    seen_unresolved: dict[str, NewExerciseCandidate] = {}

    def resolve(raw_name: str) -> int | None:
        resolved = resolve_exercise(conn, raw_name)
        if resolved is not None:
            return resolved.exercise_id

        key = " ".join(raw_name.strip().lower().split())
        if key and key not in seen_unresolved:
            seen_unresolved[key] = NewExerciseCandidate(
                raw_name=raw_name,
                suggested_name=raw_name.strip(),
                suggested_tier=DEFAULT_TIER,
                suggested_muscle_group=None,
                confidence=0.5,
            )
        return None

    for session in batch.sessions:
        for parsed_set in session.sets:
            parsed_set.exercise_id = resolve(parsed_set.exercise_raw)
        for slot in session.programmed_slots:
            slot.exercise_id = resolve(slot.exercise_raw)

    batch.new_exercise_candidates.extend(seen_unresolved.values())

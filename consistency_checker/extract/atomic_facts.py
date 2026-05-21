"""Atomic-fact extraction.

Takes a :class:`Chunk` and returns a list of :class:`Assertion` records, each
holding one decontextualised, verifiable claim. Two backends:

- :class:`FixtureExtractor` — returns canned responses keyed by chunk id. Used
  by hermetic tests; no network.
- :class:`AnthropicExtractor` — calls Claude with the FActScore-style prompt
  in ``prompts/atomic_facts.txt``. Uses the SDK's tool-use feature to force a
  JSON schema, with Pydantic validation on the response.

OpenAI support can be added later as another implementation of
:class:`Extractor` without touching downstream code.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from consistency_checker.corpus.chunker import Chunk
from consistency_checker.corpus.junk_filter import JunkAudit, is_junk_assertion
from consistency_checker.extract.schema import Assertion
from consistency_checker.logging_setup import get_logger

if TYPE_CHECKING:
    import anthropic

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "atomic_facts.txt"

TOOL_NAME = "record_extraction"
TOOL_SCHEMA: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": "Record both atomic assertions and any definitions extracted from a chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "assertions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of atomic, decontextualised assertions extracted from the text. "
                    "May be empty if the chunk contains no verifiable claims."
                ),
            },
            "definitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "The term being defined, as written.",
                        },
                        "definition_text": {
                            "type": "string",
                            "description": "What the term is said to mean (the right-hand side).",
                        },
                        "containing_sentence": {
                            "type": "string",
                            "description": (
                                "The full sentence (or clause) in which the definition appears, "
                                "copied verbatim from the source."
                            ),
                        },
                    },
                    "required": ["term", "definition_text", "containing_sentence"],
                },
                "description": (
                    "Definitions found in the text, whether formally ('X means …') or "
                    "informally ('by X we mean …', 'an eligible Y is …'). May be empty."
                ),
            },
        },
        "required": ["assertions", "definitions"],
    },
}

_log = get_logger(__name__)


class _DefinitionItem(BaseModel):
    """One definition extracted from a chunk."""

    term: str = Field(min_length=1)
    definition_text: str = Field(min_length=1)
    containing_sentence: str = Field(min_length=1)


class _ExtractionPayload(BaseModel):
    """Pydantic guard on the combined tool-use payload."""

    assertions: list[str] = Field(default_factory=list)
    definitions: list[_DefinitionItem] = Field(default_factory=list)


def render_prompt(chunk_text: str) -> str:
    """Substitute ``{chunk_text}`` into the prompt template."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{chunk_text}", chunk_text)


def parse_tool_response(response: Any) -> _ExtractionPayload:
    """Extract the combined assertion + definition payload from an Anthropic response."""
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        block_type = getattr(block, "type", None)
        block_name = getattr(block, "name", None)
        if block_type == "tool_use" and block_name == TOOL_NAME:
            payload = getattr(block, "input", None) or {}
            return _ExtractionPayload.model_validate(payload)
    raise ValueError(f"No tool_use block named {TOOL_NAME!r} found in response")


def _assertions_from_payload(chunk: Chunk, payload: _ExtractionPayload) -> list[Assertion]:
    out: list[Assertion] = []
    for text in payload.assertions:
        if not text.strip():
            continue
        out.append(
            Assertion.build(
                chunk.doc_id,
                text,
                chunk_id=chunk.chunk_id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
        )
    for d in payload.definitions:
        if not d.containing_sentence.strip():
            continue
        out.append(
            Assertion.build(
                chunk.doc_id,
                d.containing_sentence,
                chunk_id=chunk.chunk_id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                kind="definition",
                term=d.term,
                definition_text=d.definition_text,
            )
        )
    return out


class Extractor(Protocol):
    """Anything that turns a Chunk into a list of Assertion records."""

    def extract(self, chunk: Chunk) -> list[Assertion]: ...


class FixtureExtractor:
    """Canned-response extractor for hermetic tests.

    Two call forms — the legacy positional ``FixtureExtractor({chunk_id: [facts]})``
    keeps working for older tests; the keyword form
    ``FixtureExtractor(facts=..., definitions=...)`` adds definition support.
    """

    def __init__(
        self,
        fixtures: Mapping[str, list[str]] | None = None,
        *,
        facts: Mapping[str, list[str]] | None = None,
        definitions: Mapping[str, list[Mapping[str, str]]] | None = None,
    ) -> None:
        if fixtures is not None and facts is not None:
            raise ValueError("pass either fixtures (legacy) or facts=, not both")
        self._facts: dict[str, list[str]] = dict(fixtures or facts or {})
        self._definitions: dict[str, list[Mapping[str, str]]] = dict(definitions or {})

    def extract(self, chunk: Chunk) -> list[Assertion]:
        payload = _ExtractionPayload(
            assertions=list(self._facts.get(chunk.chunk_id, [])),
            definitions=[_DefinitionItem(**d) for d in self._definitions.get(chunk.chunk_id, [])],
        )
        return _assertions_from_payload(chunk, payload)


class AnthropicExtractor:
    """Calls Anthropic Claude with the atomic-facts prompt and parses the tool response."""

    def __init__(
        self,
        *,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ) -> None:
        if client is None:
            import anthropic  # imported lazily so the test suite need not have a real API key

            self._client: Any = anthropic.Anthropic()
        else:
            self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def extract(self, chunk: Chunk) -> list[Assertion]:
        prompt = render_prompt(chunk.text)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
        try:
            payload = parse_tool_response(response)
        except ValueError as exc:
            _log.warning("Atomic-fact extraction returned no usable payload: %s", exc)
            return []
        return _assertions_from_payload(chunk, payload)


class MoonshotExtractor:
    """Atomic-fact extraction via Moonshot (Kimi) using the OpenAI-compatible API.

    Mirrors :class:`AnthropicExtractor` but uses ``beta.chat.completions.parse``
    with the same :class:`_ExtractionPayload` schema, so the whole pipeline can
    run on a single Moonshot key without an Anthropic credential.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        model: str = "kimi-k2.6",
        api_key: str | None = None,
        max_tokens: int = 8192,
        disable_thinking: bool = True,
    ) -> None:
        if client is None:
            import os

            import openai

            key = api_key or os.getenv("MOONSHOT_API_KEY")
            if not key:
                raise ValueError(
                    "MOONSHOT_API_KEY not set. Set via env var, .env file, or pass to __init__"
                )
            self._client: Any = openai.OpenAI(api_key=key, base_url="https://api.moonshot.ai/v1")
        else:
            self._client = client
        self._model = model
        self._max_tokens = max_tokens
        # Extraction is mechanical — Kimi's reasoning mode adds ~50x latency
        # (>240s vs ~5s per chunk) for no quality gain, so disable it by default.
        self._extra_body: dict[str, Any] = (
            {"thinking": {"type": "disabled"}} if disable_thinking else {}
        )

    def extract(self, chunk: Chunk) -> list[Assertion]:
        import openai

        prompt = render_prompt(chunk.text)
        try:
            response = self._client.beta.chat.completions.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract atomic, decontextualised assertions and any definitions "
                            "from the user's text. Respond only via the structured format."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=_ExtractionPayload,
                extra_body=self._extra_body,
            )
            payload = response.choices[0].message.parsed
            if payload is None:
                raise ValueError("Moonshot extraction returned None payload")
            if not isinstance(payload, _ExtractionPayload):
                payload = _ExtractionPayload.model_validate(payload)
        except (ValueError, ValidationError, openai.LengthFinishReasonError) as exc:
            # A single oversized/garbled chunk must not abort a whole upload; skip it.
            # Auth/network errors deliberately propagate so a bad key still surfaces.
            _log.warning("Atomic-fact extraction (moonshot) returned no usable payload: %s", exc)
            return []
        return _assertions_from_payload(chunk, payload)


class JunkFilteringExtractor:
    """Wraps an :class:`Extractor`, dropping assertions flagged by ``is_junk_assertion``.

    Applied in :func:`make_extractor` so every ingest path (CLI, web, headless)
    inherits assertion-stage filtering with no per-path wiring.
    """

    def __init__(self, inner: Extractor, *, audit: JunkAudit | None = None) -> None:
        self._inner = inner
        self._audit = audit

    def extract(self, chunk: Chunk) -> list[Assertion]:
        kept: list[Assertion] = []
        for assertion in self._inner.extract(chunk):
            reason = is_junk_assertion(assertion.assertion_text)
            if reason is None:
                kept.append(assertion)
            elif self._audit is not None:
                self._audit.record(
                    stage="assertion",
                    reason=reason,
                    doc_id=assertion.doc_id,
                    text=assertion.assertion_text,
                )
        return kept

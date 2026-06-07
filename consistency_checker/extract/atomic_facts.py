"""Atomic-fact extraction.

Takes a :class:`Chunk` and returns a list of :class:`Assertion` records, each
holding one decontextualised, verifiable claim. Backends:

- :class:`FixtureExtractor` — returns canned responses keyed by chunk id. Used
  by hermetic tests; no network.
- :class:`AnthropicExtractor` — calls Claude with the FActScore-style prompt
  in ``prompts/atomic_facts.txt``. Uses the SDK's tool-use feature to force a
  JSON schema, with Pydantic validation on the response.
- :class:`MoonshotExtractor` — calls Kimi (Moonshot AI) via the OpenAI-
  compatible ``beta.chat.completions.parse`` endpoint with the same schema.
- :class:`JunkFilteringExtractor` — decorator that drops junk assertions from
  any extractor.

New backends implement the :class:`Extractor` Protocol without touching
downstream code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

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


OrgIdentificationReason = Literal["org_found", "no_org", "llm_error", "truncated"]


@dataclass(frozen=True, slots=True)
class OrgIdentification:
    """Result of a document-level org-identification call."""

    label: str | None
    reason: OrgIdentificationReason


ORG_PROMPT_CHAR_CAP = 2000

ORG_TOOL_SCHEMA: dict[str, Any] = {
    "name": "identify_org",
    "description": "Identify the primary issuing organization of a document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": ["string", "null"]},
            "reason": {"type": "string", "enum": ["org_found", "no_org"]},
        },
        "required": ["label", "reason"],
    },
}


def _render_org_prompts(title: str | None, text: str) -> tuple[str, str, bool]:
    """Return (system, user, truncated). truncated is True iff text was clipped."""
    here = Path(__file__).resolve().parent / "prompts"
    system = (here / "org_identifier_system.txt").read_text(encoding="utf-8")
    template = (here / "org_identifier_user.txt").read_text(encoding="utf-8")
    truncated = len(text) > ORG_PROMPT_CHAR_CAP
    text_prefix = text[:ORG_PROMPT_CHAR_CAP]
    user = template.replace("{title}", title or "(none)").replace("{text_prefix}", text_prefix)
    return system, user, truncated


class _OrgIdentificationPayload(BaseModel):
    label: str | None
    reason: Literal["org_found", "no_org"]


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
    def identify_org(self, *, title: str | None, text: str) -> OrgIdentification: ...


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
        org_fixtures: Mapping[tuple[str | None, str], OrgIdentification] | None = None,
    ) -> None:
        if fixtures is not None and facts is not None:
            raise ValueError("pass either fixtures (legacy) or facts=, not both")
        self._facts: dict[str, list[str]] = dict(fixtures or facts or {})
        self._definitions: dict[str, list[Mapping[str, str]]] = dict(definitions or {})
        self._org_fixtures: dict[tuple[str | None, str], OrgIdentification] = dict(
            org_fixtures or {}
        )

    def extract(self, chunk: Chunk) -> list[Assertion]:
        payload = _ExtractionPayload(
            assertions=list(self._facts.get(chunk.chunk_id, [])),
            definitions=[_DefinitionItem(**d) for d in self._definitions.get(chunk.chunk_id, [])],
        )
        return _assertions_from_payload(chunk, payload)

    def identify_org(self, *, title: str | None, text: str) -> OrgIdentification:
        for (k_title, k_prefix), res in self._org_fixtures.items():
            if k_title == title and text.startswith(k_prefix):
                return res
        return OrgIdentification(label=None, reason="no_org")


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

    def identify_org(self, *, title: str | None, text: str) -> OrgIdentification:
        system, user, truncated = _render_org_prompts(title, text)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=200,
                system=system,
                tools=[ORG_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "identify_org"},
                messages=[{"role": "user", "content": user}],
            )
        except Exception:
            return OrgIdentification(label=None, reason="llm_error")
        for block in getattr(response, "content", []) or []:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "identify_org"
            ):
                payload = block.input or {}
                label = payload.get("label")
                reason = payload.get("reason", "no_org")
                if reason == "no_org" and truncated:
                    return OrgIdentification(label=None, reason="truncated")
                if reason not in ("org_found", "no_org"):
                    return OrgIdentification(label=None, reason="llm_error")
                if reason == "org_found" and not (isinstance(label, str) and label.strip()):
                    return OrgIdentification(label=None, reason="llm_error")
                return OrgIdentification(
                    label=label if reason == "org_found" else None, reason=reason
                )
        return OrgIdentification(label=None, reason="llm_error")


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
            payload = response.choices[0].message.parsed if response.choices else None
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

    def identify_org(self, *, title: str | None, text: str) -> OrgIdentification:
        system, user, truncated = _render_org_prompts(title, text)
        try:
            response = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=_OrgIdentificationPayload,
                max_tokens=200,
                extra_body=self._extra_body,
            )
        except Exception:
            return OrgIdentification(label=None, reason="llm_error")
        parsed = response.choices[0].message.parsed if response.choices else None
        if parsed is None:
            return OrgIdentification(label=None, reason="llm_error")
        label, reason = parsed.label, parsed.reason
        if reason == "no_org" and truncated:
            return OrgIdentification(label=None, reason="truncated")
        if reason == "org_found" and not (isinstance(label, str) and label.strip()):
            return OrgIdentification(label=None, reason="llm_error")
        return OrgIdentification(label=label if reason == "org_found" else None, reason=reason)


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

    def identify_org(self, *, title: str | None, text: str) -> OrgIdentification:
        return self._inner.identify_org(title=title, text=text)

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

from pydantic import BaseModel, Field

from consistency_checker.corpus.chunker import Chunk
from consistency_checker.extract.schema import Assertion
from consistency_checker.logging_setup import get_logger

if TYPE_CHECKING:
    import anthropic

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "atomic_facts.txt"

TOOL_NAME = "record_assertions"
TOOL_SCHEMA: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": "Record the atomic assertions extracted from a chunk of text.",
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
            }
        },
        "required": ["assertions"],
    },
}

_log = get_logger(__name__)


class _AssertionList(BaseModel):
    """Pydantic guard on the tool-use payload."""

    assertions: list[str] = Field(default_factory=list)


def render_prompt(chunk_text: str) -> str:
    """Substitute ``{chunk_text}`` into the prompt template."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{chunk_text}", chunk_text)


def parse_tool_response(response: Any) -> list[str]:
    """Extract the assertion list from an Anthropic Messages response.

    Looks for the first ``tool_use`` block whose name matches :data:`TOOL_NAME`.
    Raises ``ValueError`` if no such block is present or the payload fails schema
    validation — Stage A precision depends on never accepting malformed output.
    """
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        block_type = getattr(block, "type", None)
        block_name = getattr(block, "name", None)
        if block_type == "tool_use" and block_name == TOOL_NAME:
            payload = getattr(block, "input", None) or {}
            return _AssertionList.model_validate(payload).assertions
    raise ValueError(f"No tool_use block named {TOOL_NAME!r} found in response")


def _assertions_from_texts(chunk: Chunk, texts: list[str]) -> list[Assertion]:
    return [
        Assertion.build(
            chunk.doc_id,
            text,
            chunk_id=chunk.chunk_id,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for text in texts
        if text.strip()
    ]


class Extractor(Protocol):
    """Anything that turns a Chunk into a list of Assertion records."""

    def extract(self, chunk: Chunk) -> list[Assertion]: ...


class FixtureExtractor:
    """Returns canned assertion texts keyed by chunk id.

    Intended for hermetic tests. If a chunk id is missing from the fixture map,
    yields an empty list (matching the LLM's "no claims" path).
    """

    def __init__(self, fixtures: Mapping[str, list[str]]) -> None:
        self._fixtures = dict(fixtures)

    def extract(self, chunk: Chunk) -> list[Assertion]:
        texts = self._fixtures.get(chunk.chunk_id, [])
        return _assertions_from_texts(chunk, texts)


class AnthropicExtractor:
    """Calls Anthropic Claude with the atomic-facts prompt and parses the tool response."""

    def __init__(
        self,
        *,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ) -> None:
        import anthropic  # imported lazily so the test suite need not have a real API key

        self._client = client or anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def extract(self, chunk: Chunk) -> list[Assertion]:
        prompt = render_prompt(chunk.text)
        # The Anthropic SDK uses TypedDicts that mypy can't infer from dict literals.
        # Structural shape is validated by SDK at runtime; suppress the overload noise.
        response = self._client.messages.create(  # type: ignore[call-overload]
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
        try:
            texts = parse_tool_response(response)
        except ValueError as exc:
            _log.warning("Atomic-fact extraction returned no usable payload: %s", exc)
            return []
        return _assertions_from_texts(chunk, texts)

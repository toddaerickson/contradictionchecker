"""Tests for the atomic-fact extractor."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from consistency_checker.corpus.chunker import Chunk
from consistency_checker.extract.atomic_facts import (
    PROMPT_PATH,
    TOOL_NAME,
    TOOL_SCHEMA,
    AnthropicExtractor,
    FixtureExtractor,
    _ExtractionPayload,
    parse_tool_response,
    render_prompt,
)
from consistency_checker.extract.schema import hash_id


def make_chunk(text: str = "Revenue grew 12%.", doc_id: str = "doc_a") -> Chunk:
    return Chunk(
        chunk_id=hash_id(doc_id, "0", str(len(text))),
        doc_id=doc_id,
        text=text,
        char_start=0,
        char_end=len(text),
    )


# --- prompt rendering -------------------------------------------------------


def test_prompt_template_exists() -> None:
    assert PROMPT_PATH.exists()


def test_render_prompt_substitutes_chunk_text() -> None:
    out = render_prompt("HELLO WORLD")
    assert "HELLO WORLD" in out
    assert "{chunk_text}" not in out


def test_render_prompt_includes_tool_directive() -> None:
    out = render_prompt("anything")
    assert TOOL_NAME in out


# --- tool response parsing --------------------------------------------------


def test_parse_tool_response_extracts_assertions() -> None:
    block = SimpleNamespace(
        type="tool_use",
        name=TOOL_NAME,
        input={"assertions": ["First claim.", "Second claim."], "definitions": []},
    )
    response = SimpleNamespace(content=[block])
    result = parse_tool_response(response)
    assert result.assertions == ["First claim.", "Second claim."]
    assert result.definitions == []


def test_parse_tool_response_ignores_other_blocks() -> None:
    text_block = SimpleNamespace(type="text", text="some commentary")
    tool_block = SimpleNamespace(
        type="tool_use",
        name=TOOL_NAME,
        input={"assertions": ["Only one."], "definitions": []},
    )
    response = SimpleNamespace(content=[text_block, tool_block])
    result = parse_tool_response(response)
    assert result.assertions == ["Only one."]


def test_parse_tool_response_missing_tool_raises() -> None:
    text_block = SimpleNamespace(type="text", text="no tool call")
    response = SimpleNamespace(content=[text_block])
    with pytest.raises(ValueError, match="No tool_use block"):
        parse_tool_response(response)


def test_parse_tool_response_wrong_tool_name_raises() -> None:
    block = SimpleNamespace(type="tool_use", name="other_tool", input={"assertions": []})
    response = SimpleNamespace(content=[block])
    with pytest.raises(ValueError, match="No tool_use block"):
        parse_tool_response(response)


def test_parse_tool_response_invalid_payload_raises() -> None:
    block = SimpleNamespace(type="tool_use", name=TOOL_NAME, input={"assertions": [1, 2]})
    response = SimpleNamespace(content=[block])
    with pytest.raises(ValidationError):
        parse_tool_response(response)


def test_parse_tool_response_empty_assertions_ok() -> None:
    block = SimpleNamespace(
        type="tool_use", name=TOOL_NAME, input={"assertions": [], "definitions": []}
    )
    response = SimpleNamespace(content=[block])
    result = parse_tool_response(response)
    assert result.assertions == []
    assert result.definitions == []


# --- FixtureExtractor -------------------------------------------------------


def test_fixture_extractor_returns_canned_assertions() -> None:
    chunk = make_chunk("Revenue grew 12% in 2025.")
    extractor = FixtureExtractor(
        {chunk.chunk_id: ["Revenue grew 12% in 2025.", "Growth was year-over-year."]}
    )
    assertions = extractor.extract(chunk)
    assert len(assertions) == 2
    assert {a.assertion_text for a in assertions} == {
        "Revenue grew 12% in 2025.",
        "Growth was year-over-year.",
    }


def test_fixture_extractor_unknown_chunk_returns_empty() -> None:
    chunk = make_chunk()
    extractor = FixtureExtractor({})
    assert extractor.extract(chunk) == []


def test_fixture_extractor_carries_provenance() -> None:
    chunk = Chunk(
        chunk_id="abc123",
        doc_id="doc_xyz",
        text="Some text.",
        char_start=10,
        char_end=20,
    )
    extractor = FixtureExtractor({"abc123": ["Some claim."]})
    [a] = extractor.extract(chunk)
    assert a.doc_id == "doc_xyz"
    assert a.chunk_id == "abc123"
    assert a.char_start == 10
    assert a.char_end == 20


def test_fixture_extractor_drops_blank_strings() -> None:
    chunk = make_chunk()
    extractor = FixtureExtractor({chunk.chunk_id: ["A claim.", "", "   "]})
    assertions = extractor.extract(chunk)
    assert [a.assertion_text for a in assertions] == ["A claim."]


def test_fixture_extractor_produces_stable_ids() -> None:
    chunk = make_chunk()
    extractor = FixtureExtractor({chunk.chunk_id: ["Stable claim."]})
    first = extractor.extract(chunk)
    second = extractor.extract(chunk)
    assert [a.assertion_id for a in first] == [a.assertion_id for a in second]


# --- AnthropicExtractor (mocked client) -------------------------------------


class _FakeAnthropicClient:
    """Minimal stand-in matching the surface the extractor uses."""

    def __init__(self, assertions: list[str]) -> None:
        self._assertions = assertions
        self.last_kwargs: dict[str, object] | None = None

    @property
    def messages(self) -> _FakeAnthropicClient:
        return self

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        block = SimpleNamespace(
            type="tool_use",
            name=TOOL_NAME,
            input={"assertions": self._assertions, "definitions": []},
        )
        return SimpleNamespace(content=[block])


def test_anthropic_extractor_returns_assertions_with_provenance() -> None:
    fake = _FakeAnthropicClient(["Revenue grew 12%.", "Operating margin improved."])
    extractor = AnthropicExtractor(client=fake)  # type: ignore[arg-type]
    chunk = make_chunk("Revenue grew 12%. Operating margin improved.")
    assertions = extractor.extract(chunk)
    assert [a.assertion_text for a in assertions] == [
        "Revenue grew 12%.",
        "Operating margin improved.",
    ]
    assert all(a.chunk_id == chunk.chunk_id and a.doc_id == chunk.doc_id for a in assertions)


def test_anthropic_extractor_uses_tool_choice_and_model() -> None:
    fake = _FakeAnthropicClient([])
    extractor = AnthropicExtractor(client=fake, model="claude-test-model")  # type: ignore[arg-type]
    extractor.extract(make_chunk())
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["model"] == "claude-test-model"
    assert fake.last_kwargs["tool_choice"] == {"type": "tool", "name": TOOL_NAME}


def test_anthropic_extractor_handles_no_tool_block_gracefully() -> None:
    """When the model fails to call the tool, the extractor must log + return empty."""

    class _NoToolClient:
        @property
        def messages(self) -> _NoToolClient:
            return self

        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")])

    extractor = AnthropicExtractor(client=_NoToolClient())  # type: ignore[arg-type]
    assert extractor.extract(make_chunk()) == []


# --- live API test (gated) --------------------------------------------------


@pytest.mark.live
def test_anthropic_extractor_live_call() -> None:
    """Smoke-test against the real Anthropic API. Requires ``ANTHROPIC_API_KEY``."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    extractor = AnthropicExtractor()
    chunk = make_chunk("The Alpha project shipped in Q1 2025. Revenue from Alpha grew 12% YoY.")
    assertions = extractor.extract(chunk)
    assert any("Alpha" in a.assertion_text for a in assertions)


# --- new schema + payload tests (Task 7) ------------------------------------


def test_tool_schema_includes_definitions_array() -> None:
    props = TOOL_SCHEMA["input_schema"]["properties"]
    assert "assertions" in props
    assert "definitions" in props
    assert props["definitions"]["type"] == "array"
    item = props["definitions"]["items"]
    assert item["type"] == "object"
    assert "term" in item["properties"]
    assert "definition_text" in item["properties"]
    assert "containing_sentence" in item["properties"]
    assert set(item["required"]) == {"term", "definition_text", "containing_sentence"}


def test_extraction_payload_parses_combined_response() -> None:
    payload = _ExtractionPayload.model_validate(
        {
            "assertions": ["Revenue grew 12 percent in fiscal 2025."],
            "definitions": [
                {
                    "term": "Borrower",
                    "definition_text": "ABC Corp and its Subsidiaries",
                    "containing_sentence": '"Borrower" means ABC Corp and its Subsidiaries.',
                }
            ],
        }
    )
    assert len(payload.assertions) == 1
    assert len(payload.definitions) == 1
    assert payload.definitions[0].term == "Borrower"


def test_extraction_payload_defaults_empty() -> None:
    payload = _ExtractionPayload.model_validate({"assertions": [], "definitions": []})
    assert payload.assertions == []
    assert payload.definitions == []


def test_assertions_from_payload_routes_definition_to_kind_definition() -> None:
    from consistency_checker.extract.atomic_facts import (
        _assertions_from_payload,
        _DefinitionItem,
    )
    chunk = make_chunk(text='"Borrower" means ABC Corp.', doc_id="doc_a")
    payload = _ExtractionPayload(
        assertions=["Revenue grew 12 percent in fiscal 2025."],
        definitions=[
            _DefinitionItem(
                term="Borrower",
                definition_text="ABC Corp",
                containing_sentence='"Borrower" means ABC Corp.',
            )
        ],
    )
    out = _assertions_from_payload(chunk, payload)
    assert len(out) == 2
    claim = next(a for a in out if a.kind == "claim")
    defn = next(a for a in out if a.kind == "definition")
    assert claim.assertion_text == "Revenue grew 12 percent in fiscal 2025."
    assert claim.term is None
    assert claim.definition_text is None
    assert defn.term == "Borrower"
    assert defn.definition_text == "ABC Corp"
    assert defn.assertion_text == '"Borrower" means ABC Corp.'

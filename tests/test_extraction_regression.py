"""Slow A/B regression test for the combined extractor prompt.

Gates the deferred "should we split into two LLM calls?" decision (see
``docs/superpowers/specs/2026-05-15-definition-inconsistency-detector-design.md``).
Asserts the combined prompt retains atomic-fact recall on the fixture corpus.

Marked ``slow`` AND ``live`` so it requires both an API key and explicit opt-in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from consistency_checker.corpus.chunker import Chunk
from consistency_checker.extract.atomic_facts import AnthropicExtractor
from consistency_checker.extract.schema import hash_id

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "extraction_regression"

RECALL_FLOOR = 0.50  # tolerant — the fixture is small + extraction may paraphrase


@pytest.mark.slow
@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_combined_extractor_recall_within_tolerance() -> None:
    """Asserts the combined prompt still recalls expected anchor phrases.

    The expected file is a list of substrings that should appear (case-insensitive)
    in at least one extracted assertion text. We compute recall as
    (matched anchors) / (total anchors) and assert it stays above
    :data:`RECALL_FLOOR`.

    The fixture is intentionally small; the test is a smoke gate, not a
    statistical benchmark.
    """
    extractor = AnthropicExtractor(model="claude-sonnet-4-6")
    total_expected = 0
    total_recalled = 0

    for txt_path in sorted(FIXTURE_DIR.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        expected_path = txt_path.with_suffix(".expected.json")
        if not expected_path.exists():
            continue
        anchors = [s.lower() for s in json.loads(expected_path.read_text())]

        chunk = Chunk(
            chunk_id=hash_id(txt_path.stem, "0", str(len(text))),
            doc_id=txt_path.stem,
            text=text,
            char_start=0,
            char_end=len(text),
        )
        extracted = extractor.extract(chunk)
        bag = " ".join(a.assertion_text for a in extracted).lower()
        matched = sum(1 for anchor in anchors if anchor in bag)
        total_expected += len(anchors)
        total_recalled += matched

    assert total_expected > 0
    recall = total_recalled / total_expected
    assert recall >= RECALL_FLOOR, (
        f"combined extractor recall regressed: {recall:.2%} "
        f"(floor: {RECALL_FLOOR:.0%}, matched {total_recalled}/{total_expected})"
    )

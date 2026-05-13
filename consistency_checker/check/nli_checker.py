"""Stage A — natural-language inference checker.

Scores each candidate pair as ``contradiction`` / ``entailment`` / ``neutral``
with class probabilities, using a DeBERTa-class MNLI model by default. The
intent of Stage A is recall: drop the O(n²) pair space to something Stage B
(the LLM judge) can afford, while keeping every real contradiction.

Two implementations:

- :class:`FixtureNliChecker` — looks up canned results by ``(premise,
  hypothesis)`` tuple. Required for hermetic CI (the real model is ~800 MB).
- :class:`TransformerNliChecker` — wraps a HuggingFace text-classification
  pipeline against ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``
  (configurable).

:func:`score_bidirectional` runs the checker in both directions and returns the
side with the higher contradiction probability, because MNLI models are not
symmetric in expectation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

NliLabel = Literal["contradiction", "entailment", "neutral"]


@dataclass(frozen=True, slots=True)
class NliResult:
    """Class probabilities plus the argmax label for one premise/hypothesis pair."""

    label: NliLabel
    p_contradiction: float
    p_entailment: float
    p_neutral: float

    @classmethod
    def from_scores(
        cls, *, p_contradiction: float, p_entailment: float, p_neutral: float
    ) -> NliResult:
        scores: dict[NliLabel, float] = {
            "contradiction": p_contradiction,
            "entailment": p_entailment,
            "neutral": p_neutral,
        }
        label: NliLabel = max(scores, key=lambda k: scores[k])
        return cls(
            label=label,
            p_contradiction=p_contradiction,
            p_entailment=p_entailment,
            p_neutral=p_neutral,
        )


class NliChecker(Protocol):
    """Anything that scores a single premise/hypothesis direction."""

    def score(self, premise: str, hypothesis: str) -> NliResult: ...


class FixtureNliChecker:
    """Returns canned :class:`NliResult` keyed by ``(premise, hypothesis)``.

    Unknown pairs fall back to a neutral (1.0) result so tests that only set up
    a subset of pairs do not crash.
    """

    def __init__(self, fixtures: Mapping[tuple[str, str], NliResult]) -> None:
        self._fixtures = dict(fixtures)

    def score(self, premise: str, hypothesis: str) -> NliResult:
        if (premise, hypothesis) in self._fixtures:
            return self._fixtures[(premise, hypothesis)]
        return NliResult.from_scores(p_contradiction=0.0, p_entailment=0.0, p_neutral=1.0)


class TransformerNliChecker:
    """HuggingFace text-classification pipeline wrapper for an MNLI model."""

    DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from transformers import pipeline

        # Typed as Any: transformers' pipeline return is structurally Any in
        # practice (text-classification returns list-of-lists-of-dicts with
        # top_k=None) but typed too narrowly in some SDK versions.
        self._pipe: Any = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            truncation=True,
        )

    def score(self, premise: str, hypothesis: str) -> NliResult:
        outputs = self._pipe([{"text": premise, "text_pair": hypothesis}])
        # With top_k=None, outputs is a list (per input) of lists of {label, score} dicts.
        scores_list = outputs[0]
        by_label = {item["label"].lower(): float(item["score"]) for item in scores_list}
        return NliResult.from_scores(
            p_contradiction=by_label.get("contradiction", 0.0),
            p_entailment=by_label.get("entailment", 0.0),
            p_neutral=by_label.get("neutral", 0.0),
        )


def score_bidirectional(checker: NliChecker, a: str, b: str) -> NliResult:
    """Score in both directions; return the side with higher ``p_contradiction``."""
    forward = checker.score(a, b)
    reverse = checker.score(b, a)
    return forward if forward.p_contradiction >= reverse.p_contradiction else reverse


def passes_threshold(result: NliResult, threshold: float) -> bool:
    """Convenience predicate for gating into Stage B."""
    return result.p_contradiction >= threshold

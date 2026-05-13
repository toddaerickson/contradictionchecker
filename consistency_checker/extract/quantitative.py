"""Quantitative tuple extraction for the numeric short-circuit (ADR-0005).

Pure function. Takes an assertion text, returns a list of :class:`QuantitativeTuple`
records each describing a (metric, value, unit, polarity, scope) shape found in
the text. Step E2 uses these tuples to short-circuit sign-flip pairs before the
LLM judge runs; Step E3 uses range overlaps as a prompt hint when the values
don't flip sign but disagree.

The extractor is intentionally **conservative**:

- No synonym expansion (ADR-0005 non-goal). "Revenue" and "Top-line revenue"
  canonicalise differently, so the short-circuit won't fire across them — the
  LLM judge gets that pair as before.
- No entity-NER (deferred to v0.3). Scope canonicalisation is a regex
  heuristic over time-period phrases.
- No cross-currency conversion. "$12 million" vs. "€12 million" are different
  units; the short-circuit won't fire.

False positives on the short-circuit hurt precision more than false negatives
hurt recall (recall has Stage A + Stage B to fall back on); the bar is set
accordingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Polarity = Literal["pos", "neg", "up", "down", "none"]

# Direction verbs. Lowercased lemma plus common inflections.
UP_VERBS: frozenset[str] = frozenset(
    {
        "grew",
        "grow",
        "grows",
        "growing",
        "rose",
        "rise",
        "rises",
        "rising",
        "risen",
        "increased",
        "increase",
        "increases",
        "increasing",
        "gained",
        "gain",
        "gains",
        "gaining",
        "expanded",
        "expand",
        "expands",
        "expanding",
        "jumped",
        "jump",
        "jumps",
        "jumping",
        "surged",
        "surge",
        "surges",
        "surging",
        "added",
        "adds",
        "adding",
        "climbed",
        "climb",
        "climbs",
        "climbing",
    }
)

DOWN_VERBS: frozenset[str] = frozenset(
    {
        "fell",
        "fall",
        "falls",
        "falling",
        "fallen",
        "declined",
        "decline",
        "declines",
        "declining",
        "decreased",
        "decrease",
        "decreases",
        "decreasing",
        "dropped",
        "drop",
        "drops",
        "dropping",
        "lost",
        "lose",
        "loses",
        "losing",
        "shrank",
        "shrink",
        "shrinks",
        "shrinking",
        "shrunk",
        "contracted",
        "contract",
        "contracts",
        "contracting",
        "slid",
        "slide",
        "slides",
        "sliding",
    }
)

ARTICLES: frozenset[str] = frozenset({"the", "a", "an"})

# Time-period scope phrases. Order matters — more specific first.
_SCOPE_PATTERN = re.compile(
    r"\b("
    r"fiscal\s+year\s+\d{4}"
    r"|fiscal\s+\d{4}"
    r"|FY\s*\d{2,4}"
    r"|Q[1-4]\s+\d{4}"
    r"|H[12]\s+\d{4}"
    r"|year\s+\d{4}"
    r"|in\s+\d{4}"
    r"|during\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)

# Numeric expression. Optional currency prefix; optional sign; mandatory value;
# optional unit suffix. The unit alternation must match the longest first so
# "basis points" wins over "basis".
_NUMBER_PATTERN = re.compile(
    r"(?P<currency>[$€£])?\s*"
    r"(?P<sign>[+-])?\s*"
    r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|basis\s*points|bps|million|billion|thousand|"
    r"dollars?|euros?|pounds?)?",
    re.IGNORECASE,
)

_UNIT_CANON: dict[str, str] = {
    "%": "percent",
    "percent": "percent",
    "bps": "basis_points",
    "basis points": "basis_points",
    "basispoints": "basis_points",
    "million": "million",
    "billion": "billion",
    "thousand": "thousand",
    "dollars": "usd",
    "dollar": "usd",
    "euros": "eur",
    "euro": "eur",
    "pounds": "gbp",
    "pound": "gbp",
}

_CURRENCY_CANON: dict[str, str] = {
    "$": "usd",
    "€": "eur",
    "£": "gbp",
}


@dataclass(frozen=True, slots=True)
class QuantitativeTuple:
    """One numeric claim extracted from an assertion.

    Two tuples match for the short-circuit when ``metric``, ``scope``, and
    ``unit`` are equal and ``polarity`` flips between ``{up, pos}`` and
    ``{down, neg}``. ``"none"`` polarity never participates in a short-circuit.
    """

    metric: str
    value: float
    unit: str | None
    polarity: Polarity
    scope: str | None


def _canonicalise(text: str) -> str:
    tokens = [t.strip(".,;:()") for t in text.lower().split()]
    tokens = [t for t in tokens if t and t not in ARTICLES]
    return " ".join(tokens)


def _canonicalise_scope(scope: str) -> str:
    canon = re.sub(r"\s+", " ", scope.strip().lower())
    canon = re.sub(r"^(in|during)\s+", "", canon)
    return canon


def _canonicalise_unit(unit_token: str | None, currency_token: str | None) -> str | None:
    if unit_token:
        key = re.sub(r"\s+", " ", unit_token.strip().lower())
        canon = _UNIT_CANON.get(key)
        if canon is None:
            return None
        # When the unit is a magnitude word and a currency prefix was present,
        # combine: "$12 million" → "million_usd".
        if canon in {"million", "billion", "thousand"} and currency_token:
            currency = _CURRENCY_CANON.get(currency_token, currency_token)
            return f"{canon}_{currency}"
        return canon
    if currency_token:
        return _CURRENCY_CANON.get(currency_token, currency_token)
    return None


def _find_scope(text: str) -> tuple[str, tuple[int, int]] | None:
    """Return ``(canonicalised_scope, (start, end))`` or ``None``."""
    match = _SCOPE_PATTERN.search(text)
    if not match:
        return None
    return _canonicalise_scope(match.group(1)), match.span()


def _mask_span(text: str, span: tuple[int, int] | None) -> str:
    """Blank out the scope substring so its year digits aren't mistaken for values."""
    if span is None:
        return text
    start, end = span
    return text[:start] + (" " * (end - start)) + text[end:]


def _polarity_from_verb(verb: str) -> Polarity:
    lower = verb.lower()
    if lower in UP_VERBS:
        return "up"
    if lower in DOWN_VERBS:
        return "down"
    return "none"


def _split_around_verb(text: str) -> tuple[str, str, str] | None:
    """Return ``(subject, verb, predicate)`` if any direction verb is present."""
    tokens = text.split()
    for i, token in enumerate(tokens):
        stripped = token.strip(".,;:()'\"`").lower()
        if stripped in UP_VERBS or stripped in DOWN_VERBS:
            subject = " ".join(tokens[:i]).rstrip(".,;:")
            predicate = " ".join(tokens[i + 1 :])
            return subject, stripped, predicate
    return None


def _polarity_from_sign(sign: str | None) -> Polarity:
    if sign == "+":
        return "pos"
    if sign == "-":
        return "neg"
    return "none"


def _value_to_float(value_token: str) -> float | None:
    cleaned = value_token.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_quantities(text: str) -> list[QuantitativeTuple]:
    """Extract all quantitative tuples found in ``text``.

    Returns an empty list when no numeric expressions are present or no
    interpretable (metric, value) pair can be constructed. Pure function — no
    I/O, no global state.
    """
    if not text:
        return []

    scope_info = _find_scope(text)
    scope = scope_info[0] if scope_info else None
    scope_span = scope_info[1] if scope_info else None
    masked_text = _mask_span(text, scope_span)
    out: list[QuantitativeTuple] = []

    parts = _split_around_verb(masked_text)
    if parts is not None:
        subject, verb, predicate = parts
        metric = _canonicalise(subject)
        verb_polarity = _polarity_from_verb(verb)
        for num_match in _NUMBER_PATTERN.finditer(predicate):
            value = _value_to_float(num_match.group("value"))
            if value is None:
                continue
            sign_polarity = _polarity_from_sign(num_match.group("sign"))
            polarity = sign_polarity if sign_polarity != "none" else verb_polarity
            unit = _canonicalise_unit(num_match.group("unit"), num_match.group("currency"))
            if metric and (unit is not None or polarity != "none"):
                out.append(
                    QuantitativeTuple(
                        metric=metric,
                        value=value,
                        unit=unit,
                        polarity=polarity,
                        scope=scope,
                    )
                )
        return out

    # No direction verb. Fall back to "metric is value" / "metric of value" /
    # leading-subject patterns. Metric = tokens before the first number; scope
    # comes from the global match. Polarity is "none" unless an explicit sign
    # is present.
    for num_match in _NUMBER_PATTERN.finditer(masked_text):
        value = _value_to_float(num_match.group("value"))
        if value is None:
            continue
        preceding = masked_text[: num_match.start()]
        metric = _canonicalise(preceding)
        if not metric:
            continue
        sign_polarity = _polarity_from_sign(num_match.group("sign"))
        unit = _canonicalise_unit(num_match.group("unit"), num_match.group("currency"))
        if unit is None and sign_polarity == "none":
            # Nothing to short-circuit on; skip rather than emit a degenerate tuple.
            continue
        out.append(
            QuantitativeTuple(
                metric=metric,
                value=value,
                unit=unit,
                polarity=sign_polarity,
                scope=scope,
            )
        )
    return out


def is_sign_flip(a: QuantitativeTuple, b: QuantitativeTuple) -> bool:
    """Return True if ``a`` and ``b`` should trigger the numeric short-circuit.

    Strict equality on ``metric``, ``scope``, and ``unit`` (so the short-circuit
    only fires on unambiguous cases — false positives here bypass the LLM
    judge's safety net). Polarity must flip between ``{up, pos}`` and
    ``{down, neg}``; ``"none"`` polarity never participates.
    """
    if a.metric != b.metric:
        return False
    if a.scope != b.scope:
        return False
    if a.unit != b.unit:
        return False
    positive = {"up", "pos"}
    negative = {"down", "neg"}
    return (a.polarity in positive and b.polarity in negative) or (
        a.polarity in negative and b.polarity in positive
    )


def find_value_disagreements(
    a_text: str, b_text: str, *, threshold: float = 0.10
) -> list[tuple[QuantitativeTuple, QuantitativeTuple]]:
    """Return cross-pair tuples that share (metric, scope, unit) and disagree
    above the relative ``threshold`` — but **don't** sign-flip (those are owned
    by :func:`is_sign_flip` and the short-circuit). Used by Step E3 to attach a
    structured hint to the judge prompt.

    Relative difference: ``abs(a.value - b.value) / max(abs(a.value), abs(b.value))``.
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    out: list[tuple[QuantitativeTuple, QuantitativeTuple]] = []
    a_tuples = extract_quantities(a_text)
    if not a_tuples:
        return out
    b_tuples = extract_quantities(b_text)
    if not b_tuples:
        return out
    for ta in a_tuples:
        for tb in b_tuples:
            if ta.metric != tb.metric:
                continue
            if ta.scope != tb.scope:
                continue
            if ta.unit != tb.unit:
                continue
            if is_sign_flip(ta, tb):
                continue
            max_abs = max(abs(ta.value), abs(tb.value))
            if max_abs == 0:
                continue
            rel = abs(ta.value - tb.value) / max_abs
            if rel >= threshold:
                out.append((ta, tb))
    return out

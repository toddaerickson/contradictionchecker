"""Term canonicalisation for the definition-inconsistency detector.

Two definition assertions are grouped if their canonical terms match. The
canonical form folds case, strips surrounding whitespace and quote characters,
removes a leading "the ", and trims a trailing plural "s". Intentionally
conservative — we'd rather miss a group (false negative) than merge unrelated
terms (false positive).
"""

from __future__ import annotations

import string

_QUOTE_CHARS = ('"', "'", "“", "”", "‘", "’", "`")  # noqa: RUF001
_STRIP_CHARS = string.punctuation + "".join(c for c in _QUOTE_CHARS if c not in string.punctuation)


def canonicalize_term(raw: str) -> str:
    """Return the canonical, comparison-ready form of a defined-term string."""
    text = raw.strip()
    while text and text[0] in _QUOTE_CHARS:
        text = text[1:]
    while text and text[-1] in _QUOTE_CHARS:
        text = text[:-1]
    text = text.strip().lower()
    if text.startswith("the "):
        text = text[4:]
    if len(text) > 2 and text.endswith("s") and not text.endswith("ss"):
        text = text[:-1]
    return text


def definitions_equivalent(a_text: str, b_text: str) -> bool:
    """True if two definition texts are equal after normalization.

    Normalization (in order): casefold, collapse all whitespace runs to a
    single space, strip leading/trailing punctuation and quote characters.
    INTERNAL punctuation is left untouched, so a mid-string comma that changes
    scope keeps the texts unequal.

    This is deliberately case-INSENSITIVE body comparison, distinct from
    :func:`canonicalize_term`, which is the case-folding term-grouping key and
    is intentionally left unchanged here. Do not unify their case handling.
    """

    def _norm(text: str) -> str:
        return " ".join(text.casefold().split()).strip(_STRIP_CHARS)

    return _norm(a_text) == _norm(b_text)


_LEGAL_SUFFIXES: tuple[str, ...] = (
    "limited", "ltd", "company", "co",
    "corporation", "corp", "lp", "l.p.", "llc", "inc",
)


def normalize_org(label: str) -> str:
    """Return the canonical, comparison-ready org key for ``label``.

    Rules (in order):
      1. casefold; collapse internal whitespace and punctuation to single spaces.
      2. strip a single leading article ('the ').
      3. strip ONE trailing legal-form suffix, but only when at least one other
         significant token would remain. Entity-type words (Trust, Foundation)
         are NOT suffixes — they distinguish organizations and stay in the key.
      4. trim.

    Idempotent: ``normalize_org(normalize_org(x)) == normalize_org(x)``.
    """
    if not label or not label.strip():
        return ""
    chars = []
    for ch in label.casefold():
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")
    text = " ".join("".join(chars).split())
    if text.startswith("the "):
        text = text[4:]
    tokens = text.split()
    if len(tokens) >= 3 and tokens[-2:] == ["l", "p"]:
        tokens = tokens[:-2]
    elif len(tokens) >= 2 and tokens[-1] in _LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens).strip()

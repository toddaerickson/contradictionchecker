"""Term canonicalisation for the definition-inconsistency detector.

Two definition assertions are grouped if their canonical terms match. The
canonical form folds case, strips surrounding whitespace and quote characters,
removes a leading "the ", and trims a trailing plural "s". Intentionally
conservative — we'd rather miss a group (false negative) than merge unrelated
terms (false positive).
"""

from __future__ import annotations

_QUOTE_CHARS = ('"', "'", "“", "”", "‘", "’", "`")


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

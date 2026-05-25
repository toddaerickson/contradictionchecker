"""Pure formatters for the corpus-composition warnings.

Kept separate from cli/main.py so the formatters are unit-testable
without invoking typer, and reusable by the web Stats banner (Task 15).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from consistency_checker.check.definition_terms import normalize_org


@dataclass(frozen=True, slots=True)
class BucketSummary:
    display_label: str
    org_key: str
    doc_count: int


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    known: list[BucketSummary] = field(default_factory=list)
    unknown_count: int = 0


def _pre_suffix_key(label: str) -> str:
    """normalize_org WITHOUT the legal-suffix strip step.

    Why: the fragmentation guard fires when two labels resolve to different
    org_keys ONLY because one carries a legal suffix and the other does not.
    Computing the key before the suffix step lets us detect that case.
    """
    if not label:
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
    return text


def summarize_buckets(rows: list[tuple[str, str | None]]) -> CorpusSummary:
    """rows: [(doc_id, org_label_or_None)]. Returns bucketed summary."""
    known: dict[str, BucketSummary] = {}
    unknown = 0
    for _doc_id, label in rows:
        if not label:
            unknown += 1
            continue
        key = normalize_org(label)
        if key in known:
            prev = known[key]
            known[key] = BucketSummary(prev.display_label, key, prev.doc_count + 1)
        else:
            known[key] = BucketSummary(label, key, 1)
    return CorpusSummary(known=list(known.values()), unknown_count=unknown)


def render_corpus_warning(
    known: list[BucketSummary], unknown_count: int, *, scope_enabled: bool
) -> str:
    if len(known) <= 1:
        return ""
    bucket_strs = []
    for i, b in enumerate(known):
        suffix = " docs" if i == 0 else ""
        bucket_strs.append(f"{b.display_label} ({b.doc_count}{suffix})")
    head = f"⚠ Corpus spans {len(known)} organizations: " + ", ".join(bucket_strs) + "."
    extra = ""
    if unknown_count:
        word = "doc" if unknown_count == 1 else "docs"
        extra = f" Plus {unknown_count} {word} with no identified organization."
    if scope_enabled:
        tail = (
            " Cross-org definition pairs are suppressed (--org-scope);"
            " pass --no-org-scope to compare across orgs."
        )
    else:
        tail = (
            " Cross-org definition pairs are still compared;"
            " pass --org-scope to suppress them."
            " Best results come from one organization's documents at a time."
        )
    return head + extra + tail


def render_fragmentation_warning(known: list[BucketSummary]) -> str:
    n = len(known)
    fragments: list[tuple[BucketSummary, BucketSummary]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = known[i], known[j]
            psa, psb = _pre_suffix_key(a.display_label), _pre_suffix_key(b.display_label)
            if psa and psa == psb:
                fragments.append((a, b))
                continue
            ka, kb = a.org_key.split(), b.org_key.split()
            if (
                ka
                and kb
                and ka[0] == kb[0]
                and (a.org_key.startswith(b.org_key + " ") or b.org_key.startswith(a.org_key + " "))
            ):
                fragments.append((a, b))
    if not fragments:
        return ""
    a, b = fragments[0]
    return (
        f"⚠ Possible fragmentation: '{a.display_label}' and '{b.display_label}' "
        f"resolved to different org keys. If they are the same entity, file a normalize_org issue."
    )


def render_identification_failure_notice(*, failures: int, total: int) -> str:
    if total == 0:
        return ""
    pct = failures / total
    if pct <= 0.20:
        return ""
    return (
        f"⚠ Organization identification failed on {failures} of {total} documents "
        f"({round(pct * 100)}%). Check your provider/API key. "
        "Org warnings below may be incomplete."
    )

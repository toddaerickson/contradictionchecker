"""Mine candidate same-term definition pairs from an ingested corpus into a
review file for human labeling. The output feeds benchmarks/definition_eval/
after the maintainer fills the empty ``label`` field (consistent | divergent)
and deletes junk rows.

Run:
    uv run python -m benchmarks.definition_eval.mine_pairs \
        --db data/store/assertions.db --corpus <corpus_id> \
        --out benchmarks/definition_eval/candidates.jsonl --max 100
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from consistency_checker.check.definition_terms import canonicalize_term
from consistency_checker.extract.schema import Assertion, Corpus
from consistency_checker.index.assertion_store import AssertionStore


def build_candidates(definitions: list[Assertion], max_pairs: int) -> list[dict[str, Any]]:
    """Group definitions by canonical term and enumerate unordered candidate pairs.

    ``label`` is left empty for the maintainer to fill. ``category`` is a heuristic
    seed (``identical`` when the two definition texts match exactly, else ``review``).
    Non-identical ("review") pairs are surfaced first (more label-informative) before
    the ``max_pairs`` cap is applied.
    """
    groups: dict[str, list[Assertion]] = {}
    for a in definitions:
        if a.kind != "definition" or a.term is None or a.definition_text is None:
            continue
        canon = canonicalize_term(a.term)
        if not canon:
            continue
        groups.setdefault(canon, []).append(a)

    candidates: list[dict[str, Any]] = []
    for canon, defs in sorted(groups.items()):
        if len(defs) < 2:
            continue
        for a, b in itertools.combinations(defs, 2):
            ta = (a.definition_text or "").strip()
            tb = (b.definition_text or "").strip()
            if not ta or not tb:
                continue
            lo, hi = sorted([a.assertion_id, b.assertion_id])
            pid = hashlib.sha256(f"{lo}:{hi}".encode()).hexdigest()[:10]
            candidates.append(
                {
                    "pair_id": f"{canon[:24]}_{pid}",
                    "category": "identical" if ta == tb else "review",
                    "term": a.term,
                    "def_a": ta,
                    "def_b": tb,
                    "doc_a": a.doc_id,
                    "doc_b": b.doc_id,
                    "label": "",  # maintainer: "consistent" | "divergent" (or delete the row)
                }
            )
    candidates.sort(key=lambda c: 0 if c["category"] == "review" else 1)
    return candidates[:max_pairs]


def resolve_corpus_id(corpora: list[Corpus], corpus_arg: str | None) -> str | None:
    """Map a user-supplied corpus NAME (or raw id) to the internal corpus_id.

    ``None`` means "all corpora". Raises ValueError with the available names
    if the arg matches neither a name nor an id.
    """
    if corpus_arg is None:
        return None
    by_name = {c.corpus_name: c.corpus_id for c in corpora}
    if corpus_arg in by_name:
        return by_name[corpus_arg]
    ids = {c.corpus_id for c in corpora}
    if corpus_arg in ids:
        return corpus_arg
    available = ", ".join(sorted(by_name)) or "(none)"
    raise ValueError(f"unknown corpus {corpus_arg!r}; available names: {available}")


def mine(db_path: Path, corpus: str | None, max_pairs: int) -> list[dict[str, Any]]:
    with AssertionStore(db_path) as store:
        store.migrate()
        corpus_id = resolve_corpus_id(store.list_corpora(), corpus)
        definitions = [a for a, _org in store.iter_definitions(corpus_id=corpus_id)]
    return build_candidates(definitions, max_pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="corpus NAME (from 'consistency-check corpus list') or id; omit for all corpora.",
    )
    ap.add_argument("--out", type=Path, default=Path("benchmarks/definition_eval/candidates.jsonl"))
    ap.add_argument("--max", type=int, default=100)
    args = ap.parse_args()
    cands = mine(args.db, args.corpus, args.max)
    with args.out.open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    n_review = sum(1 for c in cands if c["category"] == "review")
    print(
        f"Wrote {len(cands)} candidate pairs ({n_review} 'review', rest 'identical') to {args.out}"
    )
    print('Next: set each row\'s "label" to consistent|divergent, delete junk rows,')
    print("then save the curated subset as benchmarks/definition_eval/pairs.jsonl")


if __name__ == "__main__":
    main()

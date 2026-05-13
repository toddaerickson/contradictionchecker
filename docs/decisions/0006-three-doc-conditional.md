# ADR 0006 — Three-document conditional contradictions via graph triangles

**Status**: Accepted

## Context

The v0.1 pipeline is strictly pairwise. It handles cases where assertion A and assertion B directly contradict each other. CONTRADOC also includes conditional contradictions — assertions A, B, C that are individually consistent but where `A ∧ B ⇒ ¬C`. A canonical example:

- A: "All employees get four weeks vacation."
- B: "Engineers are employees."
- C: "Engineers get two weeks vacation."

Each pair is consistent under some interpretation; the conjunction is contradictory. The pairwise pipeline can't surface this — at least one of the three pairs needs a deeper view to flag it.

Two designs were considered:

- **Cluster-prompt by entity NER.** Run a named-entity recognition pass; cluster all assertions referencing the same entity (or coreferring entities); prompt the judge with the cluster and ask whether any subset jointly contradicts the others. High recall in principle but requires entity-NER (deferred to v0.3 per `futureplans.md` #7), depends on coreference quality, and scales as O(cluster_size choose 3) judge calls in the worst case.
- **Graph triangles on FAISS-similarity edges.** Treat each high-similarity gate output as an undirected edge between assertions. Enumerate triangles where all three edges exist. For each triangle, send the three assertions to the judge in one call and ask whether any subset is conditionally contradictory. Reuses the FAISS edges we already compute in Stage A; scales as O(|edges| × avg-degree); doesn't need entity NER.

## Decision

**Graph triangles on FAISS-similarity edges**, opt-in via a new `--deep` CLI flag and a new `Config.enable_multi_party: bool = False`. Default off because:

1. The pairwise pipeline is cheap and handles the common case.
2. Three-doc detection adds an LLM call per triangle on top of the pairwise judges.
3. The recall profile is unclear until we sweep CONTRADOC.

### Schema: `multi_party_findings` table

A new table rather than overloading `findings`. The pair report has been pair-shaped end-to-end since v0.1 (two assertion columns, summary grouped by `(doc_a, doc_b)`); overloading it with N-ary rows would push complexity into every downstream consumer. The sibling table keeps the pair renderer untouched and lets the multi-party renderer evolve independently.

```sql
CREATE TABLE multi_party_findings (
    finding_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    assertion_ids_json TEXT NOT NULL,        -- JSON array, length >= 3
    doc_ids_json TEXT NOT NULL,              -- distinct doc ids, length >= 2
    triangle_edge_scores_json TEXT,          -- JSON list of (a_id, b_id, similarity)
    judge_verdict TEXT,                      -- multi_party_contradiction | not_contradiction | uncertain
    judge_confidence REAL,
    judge_rationale TEXT,
    evidence_spans_json TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_mpf_run ON multi_party_findings(run_id);
CREATE INDEX idx_mpf_verdict ON multi_party_findings(judge_verdict);
```

### Verdict label: `multi_party_contradiction`

New value, distinct from the pairwise `contradiction` and the deterministic `numeric_short_circuit`. Lives on a sibling `MultiPartyVerdictLabel` Literal to keep the existing pair-judge schema untouched.

### Pipeline shape

```
ingest → embed → AnnGate → pairwise NLI → pairwise judge (E2 short-circuit + E3 hint) → audit findings
                              ↓
                          gate output (high-similarity edges)
                              ↓
                          F2 find_triangles  (only when --deep)
                              ↓
                          F3 multi-party judge
                              ↓
                          audit multi_party_findings
```

The triangle pass reads the same `CandidatePair` iterator the pairwise pass consumes — no second FAISS query, no second gate construction.

### Triangle discipline

- Triangles span at least **2 distinct documents** (a triangle inside one document is irrelevant — same intra-doc filter the pairwise gate already applies).
- Triangles are deduplicated by sorted assertion-id tuple before judging.
- A per-run cap (default `max_triangles_per_run = 1000`) prevents combinatorial blow-up on dense corpora. Triangles are sorted by minimum edge similarity descending; the top-N go to the judge.

## Non-goals (v0.2)

- **No entity-NER second pass.** This ADR's known limitation: graph triangles only find conditional contradictions whose three pairwise edges all clear the FAISS gate threshold. Triangles that share an entity but use vocabulary the embedder doesn't relate will be missed. v0.3 #6 (already in `futureplans.md`) extends this with a cluster-by-entity-NER second pass that shares the same `multi_party_findings` table.
- **No N > 3 multi-party detection.** Four-assertion conjunctions are out of scope. If a 4-clique exists, only its constituent triangles are judged.
- **No reuse of the pairwise prompt.** F3 introduces a new prompt template (`check/prompts/judge_multi.txt`) because the question changes from "do these contradict?" to "does any subset of these jointly contradict the others?"
- **No incremental "what changed since last --deep run."** Deep mode is full re-run only in v0.2.

## Consequences

- New SQLite table via migration `0003_multi_party.sql` — additive, doesn't touch existing schema. `findings` stays pair-shaped.
- `JudgeVerdictLabel` (pair) and `MultiPartyVerdictLabel` (triangle) are sibling Literals; no widening of the existing Pair payload schema.
- The pipeline contract grows: `pipeline.check` gains optional `multi_party_judge: MultiPartyJudge | None` arg. When `None`, the new code path is silent — backwards-compatible with v0.1 callers.
- The CLI grows: `consistency-check check --deep` enables the multi-party pass. Without the flag, behaviour is unchanged.
- The report renderer grows: a new "## Multi-document conditional contradictions" section appears only when multi-party findings exist for the run.
- Audit logger grows: new `record_multi_party_finding` and `iter_multi_party_findings` methods. The existing `record_finding` and `iter_findings` are unchanged.
- The "Known limitation" — missed low-similarity triangles — is documented here and in `futureplans.md` #6 so v0.3 reviewers have a clear handoff.

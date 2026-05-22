# Definition-judge precision (identical-text short-circuit + prompt) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the definition detector from spending an LLM call — or returning a wrong `definition_divergent` — on definition pairs whose texts are identical after trivial normalization, and tighten the judge prompt for near-identical cases.

**Architecture:** Add a pure `definitions_equivalent` predicate and call it at the checker layer (`DefinitionChecker.find_inconsistencies`) immediately before the judge call — mirroring the existing `_try_numeric_short_circuit` precedent (ADR-0005). Equivalent pairs resolve to a distinguishable machine verdict `definition_consistent_auto` (no LLM, no migration — `findings.judge_verdict` is free text). The `canonicalize_term` grouping key is left UNCHANGED (its rewrite is deferred). A small labeled eval set + harness guards against regression; the real-corpus run + the existing `consistency-check eval` telemetry are the primary acceptance signals.

**Tech Stack:** Python 3.11+, pytest, Pydantic v2, typer; existing `consistency_checker` package; `uv` for commands.

**Spec:** `docs/superpowers/specs/2026-05-21-canonicalizer-precision-design.md`

---

## File structure

- Modify `consistency_checker/check/definition_terms.py` — add `definitions_equivalent` (pure predicate). `canonicalize_term` untouched.
- Modify `consistency_checker/check/providers/definition_base.py` — add `DEFINITION_CONSISTENT_AUTO` constant.
- Modify `consistency_checker/check/definition_judge.py` — widen `DefinitionJudgeVerdict.verdict` type; add `definition_short_circuit_verdict(a, b)` builder.
- Modify `consistency_checker/check/definition_checker.py` — call the short-circuit before the judge.
- Modify `consistency_checker/pipeline.py` — count short-circuits; add `CheckResult.n_definition_short_circuited`; update call site + log line.
- Modify `consistency_checker/check/prompts/definition_judge_system.txt` — equivalence rule + 2 examples.
- Modify `docs/decisions/0005-numeric-short-circuit.md` — consistent-polarity addendum.
- Create `benchmarks/definition_eval/__init__.py`, `pairs.jsonl`, `harness.py` — labeled regression set.
- Modify `tests/test_definition_terms.py`, `tests/test_definition_checker.py`, `tests/test_pipeline_definition_stage.py` — new tests.
- Create `tests/test_definition_eval_set.py` — hermetic assertion over the `identical` rows.
- Modify `futureplans.md` — roadmap bookkeeping (final task).

---

## Task 1: `definitions_equivalent` pure predicate

**Files:**
- Modify: `consistency_checker/check/definition_terms.py`
- Test: `tests/test_definition_terms.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_definition_terms.py`:

```python
from consistency_checker.check.definition_terms import definitions_equivalent


@pytest.mark.parametrize(
    "a,b,expected",
    [
        # identical
        ("the board of directors of the Corporation", "the board of directors of the Corporation", True),
        # whitespace-only difference
        ("a majority   of the\tdirectors", "a majority of the directors", True),
        # case-only difference
        ("The Board of Directors", "the board of directors", True),
        # surrounding punctuation / quotes only
        ('"the board of directors."', "the board of directors", True),
        ("(the board of directors)", "the board of directors", True),
        # genuine wording difference
        ("a majority of the directors", "two-thirds of the directors", False),
        # mid-string comma that changes scope must NOT be equivalent
        ("directors, officers and employees", "directors officers and employees", False),
    ],
)
def test_definitions_equivalent(a: str, b: str, expected: bool) -> None:
    assert definitions_equivalent(a, b) is expected


def test_definitions_equivalent_is_symmetric() -> None:
    a, b = "The Board.", "the board"
    assert definitions_equivalent(a, b) == definitions_equivalent(b, a)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_definition_terms.py::test_definitions_equivalent -v`
Expected: FAIL — `ImportError: cannot import name 'definitions_equivalent'`.

- [ ] **Step 3: Implement the predicate** — add to `consistency_checker/check/definition_terms.py` (keep the existing `canonicalize_term` and `_QUOTE_CHARS` exactly as they are; add `import string` at the top with the other imports):

```python
import string

_STRIP_CHARS = string.punctuation + "".join(_QUOTE_CHARS)


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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_definition_terms.py -v`
Expected: PASS (the new tests AND the pre-existing `test_canonicalize_term` table, which is unchanged).

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/check/definition_terms.py tests/test_definition_terms.py
git commit -m "feat(definitions): add definitions_equivalent predicate

Pure, case-insensitive body-equivalence check for the definition
short-circuit. Distinct from canonicalize_term (term key, unchanged)."
```

---

## Task 2: `definition_consistent_auto` verdict + builder

**Files:**
- Modify: `consistency_checker/check/providers/definition_base.py`
- Modify: `consistency_checker/check/definition_judge.py`
- Test: `tests/test_definition_judge.py`

- [ ] **Step 1: Add the constant** — in `consistency_checker/check/providers/definition_base.py`, directly below the `DEFINITION_INCONSISTENCY_VERDICTS` definition:

```python
#: Verdict string for the deterministic identical-text short-circuit. NOT a
#: member of ``DefinitionVerdictLabel`` — kept out of the LLM payload's accepted
#: vocabulary on purpose, so the strict ``DefinitionJudgePayload`` schema cannot
#: emit it. Machine-set only; persisted to ``findings.judge_verdict`` (free text).
DEFINITION_CONSISTENT_AUTO = "definition_consistent_auto"
```

- [ ] **Step 2: Write the failing test** — append to `tests/test_definition_judge.py`:

```python
from consistency_checker.check.definition_judge import definition_short_circuit_verdict
from consistency_checker.check.providers.definition_base import DEFINITION_CONSISTENT_AUTO
from consistency_checker.extract.schema import Assertion


def test_definition_short_circuit_verdict() -> None:
    a = Assertion.build("docA", '"Board" means the board of directors.',
                        kind="definition", term="Board", definition_text="the board of directors")
    b = Assertion.build("docB", '"Board" means the board of directors.',
                        kind="definition", term="Board", definition_text="the board of directors")
    v = definition_short_circuit_verdict(a, b)
    assert v.verdict == DEFINITION_CONSISTENT_AUTO
    assert v.confidence == 1.0
    assert v.assertion_a_id == a.assertion_id
    assert v.assertion_b_id == b.assertion_id
    assert "machine-resolved" in v.rationale
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_definition_judge.py::test_definition_short_circuit_verdict -v`
Expected: FAIL — `ImportError: cannot import name 'definition_short_circuit_verdict'`.

- [ ] **Step 4: Widen the dataclass type and add the builder** — in `consistency_checker/check/definition_judge.py`:

First, extend the imports from `definition_base` (the file already imports `DefinitionJudgePayload, DefinitionJudgeProvider, DefinitionVerdictLabel`) to also import the new constant:

```python
from consistency_checker.check.providers.definition_base import (
    DEFINITION_CONSISTENT_AUTO,
    DefinitionJudgePayload,
    DefinitionJudgeProvider,
    DefinitionVerdictLabel,
)
```

Add `Literal` to the `typing` import line (currently `from typing import Protocol`):

```python
from typing import Literal, Protocol
```

Change the `DefinitionJudgeVerdict.verdict` field type from:

```python
    verdict: DefinitionVerdictLabel
```

to:

```python
    verdict: DefinitionVerdictLabel | Literal["definition_consistent_auto"]
```

Add the builder function directly below `definition_uncertain_fallback`:

```python
def definition_short_circuit_verdict(
    a: Assertion, b: Assertion
) -> DefinitionJudgeVerdict:
    """Machine verdict for a pair whose texts are equivalent — no LLM call."""
    return DefinitionJudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict=DEFINITION_CONSISTENT_AUTO,
        confidence=1.0,
        rationale="Definitions textually identical after normalization (machine-resolved, no LLM call).",
        evidence_spans=[],
    )
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_definition_judge.py -v`
Expected: PASS.

- [ ] **Step 6: Type-check**

Run: `uv run mypy consistency_checker`
Expected: no new errors.

- [ ] **Step 7: Commit**

```bash
git add consistency_checker/check/providers/definition_base.py consistency_checker/check/definition_judge.py tests/test_definition_judge.py
git commit -m "feat(definitions): add definition_consistent_auto machine verdict

Distinguishable verdict string for the short-circuit, kept out of the
strict LLM payload vocabulary. Persisted to free-text judge_verdict."
```

---

## Task 3: Wire the short-circuit into `DefinitionChecker`

**Files:**
- Modify: `consistency_checker/check/definition_checker.py`
- Test: `tests/test_definition_checker.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_definition_checker.py`:

```python
from consistency_checker.check.providers.definition_base import DEFINITION_CONSISTENT_AUTO


class _RaisingJudge:
    """A judge that must never be called (proves the short-circuit fired)."""

    def judge(self, a, b):  # type: ignore[no-untyped-def]
        raise AssertionError("judge must not be called for identical definitions")


def test_identical_definitions_short_circuit_without_judge() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "Borrower", "ABC Corp")  # identical assertion_text, different doc
    checker = DefinitionChecker(judge=_RaisingJudge())
    findings = list(checker.find_inconsistencies([a, b]))
    assert len(findings) == 1
    assert findings[0].verdict.verdict == DEFINITION_CONSISTENT_AUTO
    assert findings[0].verdict.confidence == 1.0


def test_divergent_text_still_calls_judge() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "Borrower", "ABC Corp and its Subsidiaries")  # different text
    judge = FixtureDefinitionJudge({})  # returns uncertain fallback when called
    checker = DefinitionChecker(judge=judge)
    findings = list(checker.find_inconsistencies([a, b]))
    assert len(findings) == 1
    assert findings[0].verdict.verdict == "uncertain"  # proves judge WAS called
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_definition_checker.py::test_identical_definitions_short_circuit_without_judge -v`
Expected: FAIL — `AssertionError: judge must not be called for identical definitions` (the checker currently always calls the judge).

- [ ] **Step 3: Implement the short-circuit** — in `consistency_checker/check/definition_checker.py`:

Extend the imports:

```python
from consistency_checker.check.definition_judge import (
    DefinitionJudge,
    DefinitionJudgeVerdict,
    definition_short_circuit_verdict,
)
from consistency_checker.check.definition_terms import (
    canonicalize_term,
    definitions_equivalent,
)
```

Replace the body of `find_inconsistencies`:

```python
    def find_inconsistencies(self, definitions: Sequence[Assertion]) -> Iterator[DefinitionFinding]:
        groups = _group_by_canonical_term(definitions)
        for pair in _enumerate_pairs(groups):
            if definitions_equivalent(pair.a.assertion_text, pair.b.assertion_text):
                verdict = definition_short_circuit_verdict(pair.a, pair.b)
            else:
                verdict = self._judge.judge(pair.a, pair.b)
            yield DefinitionFinding(pair=pair, verdict=verdict)
```

(`DefinitionJudgeVerdict` stays imported — it is referenced elsewhere in the module's type surface.)

- [ ] **Step 4: Run to verify both new tests pass and old ones stay green**

Run: `uv run pytest tests/test_definition_checker.py -v`
Expected: PASS — all tests, including the pre-existing grouping tests.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/check/definition_checker.py tests/test_definition_checker.py
git commit -m "feat(definitions): short-circuit identical-text pairs before the judge

Checker-layer gate (mirrors _try_numeric_short_circuit, ADR-0005);
judge-agnostic, so the fixture path is covered too."
```

---

## Task 4: Count short-circuits in the pipeline run summary

**Files:**
- Modify: `consistency_checker/pipeline.py`
- Test: `tests/test_pipeline_definition_stage.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_pipeline_definition_stage.py`. This mirrors the file's existing `_config` helper and `check(...)` call shape, but builds its own store with two **identical-text** definitions (the shared `stocked_store` fixture has divergent texts, so it can't be reused here):

```python
def test_check_counts_definition_short_circuits(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    store.add_document(Document(doc_id="docB", source_path="/B.txt"))
    text = "the board of directors of the Corporation"
    a = Assertion.build("docA", f'"Board" means {text}.', kind="definition", term="Board", definition_text=text)
    b = Assertion.build("docB", f'"Board" means {text}.', kind="definition", term="Board", definition_text=text)
    store.add_assertions([a, b])

    faiss = FaissStore.open_or_create(
        index_path=config.data_dir / "faiss.idx",
        id_map_path=config.data_dir / "faiss.idmap.json",
        dim=64,
    )
    logger = AuditLogger(store)
    run_id = logger.begin_run()

    class _RaisingJudge:
        def judge(self, a, b):  # type: ignore[no-untyped-def]
            raise AssertionError("identical definitions must short-circuit, not reach the judge")

    result = check(
        config,
        store=store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=logger,
        run_id=run_id,
        definition_checker=DefinitionChecker(judge=_RaisingJudge()),
    )

    assert result.n_definition_short_circuited == 1
    assert result.n_definition_findings == 0  # consistent_auto is not a finding
    assert result.n_definition_pairs_judged == 1  # the pair was still processed
    rows = store._conn.execute(
        "SELECT judge_verdict FROM findings WHERE run_id = ? "
        "AND detector_type = 'definition_inconsistency'",
        (run_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["judge_verdict"] == "definition_consistent_auto"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_pipeline_definition_stage.py::test_check_result_counts_short_circuits -v`
Expected: FAIL — `AttributeError: 'CheckResult' object has no attribute 'n_definition_short_circuited'`.

- [ ] **Step 3: Add the counter** — in `consistency_checker/pipeline.py`:

Add the import near the other `definition_base` import (find the existing `from consistency_checker.check.providers.definition_base import DEFINITION_INCONSISTENCY_VERDICTS` and extend it):

```python
from consistency_checker.check.providers.definition_base import (
    DEFINITION_CONSISTENT_AUTO,
    DEFINITION_INCONSISTENCY_VERDICTS,
)
```

Add the field to `CheckResult` (after `n_definition_findings`):

```python
    n_definition_short_circuited: int = 0
```

Change `_run_definition_pass` to count and return the third value. Its signature return type becomes `tuple[int, int, int]`:

```python
def _run_definition_pass(
    *,
    store: AssertionStore,
    checker: DefinitionChecker,
    audit_logger: AuditLogger,
    run_id: str,
) -> tuple[int, int, int]:
    """Run the definition checker over all stored definitions and log findings.

    Returns ``(n_judged, n_findings, n_short_circuited)``. ``n_judged`` counts
    every pair the checker emitted a verdict for (judged or short-circuited);
    ``n_short_circuited`` is the subset resolved deterministically without an
    LLM call. The NLI gate is bypassed for this stage by design.
    """
    definitions = list(store.iter_definitions())
    n_judged = 0
    n_findings = 0
    n_short_circuited = 0
    for finding in checker.find_inconsistencies(definitions):
        audit_logger.record_definition_finding(run_id, finding=finding)
        n_judged += 1
        if finding.verdict.verdict == DEFINITION_CONSISTENT_AUTO:
            n_short_circuited += 1
        if finding.verdict.verdict in DEFINITION_INCONSISTENCY_VERDICTS:
            n_findings += 1
    return n_judged, n_findings, n_short_circuited
```

Update the call site in `check()` (the `if definition_checker is not None:` block) to unpack three values and default the new one to 0:

```python
    n_definition_pairs_judged = 0
    n_definition_findings = 0
    n_definition_short_circuited = 0
    if definition_checker is not None:
        (
            n_definition_pairs_judged,
            n_definition_findings,
            n_definition_short_circuited,
        ) = _run_definition_pass(
            store=store,
            checker=definition_checker,
            audit_logger=audit_logger,
            run_id=run_id,
        )
```

Extend the `_log.info(...)` summary call: add ` / %d short-circuited` to the format string and `n_definition_short_circuited` as the final arg.

Add the field to the `return CheckResult(...)` at the end of `check()`:

```python
        n_definition_short_circuited=n_definition_short_circuited,
```

- [ ] **Step 4: Run to verify it passes (and the existing definition-stage tests stay green)**

Run: `uv run pytest tests/test_pipeline_definition_stage.py -v`
Expected: PASS — including pre-existing tests (`n_definition_pairs_judged` still counts all processed pairs, so their assertions hold).

- [ ] **Step 5: Type-check**

Run: `uv run mypy consistency_checker`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/pipeline.py tests/test_pipeline_definition_stage.py
git commit -m "feat(definitions): surface n_definition_short_circuited in CheckResult

Run-summary visibility for LLM calls saved by the short-circuit."
```

---

## Task 5: Tighten the definition judge system prompt

**Files:**
- Modify: `consistency_checker/check/prompts/definition_judge_system.txt`
- Test: `tests/test_definition_judge.py` (a load/marker assertion)

- [ ] **Step 1: Write the failing test** — append to `tests/test_definition_judge.py`:

```python
from consistency_checker.check.definition_judge import render_definition_system_prompt


def test_system_prompt_has_equivalence_rule_and_examples() -> None:
    text = render_definition_system_prompt()
    assert "differ only in whitespace, punctuation, capitalization" in text
    assert "Examples:" in text
    assert "Affiliate" in text  # the divergent worked example
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_definition_judge.py::test_system_prompt_has_equivalence_rule_and_examples -v`
Expected: FAIL — the strings are not in the current prompt.

- [ ] **Step 3: Edit the prompt** — in `consistency_checker/check/prompts/definition_judge_system.txt`, after the existing "Be conservative..." paragraph and before the "Return your verdict..." paragraph, insert:

```text
If the two definitions are identical or differ only in whitespace, punctuation,
capitalization, or cross-reference numbering, the verdict is
`definition_consistent`.

Examples:
- Term "Quorum". A: "a majority of the directors then in office". B: "more than
  half of the directors currently serving." -> definition_consistent (same
  threshold, reworded).
- Term "Affiliate". A: "any entity controlling, controlled by, or under common
  control with the Company". B: "any entity that owns more than 50% of the
  Company's voting stock." -> definition_divergent (B narrows "control" to a
  >50%-ownership test; A is broader).
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_definition_judge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/check/prompts/definition_judge_system.txt tests/test_definition_judge.py
git commit -m "feat(definitions): tighten judge prompt with equivalence rule + examples

Backstop for near-identical pairs the short-circuit normalization does
not catch (reordered clauses, cross-reference numbering)."
```

---

## Task 6: ADR-0005 consistent-polarity addendum

**Files:**
- Modify: `docs/decisions/0005-numeric-short-circuit.md`

- [ ] **Step 1: Read the ADR** — open `docs/decisions/0005-numeric-short-circuit.md` and find its "Consequences" section.

- [ ] **Step 2: Append the addendum** — add to the end of the Consequences section:

```text
**Definition extension (2026-05-21).** The same deterministic-pre-judge pattern
now also serves the definition-inconsistency detector: `definitions_equivalent`
(in `check/definition_terms.py`) short-circuits textually identical definition
pairs at the checker layer, emitting the machine verdict string
`definition_consistent_auto`. Note the inverse polarity — the numeric gate emits
a *contradiction*, this one emits a *consistent* (non-finding) verdict that is
filtered out of reports and not counted in `DEFINITION_INCONSISTENCY_VERDICTS`.
Like `numeric_short_circuit`, the label is free-text in `findings.judge_verdict`,
so no migration was required.
```

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0005-numeric-short-circuit.md
git commit -m "docs(adr): record definition short-circuit under ADR-0005"
```

---

## Task 7: Labeled eval set + harness

**Files:**
- Create: `benchmarks/definition_eval/__init__.py`
- Create: `benchmarks/definition_eval/pairs.jsonl`
- Create: `benchmarks/definition_eval/harness.py`
- Test: `tests/test_definition_eval_set.py`

- [ ] **Step 1: Create the package marker** — `benchmarks/definition_eval/__init__.py`:

```python
"""Labeled definition-pair eval set + harness for the definition judge.

Regression guard (NOT the primary precision gate — the set is synthetic and
LLM-graded). See ``docs/superpowers/specs/2026-05-21-canonicalizer-precision-design.md``.
Divergent rows should be expanded with REAL flagged pairs from a user corpus
before treating the numbers as meaningful; the operator reviews all labels.
"""
```

- [ ] **Step 2: Create the starter dataset** — `benchmarks/definition_eval/pairs.jsonl` (one JSON object per line). These are the authored starter rows; the operator extends the divergent categories with real flagged pairs to reach ~32:

```json
{"pair_id": "ident_001", "category": "identical", "term": "Board", "def_a": "the board of directors of the Corporation", "def_b": "the board of directors of the Corporation", "label": "consistent"}
{"pair_id": "ident_002", "category": "identical", "term": "Quorum", "def_a": "a majority of the directors then in office", "def_b": "A majority of the directors then in office.", "label": "consistent"}
{"pair_id": "ident_003", "category": "identical", "term": "Director", "def_a": "a member of the Board", "def_b": "a member of the board", "label": "consistent"}
{"pair_id": "ident_004", "category": "identical", "term": "Secretary", "def_a": "the secretary of the Corporation", "def_b": "the   secretary of the Corporation", "label": "consistent"}
{"pair_id": "ident_005", "category": "identical", "term": "Bylaws", "def_a": "these bylaws, as amended", "def_b": "these bylaws, as amended.", "label": "consistent"}
{"pair_id": "ident_006", "category": "identical", "term": "Chair", "def_a": "the chair of the Board", "def_b": "(the chair of the Board)", "label": "consistent"}
{"pair_id": "reworded_001", "category": "reworded_consistent", "term": "Quorum", "def_a": "a majority of the directors then in office", "def_b": "more than half of the directors currently serving", "label": "consistent"}
{"pair_id": "reworded_002", "category": "reworded_consistent", "term": "Notice", "def_a": "written notice delivered at least ten days before the meeting", "def_b": "notice in writing given no fewer than 10 days prior to the meeting", "label": "consistent"}
{"pair_id": "reworded_003", "category": "reworded_consistent", "term": "Officer", "def_a": "any person elected or appointed by the Board to an office", "def_b": "a person appointed or elected to an office by the Board", "label": "consistent"}
{"pair_id": "reworded_004", "category": "reworded_consistent", "term": "Fiscal Year", "def_a": "the twelve-month period ending December 31", "def_b": "the 12-month period that ends on December 31", "label": "consistent"}
{"pair_id": "scope_001", "category": "scope_divergent", "term": "Affiliate", "def_a": "any entity controlling, controlled by, or under common control with the Company", "def_b": "any entity that owns more than 50% of the Company's voting stock", "label": "divergent"}
{"pair_id": "scope_002", "category": "scope_divergent", "term": "Subsidiary", "def_a": "any entity in which the Company holds any equity interest", "def_b": "any entity in which the Company holds a majority of the voting power", "label": "divergent"}
{"pair_id": "scope_003", "category": "scope_divergent", "term": "Confidential Information", "def_a": "all non-public information disclosed by either party", "def_b": "information marked confidential at the time of disclosure", "label": "divergent"}
{"pair_id": "threshold_001", "category": "threshold_divergent", "term": "Quorum", "def_a": "a majority of the directors then in office", "def_b": "two-thirds of the directors then in office", "label": "divergent"}
{"pair_id": "threshold_002", "category": "threshold_divergent", "term": "Supermajority", "def_a": "at least 60% of the votes cast", "def_b": "at least 75% of the votes cast", "label": "divergent"}
{"pair_id": "incl_001", "category": "inclusion_exclusion_divergent", "term": "Indebtedness", "def_a": "all obligations for borrowed money, including capital leases", "def_b": "all obligations for borrowed money, excluding capital leases", "label": "divergent"}
{"pair_id": "incl_002", "category": "inclusion_exclusion_divergent", "term": "Permitted Investment", "def_a": "investments in government securities and money-market funds", "def_b": "investments in government securities only", "label": "divergent"}
```

- [ ] **Step 3: Create the harness** — `benchmarks/definition_eval/harness.py`:

```python
"""Definition-eval harness: scores the definition checker (short-circuit + judge)
against a labeled pair set, by category.

Regression guard, not the primary precision gate. Run manually with a provider
key configured (reads config.yml like the CLI does):

    uv run python -m benchmarks.definition_eval.harness --baseline benchmarks/definition_eval/baseline.json

Dataset format (JSONL, one object per line):
    {"pair_id": str, "category": str, "term": str,
     "def_a": str, "def_b": str, "label": "consistent" | "divergent"}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion
from consistency_checker.pipeline import make_definition_checker

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "pairs.jsonl"


def _assertion(doc: str, term: str, definition_text: str) -> Assertion:
    return Assertion.build(
        doc,
        f'"{term}" means {definition_text}.',
        kind="definition",
        term=term,
        definition_text=definition_text,
    )


def _predicted_label(verdict: str) -> str:
    # divergent is the positive class; everything else counts as "consistent"
    return "divergent" if verdict == "definition_divergent" else "consistent"


def run(pairs_path: Path, config_path: Path | None) -> dict:
    config = Config.from_yaml(config_path) if config_path and config_path.exists() else Config()
    checker = make_definition_checker(config)
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        a = _assertion("doc_a", row["term"], row["def_a"])
        b = _assertion("doc_b", row["term"], row["def_b"])
        findings = list(checker.find_inconsistencies([a, b]))
        verdict = findings[0].verdict.verdict if findings else "uncertain"
        predicted = _predicted_label(verdict)
        correct = predicted == row["label"]
        by_cat[row["category"]]["total"] += 1
        by_cat[row["category"]]["correct"] += int(correct)
        predictions.append(
            {"pair_id": row["pair_id"], "category": row["category"],
             "label": row["label"], "verdict": verdict, "predicted": predicted,
             "correct": correct}
        )
    summary = {
        cat: {"accuracy": c["correct"] / c["total"], **c}
        for cat, c in sorted(by_cat.items())
    }
    return {"summary": summary, "predictions": predictions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--config", type=Path, default=Path("config.yml"))
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Write per-category summary JSON here (the before/after baseline).")
    args = ap.parse_args()
    result = run(args.pairs, args.config)
    for cat, s in result["summary"].items():
        print(f"{cat:<32} {s['correct']:>3}/{s['total']:<3}  acc={s['accuracy']:.2f}")
    misses = [p for p in result["predictions"] if not p["correct"]]
    if misses:
        print("\nMisses:")
        for m in misses:
            print(f"  [{m['category']}] {m['pair_id']}: label={m['label']} verdict={m['verdict']}")
    if args.baseline:
        args.baseline.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
        print(f"\nWrote baseline to {args.baseline}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the hermetic test** — `tests/test_definition_eval_set.py` (asserts the dataset is well-formed and the `identical` rows resolve via the short-circuit with NO model — uses a raising judge):

```python
"""Hermetic checks on the definition eval set: schema + identical-row determinism."""

from __future__ import annotations

import json
from pathlib import Path

from consistency_checker.check.definition_checker import DefinitionChecker
from consistency_checker.check.providers.definition_base import DEFINITION_CONSISTENT_AUTO
from consistency_checker.extract.schema import Assertion

PAIRS = Path("benchmarks/definition_eval/pairs.jsonl")


def _rows() -> list[dict]:
    return [json.loads(ln) for ln in PAIRS.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_pairs_schema() -> None:
    rows = _rows()
    assert rows, "pairs.jsonl is empty"
    ids = [r["pair_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate pair_id"
    for r in rows:
        assert r["label"] in {"consistent", "divergent"}
        assert {"pair_id", "category", "term", "def_a", "def_b", "label"} <= r.keys()


class _RaisingJudge:
    def judge(self, a, b):  # type: ignore[no-untyped-def]
        raise AssertionError("identical rows must short-circuit, never reach the judge")


def test_identical_rows_short_circuit() -> None:
    checker = DefinitionChecker(judge=_RaisingJudge())
    for r in _rows():
        if r["category"] != "identical":
            continue
        a = Assertion.build("doc_a", f'"{r["term"]}" means {r["def_a"]}.',
                            kind="definition", term=r["term"], definition_text=r["def_a"])
        b = Assertion.build("doc_b", f'"{r["term"]}" means {r["def_b"]}.',
                            kind="definition", term=r["term"], definition_text=r["def_b"])
        findings = list(checker.find_inconsistencies([a, b]))
        assert len(findings) == 1
        assert findings[0].verdict.verdict == DEFINITION_CONSISTENT_AUTO, r["pair_id"]
```

- [ ] **Step 5: Run the hermetic test**

Run: `uv run pytest tests/test_definition_eval_set.py -v`
Expected: PASS. (If an `identical` row fails, its `def_a`/`def_b` differ by more than whitespace/case/edge-punctuation — fix the row, not the code.)

- [ ] **Step 6: Commit**

```bash
git add benchmarks/definition_eval/ tests/test_definition_eval_set.py
git commit -m "test(definitions): labeled eval set + harness (regression guard)

Hermetic schema + identical-row short-circuit checks; live harness scores
the judge by category. Operator extends divergent rows from real corpus."
```

---

## Task 8: Full gate + roadmap bookkeeping

**Files:**
- Modify: `futureplans.md`

- [ ] **Step 1: Run the full CI gate locally**

Run:
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy consistency_checker && uv run pytest -m "not slow and not live"
```
Expected: all green. Fix anything that is not (ruff format will auto-fix with `uv run ruff format .`).

- [ ] **Step 2: Capture the eval baseline (manual, optional pre-merge)** — with a provider key configured, record the before/after numbers per the spec's Measurement section:

```bash
# on main (before): git stash or a separate checkout, then:
uv run python -m benchmarks.definition_eval.harness --baseline /tmp/def_eval_before.json
# on this branch (after):
uv run python -m benchmarks.definition_eval.harness --baseline benchmarks/definition_eval/baseline.json
```
Expected: `identical` accuracy = 1.00 on the after-run (short-circuit); divergent-category accuracy not lower than before. This is the regression guard; the primary gate is the real bylaws-corpus re-run + `uv run consistency-check eval` per the spec.

- [ ] **Step 3: Update `futureplans.md`** — move item #1 from "Eval findings & next levers (2026-05-21)" to the Completed section with this entry (place it at the top of Completed):

```markdown
- **Definition-judge identical-text short-circuit + prompt tightening (item #1, 2026-05-21)**
  — deterministic `definitions_equivalent` short-circuit at the checker layer
  (mirrors `_try_numeric_short_circuit`, ADR-0005) emitting a distinguishable
  `definition_consistent_auto` verdict (no migration; free-text `judge_verdict`);
  tightened `definition_judge_system.txt`; `n_definition_short_circuited` in
  `CheckResult`; labeled regression set under `benchmarks/definition_eval/`.
  Spec: `docs/superpowers/specs/2026-05-21-canonicalizer-precision-design.md`.
  **Deferred:** the `canonicalize_term` rewrite (gated on eval showing real
  distinct-term over-merge; use a recall-safe casefold key, NOT case-sensitive)
  and alias-aware grouping. Item #2 (org grouping) resumes next.
```

Also update the item-#1 bullet under "Eval findings & next levers" to note it shipped (or remove it, leaving the corpus-composition and pairwise notes intact).

- [ ] **Step 4: Commit**

```bash
git add futureplans.md
git commit -m "docs: mark item #1 (definition short-circuit) shipped; record deferrals"
```

---

## Self-review notes (for the executor)

- **Canonicalizer is intentionally untouched.** Do not modify `canonicalize_term` or its test table. If a step seems to call for it, re-read the spec's "Deferred" section.
- **No migration.** If you find yourself writing one, stop — the auto verdict persists to the existing free-text `findings.judge_verdict` column.
- **Keep the two normalizers distinct.** `canonicalize_term` (case-folding term key) and `definitions_equivalent` (casefold body comparison) deliberately differ; do not "unify" them.
- **Live/slow marks:** the harness hits a provider when run, but it is invoked manually (not a pytest test), so no mark is needed on it. The hermetic `tests/test_definition_eval_set.py` must never call a model — it uses a raising judge.

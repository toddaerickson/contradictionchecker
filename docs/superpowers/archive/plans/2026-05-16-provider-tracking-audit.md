# Provider Tracking in Audit Schema — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider column to audit verdicts so users can see which judge (Anthropic/OpenAI/Moonshot) generated each verdict in the web interface.

**Architecture:** Add a `provider` column to both the `findings` and `multi_party_findings` tables via a new migration. Thread the provider name from the config through the pipeline to the audit logger so each verdict is tagged with its judge. Update the report rendering to display provider attribution.

**Tech Stack:** SQLite migrations, Python dataclasses, Pydantic models

---

## File Structure

| File | Action | Responsibility |
|------|--------|-----------------|
| `consistency_checker/index/migrations/0010_provider_tracking.sql` | Create | Add `provider` column to findings and multi_party_findings tables |
| `consistency_checker/audit/logger.py` | Modify | Update Finding/MultiPartyFinding dataclasses; add provider parameter to record_finding/record_multi_party_finding |
| `consistency_checker/check/llm_judge.py` | Modify | Add provider field to JudgeVerdict dataclass |
| `consistency_checker/pipeline.py` | Modify | Pass provider name from config through make_judge() and check() to audit logger |
| `consistency_checker/audit/report.py` | Modify | Display provider in markdown report output |
| `tests/audit/test_logger.py` | Modify | Add tests for provider tracking in findings |
| `tests/test_pipeline.py` | Modify | Test that provider is correctly recorded for all providers (anthropic/openai/moonshot) |

---

## Task 1: Create Migration for Provider Tracking

**Files:**
- Create: `consistency_checker/index/migrations/0010_provider_tracking.sql`

- [ ] **Step 1: Create the migration file**

```bash
cat > consistency_checker/index/migrations/0010_provider_tracking.sql << 'EOF'
-- Add provider column to findings and multi_party_findings tables
-- to track which judge provider (anthropic/openai/moonshot) generated each verdict.

ALTER TABLE findings ADD COLUMN provider TEXT DEFAULT 'anthropic';
ALTER TABLE multi_party_findings ADD COLUMN provider TEXT DEFAULT 'anthropic';

-- Create index for provider filtering
CREATE INDEX IF NOT EXISTS idx_findings_provider ON findings(provider);
CREATE INDEX IF NOT EXISTS idx_multi_party_findings_provider ON multi_party_findings(provider);
EOF
```

- [ ] **Step 2: Verify the migration file was created**

```bash
head -10 consistency_checker/index/migrations/0010_provider_tracking.sql
```

Expected: File contains ALTER TABLE statements for adding provider column.

- [ ] **Step 3: Run tests to verify migration is recognized**

```bash
uv run pytest tests/ -m "not live and not slow" -v
```

Expected: All tests pass (migrations are loaded by AssertionStore).

---

## Task 2: Update Finding and MultiPartyFinding Dataclasses

**Files:**
- Modify: `consistency_checker/audit/logger.py`

- [ ] **Step 1: Add provider field to Finding dataclass**

In `consistency_checker/audit/logger.py`, locate the Finding dataclass (around line 57). Add `provider: str | None = None` field after `created_at`:

```python
@dataclass(frozen=True, slots=True)
class Finding:
    """One judge verdict in a run."""

    finding_id: str
    run_id: str
    assertion_a_id: str
    assertion_b_id: str
    gate_score: float | None
    nli_label: str | None
    nli_p_contradiction: float | None
    nli_p_entailment: float | None
    nli_p_neutral: float | None
    judge_verdict: str | None
    judge_confidence: float | None
    judge_rationale: str | None
    evidence_spans: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    provider: str | None = None
```

- [ ] **Step 2: Add provider field to MultiPartyFinding dataclass**

Locate MultiPartyFinding dataclass (around line 78). Add `provider: str | None = None` field:

```python
@dataclass(frozen=True, slots=True)
class MultiPartyFinding:
    """One multi-document conditional contradiction (ADR-0006, F1)."""

    finding_id: str
    run_id: str
    assertion_ids: list[str]
    doc_ids: list[str]
    triangle_edge_scores: list[tuple[str, str, float]]
    judge_verdict: str | None
    judge_confidence: float | None
    judge_rationale: str | None
    evidence_spans: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    provider: str | None = None
```

- [ ] **Step 3: Update _row_to_finding to extract provider**

Locate `_row_to_finding` function (around line 123). Add provider to the return statement:

```python
def _row_to_finding(row: sqlite3.Row) -> Finding:
    spans_json = row["evidence_spans_json"]
    spans = json.loads(spans_json) if spans_json else []
    return Finding(
        finding_id=row["finding_id"],
        run_id=row["run_id"],
        assertion_a_id=row["assertion_a_id"],
        assertion_b_id=row["assertion_b_id"],
        gate_score=row["gate_score"],
        nli_label=row["nli_label"],
        nli_p_contradiction=row["nli_p_contradiction"],
        nli_p_entailment=row["nli_p_entailment"],
        nli_p_neutral=row["nli_p_neutral"],
        judge_verdict=row["judge_verdict"],
        judge_confidence=row["judge_confidence"],
        judge_rationale=row["judge_rationale"],
        evidence_spans=spans,
        created_at=_parse_ts(row["created_at"]),
        provider=row["provider"],
    )
```

- [ ] **Step 4: Update _row_to_multi_party_finding to extract provider**

Locate `_row_to_multi_party_finding` function (around line 144). Add provider to the return statement:

```python
def _row_to_multi_party_finding(row: sqlite3.Row) -> MultiPartyFinding:
    spans_json = row["evidence_spans_json"]
    spans = json.loads(spans_json) if spans_json else []
    edges_json = row["triangle_edge_scores_json"]
    raw_edges = json.loads(edges_json) if edges_json else []
    edges: list[tuple[str, str, float]] = [(str(a), str(b), float(s)) for a, b, s in raw_edges]
    return MultiPartyFinding(
        finding_id=row["finding_id"],
        run_id=row["run_id"],
        assertion_ids=list(json.loads(row["assertion_ids_json"])),
        doc_ids=list(json.loads(row["doc_ids_json"])),
        triangle_edge_scores=edges,
        judge_verdict=row["judge_verdict"],
        judge_confidence=row["judge_confidence"],
        judge_rationale=row["judge_rationale"],
        evidence_spans=spans,
        created_at=_parse_ts(row["created_at"]),
        provider=row["provider"],
    )
```

- [ ] **Step 5: Update record_finding to accept and store provider**

Locate `record_finding` method (around line 255). Change signature to accept provider, update INSERT:

```python
def record_finding(
    self,
    run_id: str,
    *,
    candidate: CandidatePair,
    nli: NliResult | None,
    verdict: JudgeVerdict,
    provider: str = "anthropic",
) -> str:
    a_id = candidate.a.assertion_id
    b_id = candidate.b.assertion_id
    finding_id = hash_id(run_id, a_id, b_id)
    spans_json = json.dumps(verdict.evidence_spans)
    with self._conn:
        self._conn.execute(
            "INSERT OR REPLACE INTO findings ("
            "finding_id, run_id, assertion_a_id, assertion_b_id, "
            "gate_score, nli_label, nli_p_contradiction, nli_p_entailment, nli_p_neutral, "
            "judge_verdict, judge_confidence, judge_rationale, evidence_spans_json, provider"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                finding_id,
                run_id,
                a_id,
                b_id,
                candidate.score,
                nli.label if nli else None,
                nli.p_contradiction if nli else None,
                nli.p_entailment if nli else None,
                nli.p_neutral if nli else None,
                verdict.verdict,
                verdict.confidence,
                verdict.rationale,
                spans_json,
                provider,
            ),
        )
    return finding_id
```

- [ ] **Step 6: Update record_definition_finding to accept and store provider**

Locate `record_definition_finding` method (around line 292). Add provider parameter:

```python
def record_definition_finding(
    self,
    run_id: str,
    *,
    finding: DefinitionFinding,
    provider: str = "anthropic",
) -> str:
    """Persist a definition-inconsistency finding into the shared findings table."""
    a_id = finding.pair.a.assertion_id
    b_id = finding.pair.b.assertion_id
    finding_id = hash_id(run_id, "definition", a_id, b_id)
    spans_json = json.dumps(finding.verdict.evidence_spans)
    with self._conn:
        self._conn.execute(
            "INSERT OR REPLACE INTO findings ("
            "finding_id, run_id, assertion_a_id, assertion_b_id, "
            "gate_score, nli_label, nli_p_contradiction, nli_p_entailment, nli_p_neutral, "
            "judge_verdict, judge_confidence, judge_rationale, evidence_spans_json, "
            "detector_type, provider"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                finding_id,
                run_id,
                a_id,
                b_id,
                None,
                None,
                None,
                None,
                None,
                finding.verdict.verdict,
                finding.verdict.confidence,
                finding.verdict.rationale,
                spans_json,
                "definition_inconsistency",
                provider,
            ),
        )
    return finding_id
```

- [ ] **Step 7: Update record_multi_party_finding to accept and store provider**

Locate `record_multi_party_finding` method (around line 335). Add provider parameter:

```python
def record_multi_party_finding(
    self,
    run_id: str,
    *,
    assertion_ids: list[str],
    doc_ids: list[str],
    triangle_edge_scores: list[tuple[str, str, float]] | None = None,
    judge_verdict: str,
    judge_confidence: float | None = None,
    judge_rationale: str | None = None,
    evidence_spans: list[str] | None = None,
    provider: str = "anthropic",
) -> str:
    """Insert a row into ``multi_party_findings``."""
    if len(assertion_ids) < 3:
        raise ValueError("multi-party finding needs at least 3 assertion ids")
    sorted_ids = sorted(assertion_ids)
    if len({d for d in doc_ids}) < 2:
        raise ValueError("multi-party finding spans must include >= 2 distinct doc ids")
    finding_id = hash_id(run_id, *sorted_ids)
    edges_payload = (
        json.dumps([[a, b, s] for a, b, s in triangle_edge_scores])
        if triangle_edge_scores is not None
        else None
    )
    spans_payload = json.dumps(evidence_spans if evidence_spans is not None else [])
    with self._conn:
        self._conn.execute(
            "INSERT OR REPLACE INTO multi_party_findings ("
            "finding_id, run_id, assertion_ids_json, doc_ids_json, "
            "triangle_edge_scores_json, judge_verdict, judge_confidence, "
            "judge_rationale, evidence_spans_json, provider"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                finding_id,
                run_id,
                json.dumps(sorted_ids),
                json.dumps(list(doc_ids)),
                edges_payload,
                judge_verdict,
                judge_confidence,
                judge_rationale,
                spans_payload,
                provider,
            ),
        )
        self._conn.executemany(
            "INSERT INTO multi_party_finding_assertions "
            "(finding_id, assertion_id, position) VALUES (?, ?, ?)",
            [(finding_id, aid, idx) for idx, aid in enumerate(sorted_ids)],
        )
    return finding_id
```

- [ ] **Step 8: Run tests to verify dataclass and method changes**

```bash
uv run pytest tests/audit/test_logger.py -v
```

Expected: All logger tests pass (may need to update existing tests to pass provider param).

---

## Task 3: Add Provider Field to JudgeVerdict

**Files:**
- Modify: `consistency_checker/check/llm_judge.py`

- [ ] **Step 1: Add provider field to JudgeVerdict dataclass**

Locate JudgeVerdict dataclass (around line 34). Add provider field:

```python
@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Verdict for one pair plus its provenance — outward-facing surface."""

    assertion_a_id: str
    assertion_b_id: str
    verdict: JudgeVerdictLabel
    confidence: float
    rationale: str
    evidence_spans: list[str] = field(default_factory=list)
    provider: str = "anthropic"
```

- [ ] **Step 2: Update from_payload to include provider**

Update the `from_payload` classmethod (around line 45) to accept and pass provider:

```python
@classmethod
def from_payload(
    cls, a: Assertion, b: Assertion, payload: JudgePayload, *, provider: str = "anthropic"
) -> JudgeVerdict:
    return cls(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict=payload.verdict,
        confidence=payload.confidence,
        rationale=payload.rationale,
        evidence_spans=list(payload.evidence_spans),
        provider=provider,
    )
```

- [ ] **Step 3: Run tests to verify JudgeVerdict changes**

```bash
uv run pytest tests/check/ -k "judge" -v
```

Expected: All judge tests pass.

---

## Task 4: Update LLMJudge to Accept and Pass Provider

**Files:**
- Modify: `consistency_checker/check/llm_judge.py`

- [ ] **Step 1: Locate Judge protocol and LLMJudge class**

Find the Judge Protocol and LLMJudge class (around line 85+).

- [ ] **Step 2: Add provider_name to LLMJudge initialization**

Update the `__init__` method to accept and store provider_name:

```python
class LLMJudge(Judge):
    """Wraps a JudgeProvider with retry logic and verdict marshalling."""

    def __init__(self, provider: JudgeProvider, *, provider_name: str = "anthropic") -> None:
        self._provider = provider
        self._provider_name = provider_name
```

- [ ] **Step 3: Add provider_name property**

Add a property to LLMJudge:

```python
@property
def provider_name(self) -> str:
    return self._provider_name
```

- [ ] **Step 4: Update request_verdict to pass provider_name to JudgeVerdict**

In the method that creates JudgeVerdict objects, pass provider_name:

```python
verdict = JudgeVerdict.from_payload(a, b, payload, provider=self._provider_name)
```

- [ ] **Step 5: Run tests to verify LLMJudge changes**

```bash
uv run pytest tests/check/test_llm_judge.py -v
```

Expected: All tests pass.

---

## Task 5: Update Pipeline to Thread Provider Through to Audit Logger

**Files:**
- Modify: `consistency_checker/pipeline.py`

- [ ] **Step 1: Update make_judge() to initialize LLMJudge with provider_name**

Modify the `make_judge()` function to pass provider_name to LLMJudge constructor:

```python
def make_judge(config: Config) -> Judge:
    if config.judge_provider == "anthropic":
        return LLMJudge(AnthropicProvider(model=config.judge_model), provider_name="anthropic")
    if config.judge_provider == "openai":
        return LLMJudge(OpenAIProvider(model=config.judge_model), provider_name="openai")
    if config.judge_provider == "moonshot":
        return LLMJudge(MoonshotJudgeProvider(model="kimi-k2.6"), provider_name="moonshot")
    raise ValueError(
        f"make_judge(): provider {config.judge_provider!r} has no factory; "
        "construct a FixtureJudge directly in tests."
    )
```

- [ ] **Step 2: Update make_multi_party_judge() to initialize with provider_name**

Similarly update `make_multi_party_judge()`:

```python
def make_multi_party_judge(config: Config) -> MultiPartyJudge:
    if config.judge_provider == "anthropic":
        return LLMMultiPartyJudge(AnthropicMultiPartyProvider(model=config.judge_model), provider_name="anthropic")
    if config.judge_provider == "openai":
        return LLMMultiPartyJudge(OpenAIMultiPartyProvider(model=config.judge_model), provider_name="openai")
    if config.judge_provider == "moonshot":
        return LLMMultiPartyJudge(MoonshotMultiPartyJudgeProvider(model="kimi-k2.6"), provider_name="moonshot")
    raise ValueError(
        f"make_multi_party_judge(): provider {config.judge_provider!r} has no factory"
    )
```

- [ ] **Step 3: Update check() function to pass provider to audit_logger.record_finding()**

Locate where `audit_logger.record_finding()` is called in the `check()` function. Add provider parameter:

```python
audit_logger.record_finding(
    run_id,
    candidate=candidate,
    nli=nli_result,
    verdict=verdict,
    provider=judge.provider_name,
)
```

- [ ] **Step 4: Update check() function to pass provider to audit_logger.record_multi_party_finding()**

Similarly, find where `audit_logger.record_multi_party_finding()` is called and add provider:

```python
audit_logger.record_multi_party_finding(
    run_id,
    assertion_ids=...,
    doc_ids=...,
    triangle_edge_scores=...,
    judge_verdict=...,
    judge_confidence=...,
    judge_rationale=...,
    evidence_spans=...,
    provider=multi_party_judge.provider_name,
)
```

- [ ] **Step 5: Run integration tests**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: All pipeline tests pass.

---

## Task 6: Update Report Rendering to Display Provider

**Files:**
- Modify: `consistency_checker/audit/report.py`

- [ ] **Step 1: Locate finding rendering section in render_report**

Read through `consistency_checker/audit/report.py` to find where individual findings are formatted in markdown (around line 100-150).

- [ ] **Step 2: Add provider to finding detail output**

When rendering each finding's details, add provider information. Look for lines that output verdict details and add provider:

```python
lines.append(f"- **Verdict:** {f.judge_verdict} (Provider: `{f.provider or 'unknown'}`, Confidence: {_format_score(f.judge_confidence)})")
```

- [ ] **Step 3: Add provider info to summary if available**

If the run config or run object is available, optionally display which provider was used in the summary section.

- [ ] **Step 4: Run tests to verify report changes**

```bash
uv run pytest tests/audit/test_report.py -v
```

Expected: All report tests pass (may need to update golden files if they exist).

---

## Task 7: Write Tests for Provider Tracking

**Files:**
- Modify: `tests/audit/test_logger.py`

- [ ] **Step 1: Write test for record_finding with provider**

Add this test to `tests/audit/test_logger.py`:

```python
def test_record_finding_with_provider(store: AssertionStore) -> None:
    """Verify provider is recorded and retrieved correctly."""
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    
    # Create test data
    doc = Document.from_content("test", "test.txt")
    store.insert_document(doc)
    a = Assertion.build(doc.doc_id, "claim A")
    b = Assertion.build(doc.doc_id, "claim B")
    store.insert_assertion(a)
    store.insert_assertion(b)
    
    candidate = CandidatePair(a, b, score=0.9)
    verdict = JudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="contradiction",
        confidence=0.95,
        rationale="They contradict.",
        provider="moonshot",
    )
    
    finding_id = logger.record_finding(
        run_id,
        candidate=candidate,
        nli=None,
        verdict=verdict,
        provider="moonshot",
    )
    
    # Retrieve and verify
    finding = logger.get_finding(finding_id)
    assert finding is not None
    assert finding.provider == "moonshot"
```

- [ ] **Step 2: Write test for record_multi_party_finding with provider**

Add this test to the same file:

```python
def test_record_multi_party_finding_with_provider(store: AssertionStore) -> None:
    """Verify provider is recorded for multi-party findings."""
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    
    # Create test data
    doc = Document.from_content("test", "test.txt")
    store.insert_document(doc)
    assertions = [
        Assertion.build(doc.doc_id, f"claim {i}") for i in range(3)
    ]
    for a in assertions:
        store.insert_assertion(a)
    
    finding_id = logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a.assertion_id for a in assertions],
        doc_ids=[doc.doc_id],
        judge_verdict="multi_party_contradiction",
        judge_confidence=0.9,
        judge_rationale="Triangle detected.",
        provider="openai",
    )
    
    # Retrieve and verify
    finding = logger.get_multi_party_finding(finding_id)
    assert finding is not None
    assert finding.provider == "openai"
```

- [ ] **Step 3: Run audit logger tests**

```bash
uv run pytest tests/audit/test_logger.py -v
```

Expected: All tests pass.

---

## Task 8: Write End-to-End Pipeline Test for Provider Tracking

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write test verifying provider is recorded for all providers**

Add this test to `tests/test_pipeline.py`:

```python
def test_check_records_provider_for_all_providers() -> None:
    """Verify that provider is correctly recorded for anthropic, openai, moonshot."""
    
    for provider_name in ["anthropic", "openai", "moonshot"]:
        # Create config with specific provider
        config = Config(judge_provider=provider_name)
        
        # Create minimal test setup
        store = AssertionStore(":memory:")
        audit_logger = AuditLogger(store)
        
        # Create minimal corpus
        doc = Document.from_content("test", "test.txt")
        store.insert_document(doc)
        assertions = [
            Assertion.build(doc.doc_id, f"claim {i}") for i in range(2)
        ]
        for a in assertions:
            store.insert_assertion(a)
        
        # Record a finding with specific provider
        verdict = JudgeVerdict(
            assertion_a_id=assertions[0].assertion_id,
            assertion_b_id=assertions[1].assertion_id,
            verdict="contradiction",
            confidence=0.9,
            rationale="Test.",
            provider=provider_name,
        )
        
        run_id = audit_logger.begin_run()
        candidate = CandidatePair(assertions[0], assertions[1], score=0.8)
        finding_id = audit_logger.record_finding(
            run_id,
            candidate=candidate,
            nli=None,
            verdict=verdict,
            provider=provider_name,
        )
        
        # Verify provider is recorded
        finding = audit_logger.get_finding(finding_id)
        assert finding is not None
        assert finding.provider == provider_name
```

- [ ] **Step 2: Run pipeline tests**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: All tests pass.

---

## Task 9: Run Full Test Suite and Verify No Regressions

**Files:**
- Test: All tests in the suite

- [ ] **Step 1: Run full test suite (excluding live and slow)**

```bash
uv run pytest -m "not live and not slow" -v
```

Expected: All tests pass.

- [ ] **Step 2: Check type safety**

```bash
uv run mypy consistency_checker
```

Expected: No type errors.

- [ ] **Step 3: Check lint**

```bash
uv run ruff check .
```

Expected: No lint errors.

- [ ] **Step 4: Check format**

```bash
uv run ruff format --check .
```

Expected: Code is properly formatted.

- [ ] **Step 5: Run build**

```bash
uv build
```

Expected: Build succeeds.

---

## Task 10: Create ADR for Provider Tracking

**Files:**
- Create: `docs/decisions/ADR-0011-provider-tracking-audit-schema.md`

- [ ] **Step 1: Create the ADR document**

```bash
cat > docs/decisions/ADR-0011-provider-tracking-audit-schema.md << 'EOF'
# ADR-0011: Provider Attribution in Audit Schema

**Date:** 2026-05-16  
**Status:** Accepted

## Context

With support for multiple judge providers (Anthropic, OpenAI, Moonshot/Kimi), users need visibility into which provider generated each verdict when comparing results across providers in the web interface.

## Decision

Add a `provider` column to both `findings` and `multi_party_findings` tables to track which judge generated each verdict. Thread the provider name from the config through the pipeline to the audit logger. Update report rendering to display provider attribution.

## Rationale

- **Comparison visibility:** Users can easily see which provider generated each verdict
- **Minimal schema:** Single TEXT column with index on both tables
- **No breaking changes:** Provider defaults to "anthropic" for backward compatibility
- **Schema-only:** No new dependencies or major code restructuring

## Consequences

- **Positive:** Users can compare verdicts across providers; future analysis of provider quality differences enabled
- **Negative:** Database schema migration required for existing deployments; report markdown changes slightly

## Related

- ADR-0010: Moonshot judge provider implementation
- ADR-0008: Multi-provider judge abstraction
EOF
```

- [ ] **Step 2: Commit the ADR**

```bash
git add docs/decisions/ADR-0011-provider-tracking-audit-schema.md
git commit -m "docs(adr): ADR-0011 -- provider attribution in audit schema"
```

Expected: Commit succeeds.

---

## Success Criteria

- [ ] Migration 0010 created and applied successfully
- [ ] Finding and MultiPartyFinding dataclasses have provider field
- [ ] record_finding and record_multi_party_finding accept provider parameter
- [ ] JudgeVerdict includes provider field
- [ ] LLMJudge stores and exposes provider_name
- [ ] pipeline.check() threads provider to audit logger
- [ ] Report rendering displays provider attribution
- [ ] All tests pass (no regressions)
- [ ] Code passes mypy, ruff, format checks
- [ ] Build succeeds
- [ ] Tests for provider tracking added
- [ ] ADR created documenting the decision
- [ ] User can see which provider (Anthropic/OpenAI/Moonshot) generated each verdict in the audit data and reports

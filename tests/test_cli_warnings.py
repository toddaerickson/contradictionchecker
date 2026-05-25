from consistency_checker.cli.warnings import (
    BucketSummary,
    render_corpus_warning,
    render_fragmentation_warning,
    render_identification_failure_notice,
    summarize_buckets,
)


def test_summarize_buckets_uses_first_seen_label():
    rows = [
        ("d1", "Acme Foundation, Inc."),
        ("d2", "The Acme Foundation"),
        ("d3", "Beta Trust"),
        ("d4", None),
    ]
    summary = summarize_buckets(rows)
    assert summary.known == [
        BucketSummary(
            display_label="Acme Foundation, Inc.", org_key="acme foundation", doc_count=2
        ),
        BucketSummary(display_label="Beta Trust", org_key="beta trust", doc_count=1),
    ]
    assert summary.unknown_count == 1


def test_render_corpus_warning_scope_off():
    out = render_corpus_warning(
        known=[
            BucketSummary("Acme", "acme", 2),
            BucketSummary("Beta", "beta", 1),
        ],
        unknown_count=1,
        scope_enabled=False,
    )
    assert "Corpus spans 2 organizations" in out
    assert "Acme (2 docs)" in out
    assert "Beta (1)" in out
    assert "1 doc with no identified" in out
    assert "--org-scope to suppress" in out


def test_render_corpus_warning_scope_on():
    out = render_corpus_warning(
        known=[BucketSummary("Acme", "acme", 2), BucketSummary("Beta", "beta", 1)],
        unknown_count=0,
        scope_enabled=True,
    )
    assert "suppressed" in out
    assert "--no-org-scope" in out


def test_render_corpus_warning_single_org_returns_empty():
    out = render_corpus_warning(
        known=[BucketSummary("Acme", "acme", 3)],
        unknown_count=0,
        scope_enabled=False,
    )
    assert out == ""


def test_fragmentation_warning_when_pre_suffix_keys_match():
    out = render_fragmentation_warning(
        [
            BucketSummary("Acme", "acme", 2),
            BucketSummary("Acme Foundation", "acme foundation", 1),
        ]
    )
    assert "fragmentation" in out.lower()


def test_fragmentation_warning_quiet_when_keys_distinct():
    out = render_fragmentation_warning(
        [
            BucketSummary("Acme", "acme", 1),
            BucketSummary("Beta", "beta", 1),
        ]
    )
    assert out == ""


def test_identification_failure_notice_fires_above_20pct():
    out = render_identification_failure_notice(failures=3, total=7)
    assert "failed on 3 of 7" in out


def test_identification_failure_notice_quiet_at_or_below_20pct():
    assert render_identification_failure_notice(failures=1, total=10) == ""
    assert render_identification_failure_notice(failures=2, total=10) == ""

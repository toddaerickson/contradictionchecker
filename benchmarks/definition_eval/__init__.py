"""Labeled definition-pair eval set + harness for the definition judge.

Regression guard (NOT the primary precision gate -- the set is synthetic and
LLM-graded). See ``docs/superpowers/specs/2026-05-21-canonicalizer-precision-design.md``.
Divergent rows should be expanded with REAL flagged pairs from a user corpus
before treating the numbers as meaningful; the operator reviews all labels.
"""

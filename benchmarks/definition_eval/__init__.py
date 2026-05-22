"""Labeled definition-pair eval set + harness for the definition judge.

Regression guard (NOT the primary precision gate -- the set is synthetic and
LLM-graded). See PR #62 and ``docs/decisions/0005-numeric-short-circuit.md``.
Divergent rows should be expanded with REAL flagged pairs from a user corpus
before treating the numbers as meaningful; the operator reviews all labels.
"""

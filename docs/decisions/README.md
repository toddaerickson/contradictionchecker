# Architecture Decision Records

Short records capturing decisions that shape the codebase. New ADRs are appended; existing ADRs are amended only to record status changes (`Accepted`, `Superseded by NNNN`).

Format: one decision per file, ~20–40 lines, sections `Status / Context / Decision / Consequences`.

## Index

| #     | Title                                             | Status   |
|-------|---------------------------------------------------|----------|
| 0001  | [LLM judge provider](0001-llm-judge-provider.md)  | Accepted |
| 0002  | [Embedding model](0002-embedding-model.md)        | Accepted |
| 0003  | [CONTRADOC integration timing](0003-contradoc-integration.md) | Accepted |
| 0004  | [PDF/DOCX loader backend (`unstructured`)](0004-pdf-docx-loaders.md) | Accepted |
| 0005  | [Numeric short-circuit before the LLM judge](0005-numeric-short-circuit.md) | Accepted |
| 0006  | [Three-document conditional contradictions (graph triangles)](0006-three-doc-conditional.md) | Accepted |
| 0007  | [Web UI: FastAPI + HTMX](0007-web-ui.md) | Accepted |
| 0008  | [Persona-aware analysis: a view layer, not forked agents](0008-persona-aware-analysis.md) | Proposed |
| 0009  | [Definition-inconsistency detector](0009-definition-inconsistency-detector.md) | Accepted |
| 0010  | [Moonshot experimental judge](0010-moonshot-experimental-judge.md) | Accepted |
| 0011  | [UI redesign: workflow tabs](0011-ui-redesign-workflow-tabs.md) | Superseded by 0017 |
| 0012  | [Corpus org warning](0012-corpus-org-warning.md) | Accepted |
| 0013  | [Corpus isolation](0013-corpus-isolation.md) | Accepted |
| 0014  | [OCR fallback](0014-ocr-fallback.md) | Accepted |
| 0015  | [Pairwise contradiction detector opt-in](0015-pairwise-opt-in.md) | Accepted |
| 0016  | [Pre-flight cost ceiling](0016-max-cost-ceiling.md) | Accepted |
| 0017  | [UI collapse: single-page shell](0017-ui-collapse.md) | Accepted |
| 0018  | [Remove judge confidence score](0018-remove-judge-confidence.md) | Accepted |
| 0019  | [Ingest as background job](0019-ingest-as-background-job.md) | Accepted |

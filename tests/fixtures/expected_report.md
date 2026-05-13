# Consistency check report

- Run: `run_golden`
- Assertions scanned: 4
- Candidate pairs (gated): 3
- Pairs judged: 3
- Contradictions found: 2

## Summary

| Confidence | NLI(p_contradiction) | Doc A | Doc B | Rationale |
| --- | --- | --- | --- | --- |
| 0.90 | 0.83 | Alpha report | Beta brief | Opposite revenue signs in the same fiscal year. |
| 0.72 | 0.66 | Beta brief | Gamma memo | Different start years for the same Beta initiative. |

## Findings

### Beta brief ⇄ Gamma memo

#### Finding `c83db3ca5de05b8b`

- **Confidence:** 0.72
- **NLI p(contradiction):** 0.66
- **Gate score:** 0.81
- **Evidence spans:** `began in 2024`, `began in 2023`

> **A** (Beta brief): The Beta initiative began in 2024.

> **B** (Gamma memo): The Beta initiative began in 2023.

**Rationale.** Different start years for the same Beta initiative.

### Beta brief ⇄ Alpha report

#### Finding `ae6cc897c1d380ca`

- **Confidence:** 0.90
- **NLI p(contradiction):** 0.83
- **Gate score:** 0.92
- **Evidence spans:** `grew 12%`, `declined 5%`

> **A** (Beta brief): Revenue grew 12% in fiscal 2025.

> **B** (Alpha report): Revenue declined 5% in fiscal 2025.

**Rationale.** Opposite revenue signs in the same fiscal year.


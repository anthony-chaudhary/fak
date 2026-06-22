# Code-quality scorecard

<!-- code-quality-scorecard: 2026-06-22 · process: tools/code_quality_scorecard.py -->

This is the measuring stick for the code-2x program — the code-side counterpart of the docs scorecard. Every number below is re-derived from disk and the Go toolchain by `tools/code_quality_scorecard.py` — no hand-entry. The headline metric is **code-debt**: the count of concrete, mechanical defects (an unformatted file, a `go vet` diagnostic, an egregious god-function, a non-trivial package with zero tests, an untagged honesty claim, an external dependency, an unwitnessed ship). Driving code-debt toward zero is what makes "better code" provable.

> Regenerate: `python tools/code_quality_scorecard.py --markdown --stamp DATE > docs/CODE-QUALITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Code-debt (total HARD defects)** | **7** |
| Composite score | 85.1/100 (grade B) |
| Advisory (soft) signals | 113 |

## Per-KPI

Ten KPIs, each 0–100. `debt` = units of HARD code-debt in that KPI. `godoc` is advisory (it scores but emits no hard debt — doc-comment spam is gaming, not quality).

| KPI | Score | Debt | Detail |
|---|---:|:--:|---|
| `architecture` | 0 | 7 | 3 god-file(s), 4 god-function(s) |
| `hygiene` | 70 | 0 | 5 marker(s) in 3 file(s) |
| `godoc` | 83 | 0 | 1215/1459 exported symbols documented (83.3%) |
| `build` | 100 | 0 | go build ./... exit 0 |
| `vet` | 100 | 0 | clean |
| `format` | 100 | 0 | all files gofmt-clean |
| `deps` | 100 | 0 | stdlib-only, no go.sum |
| `honesty` | 100 | 0 | 72 claims, all tagged |
| `tests` | 100 | 0 | 95/95 non-trivial packages tested (100.0%) |
| `ship_integrity` | 100 | 0 | 14 checkable commit(s) in HEAD~20..HEAD, 0 residual, cleared_rate 1.0 |

## Code-debt work-list

### `architecture` — 7 defect(s), score 0
- god-file cmd/fak/main.go (1587 lines > 1500)
- god-function cmd/fak/main.go:cmdServe (246 lines > 200)
- god-function cmd/fakchat/main.go:main (280 lines > 200)
- god-function cmd/modelbench/main.go:main (458 lines > 200)
- god-function cmd/simpledemo/main.go:main (315 lines > 200)
- god-file internal/ggufload/gguf.go (2298 lines > 1500)
- god-file internal/model/weights.go (1588 lines > 1500)


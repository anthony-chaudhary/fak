# Code-quality scorecard

<!-- code-quality-scorecard: 2026-06-22 · process: tools/code_quality_scorecard.py -->

This is the measuring stick for the code-2x program — the code-side counterpart of the docs scorecard. Every number below is re-derived from disk and the Go toolchain by `tools/code_quality_scorecard.py` — no hand-entry. The headline metric is **code-debt**: the count of concrete, mechanical defects (an unformatted file, a `go vet` diagnostic, an egregious god-function, a non-trivial package with zero tests, an untagged honesty claim, an external dependency, an unwitnessed ship). Driving code-debt toward zero is what makes "better code" provable.

> Regenerate: `python tools/code_quality_scorecard.py --markdown --stamp DATE > docs/CODE-QUALITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Code-debt (total HARD defects)** | **34** |
| Composite score | 76.6/100 (grade C) |
| Advisory (soft) signals | 113 |

## Per-KPI

Ten KPIs, each 0–100. `debt` = units of HARD code-debt in that KPI. `godoc` is advisory (it scores but emits no hard debt — doc-comment spam is gaming, not quality).

| KPI | Score | Debt | Detail |
|---|---:|:--:|---|
| `tests` | 76 | 23 | 72/95 non-trivial packages tested (75.8%) |
| `architecture` | 8 | 6 | 2 god-file(s), 4 god-function(s) |
| `format` | 52 | 4 | 4 unformatted file(s) |
| `ship_integrity` | 75 | 1 | 16 checkable commit(s) in HEAD~20..HEAD, 1 residual, cleared_rate 0.9375 |
| `hygiene` | 70 | 0 | 5 marker(s) in 3 file(s) |
| `godoc` | 80 | 0 | 1318/1637 exported symbols documented (80.5%) |
| `build` | 100 | 0 | go build ./... exit 0 |
| `vet` | 100 | 0 | clean |
| `deps` | 100 | 0 | stdlib-only, no go.sum |
| `honesty` | 100 | 0 | 72 claims, all tagged |

## Code-debt work-list

### `tests` — 23 defect(s), score 76
- non-trivial package has no _test.go: cmd/agentdojoredteam
- non-trivial package has no _test.go: cmd/batchbench
- non-trivial package has no _test.go: cmd/ctxbench
- non-trivial package has no _test.go: cmd/deletioncert
- non-trivial package has no _test.go: cmd/demorace
- non-trivial package has no _test.go: cmd/diagtok
- non-trivial package has no _test.go: cmd/fakchat
- non-trivial package has no _test.go: cmd/fleetbench
- non-trivial package has no _test.go: cmd/gemma4diag
- non-trivial package has no _test.go: cmd/ggufprobe
- non-trivial package has no _test.go: cmd/modelbench
- non-trivial package has no _test.go: cmd/paritybench
- non-trivial package has no _test.go: cmd/pipelinegen
- non-trivial package has no _test.go: cmd/prefixlint
- non-trivial package has no _test.go: cmd/q8bench
- non-trivial package has no _test.go: cmd/q8kernel
- non-trivial package has no _test.go: cmd/radixbench
- non-trivial package has no _test.go: cmd/tpcheck
- non-trivial package has no _test.go: cmd/webbench-convert
- non-trivial package has no _test.go: cmd/webbench-run
- non-trivial package has no _test.go: cmd/webbench-token-measure
- non-trivial package has no _test.go: internal/webbench
- non-trivial package has no _test.go: internal/webbench/browser

### `architecture` — 6 defect(s), score 8
- god-function cmd/fak/main.go:cmdServe (246 lines > 200)
- god-function cmd/fakchat/main.go:main (280 lines > 200)
- god-function cmd/modelbench/main.go:main (458 lines > 200)
- god-function cmd/simpledemo/main.go:main (315 lines > 200)
- god-file internal/ggufload/gguf.go (2298 lines > 1500)
- god-file internal/model/weights.go (1588 lines > 1500)

### `format` — 4 defect(s), score 52
- unformatted (run gofmt -w): cmd/ctxdemo/main.go
- unformatted (run gofmt -w): internal/ggufload/dequant_mxfp4_test.go
- unformatted (run gofmt -w): internal/kernel/explain.go
- unformatted (run gofmt -w): internal/kernel/explain_test.go

### `ship_integrity` — 1 defect(s), score 75
- unwitnessed ship (RESIDUAL) in HEAD~20..HEAD: 3df9627 fix(social): regenerate social-preview card with the fak repo URL (fak tools)


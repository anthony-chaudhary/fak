---
title: "fak code-quality scorecard — the code-debt measuring stick"
description: "fak's deterministic code-quality scorecard: ten KPIs folded into a composite score and the headline code-debt metric, re-derived from disk and Go tooling."
---

# Code-quality scorecard

<!-- code-quality-scorecard: 2026-06-30 · process: tools/code_quality_scorecard.py -->

This is the measuring stick for the code-2x program — the code-side counterpart of the docs scorecard. Every number below is re-derived from disk and the Go toolchain by `tools/code_quality_scorecard.py` — no hand-entry. The headline metric is **code-debt**: the count of concrete, mechanical defects (an unformatted file, a `go vet` diagnostic, an egregious god-function, a non-trivial package with zero tests, an untagged honesty claim, an external dependency, an unwitnessed ship). Driving code-debt toward zero is what makes "better code" provable.

> Regenerate: `python tools/code_quality_scorecard.py --markdown --stamp DATE > docs/CODE-QUALITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Code-debt (total HARD defects)** | **30** |
| Composite score | 75.5/100 (grade C) |
| Advisory (soft) signals | 326 |

## Per-KPI

Ten KPIs, each 0–100. `debt` = units of HARD code-debt in that KPI. `godoc` is advisory (it scores but emits no hard debt — doc-comment spam is gaming, not quality).

| KPI | Score | Debt | Detail |
|---|---:|:--:|---|
| `architecture` | 0 | 18 | 3 god-file(s), 15 god-function(s) |
| `tests` | 98 | 5 | 313/318 non-trivial packages tested (98.4%) |
| `format` | 52 | 4 | 4 unformatted file(s) |
| `deps` | 25 | 3 | 2 external dep(s) + go.sum |
| `hygiene` | 70 | 0 | 5 marker(s) in 3 file(s) |
| `godoc` | 89 | 0 | 6517/7334 exported symbols documented (88.9%) |
| `build` | 100 | 0 | go build ./... exit 0 |
| `vet` | 100 | 0 | clean |
| `honesty` | 100 | 0 | 170 claims, all tagged |
| `ship_integrity` | 100 | 0 | 9 checkable commit(s) in HEAD~20..HEAD, 0 residual, cleared_rate 1.0 |

## Code-debt work-list

### `architecture` — 18 defect(s), score 0
- god-function cmd/fak/accounts.go:runAccounts (231 lines > 200)
- god-function cmd/fak/dispatch_tick.go:evaluateDispatchTick (233 lines > 200)
- god-function cmd/fak/guard.go:cmdGuard (679 lines > 200)
- god-function cmd/fak/loop.go:runLoopRun (248 lines > 200)
- god-function cmd/fak/loop_drive.go:driveGoalSpec (241 lines > 200)
- god-function cmd/fak/main.go:main (293 lines > 200)
- god-function cmd/fak/serve.go:cmdServe (472 lines > 200)
- god-function cmd/fak/usage.go:usageScorecardVerbs (209 lines > 200)
- god-file internal/agent/chat.go (1584 lines > 1500)
- god-function internal/agent/stream.go:CompleteStream (224 lines > 200)
- god-file internal/gateway/gateway.go (2257 lines > 1500)
- god-function internal/gateway/gateway.go:New (209 lines > 200)
- god-function internal/gateway/messages.go:handleAnthropicMessages (208 lines > 200)
- god-function internal/gateway/messages_stream_planner.go:streamAnthropicPlannerLive (228 lines > 200)
- god-file internal/gateway/metrics.go (2612 lines > 1500)
- god-function internal/model/hal.go:tokenHALOutput (205 lines > 200)
- god-function internal/safecommit/safecommit.go:CommitWith (219 lines > 200)
- god-function internal/scorecardpane/hygienegather.go:gatherHygiene (219 lines > 200)

### `tests` — 5 defect(s), score 98
- non-trivial package has no _test.go: experiments/qwen36/gdn-divergence-sensitivity
- non-trivial package has no _test.go: experiments/qwen36/gdn-recurrence-bench
- non-trivial package has no _test.go: internal/auditpane
- non-trivial package has no _test.go: internal/loopfleet
- non-trivial package has no _test.go: internal/propagationscore

### `format` — 4 defect(s), score 52
- unformatted (run gofmt -w): internal/agent/anthropic_compact.go
- unformatted (run gofmt -w): internal/propagationscore/dispatch.go
- unformatted (run gofmt -w): internal/propagationscore/propagationscore.go
- unformatted (run gofmt -w): internal/vcachescore/score.go

### `deps` — 3 defect(s), score 25
- external dependency added: golang.org/x/term
- external dependency added: golang.org/x/sys
- go.sum exists (the zero-dep invariant broke)


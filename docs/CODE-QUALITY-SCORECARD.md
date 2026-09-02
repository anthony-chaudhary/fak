---
title: "fak code-quality scorecard — the code-debt measuring stick"
description: "fak's deterministic code-quality scorecard: ten KPIs folded into a composite score and the headline code-debt metric, re-derived from disk and Go tooling."
---

# Code-quality scorecard

<!-- code-quality-scorecard: 2026-09-01 · process: tools/code_quality_scorecard.py -->

This is the measuring stick for the code-2x program — the code-side counterpart of the docs scorecard. Every number below is re-derived from disk and the Go toolchain by `tools/code_quality_scorecard.py` — no hand-entry. The headline metric is **code-debt**: the count of concrete, mechanical defects (an unformatted file, a `go vet` diagnostic, an egregious god-function, a non-trivial package with zero tests, an untagged honesty claim, an external dependency, an unwitnessed ship). Driving code-debt toward zero is what makes "better code" provable.

> Regenerate: `python tools/code_quality_scorecard.py --markdown --stamp DATE > docs/CODE-QUALITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Code-debt (total HARD defects)** | **46** |
| Composite score | 58.1/100 (grade F) |
| Advisory (soft) signals | 1363 |

## Per-KPI

Ten KPIs, each 0–100. `debt` = units of HARD code-debt in that KPI. `godoc` is advisory (it scores but emits no hard debt — doc-comment spam is gaming, not quality).

| KPI | Score | Debt | Detail |
|---|---:|:--:|---|
| `architecture` | 0 | 35 | 5 god-file(s), 30 god-function(s) |
| `assertion_strength` | 100 | 6 | 32182/32188 test funcs have a reachable failure mechanism (100.0%) |
| `deps` | 25 | 3 | 2 external dep(s) + go.sum |
| `build` | 0 | 1 | go build ./... FAILED |
| `vet` | 85 | 1 | 1 diagnostic(s) |
| `hygiene` | 0 | 0 | 35 marker(s) in 21 file(s) |
| `godoc` | 74 | 0 | 22285/30039 exported symbols documented (74.2%) |
| `format` | 100 | 0 | all files gofmt-clean |
| `honesty` | 100 | 0 | 33 claims, all tagged |
| `tests` | 100 | 0 | 1034/1034 non-trivial packages tested (100.0%) |
| `ship_integrity` | 100 | 0 | 19 checkable commit(s) in HEAD~20..HEAD, 0 residual, cleared_rate 1.0 |

## Debt categories

Stable category identifiers group related HARD findings across detector KPIs.

| Category | Debt | Meaning |
|---|---:|---|
| `modularity` | 35 | Boundaries or units are too large or coupled to change independently. |
| `internal_consistency` | 3 | The implementation contradicts its own declared rules or conventions. |
| `internal_coherence` | 8 | Related implementation pieces do not form a complete, intelligible whole. |

## Code-debt work-list

### `architecture` — 35 defect(s), score 0; categories: `modularity`
- god-function cmd/fak/codex_resume.go:runCodexResume (227 lines > 200)
- god-function cmd/fak/guard.go:cmdManageCommand (1414 lines > 200)
- god-function cmd/fak/guard_child_supervision.go:runGuardChildSupervisedAndReport (322 lines > 200)
- god-function cmd/fak/guard_stophook.go:runGuardStopHook (341 lines > 200)
- god-function cmd/fak/info.go:runInfoOverlayWithDestination (362 lines > 200)
- god-file cmd/fak/loop.go (1502 lines > 1500)
- god-function cmd/fak/loop.go:runLoopRun (209 lines > 200)
- god-function cmd/fak/model_observe.go:runModelObserveBandwidthCollect (303 lines > 200)
- god-function cmd/fak/orchestration_launch.go:launchCodexOrchestrationWorkersWithProfiles (295 lines > 200)
- god-function cmd/fak/resume_watchdog_cli.go:runResumeWatchdogTick (398 lines > 200)
- god-function cmd/fak/session_recover.go:runSessionRecover (221 lines > 200)
- god-function cmd/fak/sync.go:runSync (213 lines > 200)
- god-function cmd/fak/token_defaults.go:collectTokenDefaultsScorecardWithInputs (261 lines > 200)
- god-file cmd/fak/validate.go (1531 lines > 1500)
- god-function cmd/fak/validate.go:runValidate (226 lines > 200)
- god-file cmd/fak/worktree_worker.go (1541 lines > 1500)
- god-function internal/agent/inkernel_decode.go:generateReusedContextWithBias (293 lines > 200)
- god-function internal/agent/inkernel_planner.go:Complete (267 lines > 200)
- god-function internal/archreport/report.go:Analyze (302 lines > 200)
- god-function internal/codexsession/adapter.go:Run (208 lines > 200)
- god-file internal/compute/vulkan.go (1582 lines > 1500)
- god-function internal/gateway/gateway.go:New (265 lines > 200)
- god-function internal/gateway/http.go:handleChatCompletions (275 lines > 200)
- god-function internal/gateway/metrics_render.go:renderMetrics (209 lines > 200)
- god-function internal/gateway/responses.go:handleResponses (238 lines > 200)
- god-function internal/ggufload/quant_q4k_loader.go:QuantModelQ4KProfileOptionsContext (274 lines > 200)
- god-function internal/loopindex/crossaudit_scorecard.go:ScoreCrossAudit (239 lines > 200)
- god-function internal/marketing/aeo.go:aeoLocalizedTerms (300 lines > 200)
- god-function internal/model/config.go:deriveConfigAxes (231 lines > 200)
- god-file internal/model/quant_kquant.go (1711 lines > 1500)
- god-function internal/model/qwen35_batch_decode_metal.go:stepBatchQwen35HybridQ4KMetal (203 lines > 200)
- god-function internal/modelperfobs/bandwidth_collect_nvidia_profile.go:parseNVIDIAProfileCSV (204 lines > 200)
- god-function internal/selfupdate/cmd/selfupdate.go:Run (224 lines > 200)
- god-function internal/selfupdate/cmd/selfupdate_install.go:performSelfUpdate (353 lines > 200)
- god-function tools/videogen/trailer/render.go:sceneFrame (206 lines > 200)

### `assertion_strength` — 6 defect(s), score 100; categories: `internal_coherence`
- zero-assertion test (cannot fail): cmd/caveman-pairwise-judge/main_test.go:148 TestExitForHappyPathReturns
- zero-assertion test (cannot fail): cmd/fak/garden_tick_test.go:352 TestGardenWatchdogHangingChildHelper
- zero-assertion test (cannot fail): cmd/fak/nightrun_ledger_path_test.go:50 TestNightrunLedgerPathHelper
- zero-assertion test (cannot fail): cmd/fak/prepush_build_test.go:862 TestPrepushClaimHelper
- zero-assertion test (cannot fail): examples/security-fleet-governance/main_test.go:70 TestContains
- zero-assertion test (cannot fail): internal/toolcatalog/command_test.go:91 TestCommandAdapterHelperProcess

### `deps` — 3 defect(s), score 25; categories: `internal_consistency`
- external dependency added: golang.org/x/term
- external dependency added: golang.org/x/sys
- go.sum exists (the zero-dep invariant broke)

### `build` — 1 defect(s), score 0; categories: `internal_coherence`
- build failure: # github.com/anthony-chaudhary/fak/cmd/cachedemo

### `vet` — 1 defect(s), score 85; categories: `internal_coherence`
- vet: # github.com/anthony-chaudhary/fak/cmd/fak


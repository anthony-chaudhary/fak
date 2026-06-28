---
title: "fak token-saving-defaults scorecard — is the out-of-the-box token economy amazing?"
description: "fak's deterministic token-saving-defaults scorecard: which stacking token-saving methods are ON by default on the fak guard / fak serve Anthropic passthrough, whether the high-value low-loss savers are turned on out of the box, and whether every default is honestly noted and locked against regression — re-derived from the entrypoint source."
---

# Token-saving-defaults scorecard — is fak's out-of-the-box token economy amazing?

<!-- token-defaults-scorecard · process: fak token-defaults-scorecard --markdown -->

The question a cost-conscious operator asks the moment they run `fak guard -- claude` / `fak serve`: **of every token-saving method fak knows how to stack, which ones are ON by default — and are the high-value, low-loss ones turned on out of the box, or left dark behind a flag nobody flips?** Every number below is re-derived from the entrypoint source (`cmd/fak/guard.go`, `cmd/fak/serve.go`, the `Default*` constants in `internal/gateway/gateway.go`, and `internal/gateway/messages.go`) by `fak token-defaults-scorecard` — a lever's on/off state is the binary's real behavior, never a claim in the roster. The headline metric is **token-defaults-debt**: the count of concrete defects — a high-value saver left off, an on-by-default saver with no honest note, a default no test locks, a front door out of step. Driving it to zero means a user who runs fak with no flags gets the full stack of safe savings, each honestly labeled, none able to regress unnoticed.

> Regenerate: `go run ./cmd/fak token-defaults-scorecard --markdown > docs/serving/token-defaults-scorecard.md`

## Headline

| Metric | Value |
|---|---|
| **Token-defaults-debt (total HARD defects)** | **0** |
| Composite score | 97.9/100 (grade A) |
| Savers stacked on by default | 5/6 |
| Groups | stack 94 · honesty 100 · regression 100 · parity 100 |
| Advisory (soft) signals | 1 |

## Per-lever status — where each token-saving method stands

`class`: **lossless** = zero model-visible change (must be on); **bounded** = lossy but an in-code guard keeps the model's working set intact (high-value → should be on, with a note); **optin** = broader blast radius (correctly off, must carry a documented gate). `gated` = an off lever documents why; `noted` = an on bounded lever documents what it sheds + cache-safety; `locked` = a test pins the default.

| Lever | Class | Default | Witness | Blocker | Flag | Gated | Noted | Locked |
|---|---|:--:|:--:|---|---|:--:|:--:|:--:|
| provider_cache — provider prompt-cache prefix (byte-faithful passthrough) | lossless | **ON** | ✓ | — | `(structural)` | · | ✓ | ✓ |
| toolfloor — tool-floor pruning (drop provably-unreachable tool defs) | lossless | **ON** | ✓ | — | `(structural)` | · | ✓ | ✓ |
| vdso — vDSO dedup fast path (collapse identical calls) | lossless | **ON** | ✓ | — | `--vdso` | · | ✓ | ✓ |
| compacthistory — history compaction (drop the un-cacheable middle past the budget) | bounded | **ON** | ✓ | — | `--compact-history-budget` | · | ✓ | ✓ |
| elideresult — oversized-result elision (shrink a scrolled-past tool_result to head+tail) | bounded | **ON** | ✓ | — | `--elide-result-bytes` | · | ✓ | ✓ |
| ctxview — ctxplan O(1) planned view (re-materialize history under a budget) | optin | **OFF** | ✓ | witnessed_gated | `--ctx-view-budget` | ✓ | · | ✓ |

## KPIs

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| stack | `stacking_depth` | 83 | 0 | 5/6 token-saving methods stacked on by default out of the box |
| stack | `lossless_stack` | 100 | 0 | 3/3 lossless savers on by default |
| stack | `high_value_defaults` | 100 | 0 | 2/2 demonstrably-safe bounded-loss savers on by default |
| honesty | `witness_status` | 100 | 0 | no off-by-default high-value savers remain — every bounded-loss saver defaults on |
| honesty | `dark_lever_gated` | 100 | 0 | 1/1 off-by-default levers carry a documented gate |
| honesty | `default_notes` | 100 | 0 | 2/2 on-by-default bounded savers carry an honest loss note |
| regression | `default_on_locked` | 100 | 0 | 5/5 on-by-default savers pinned by a regression sentinel |
| parity | `entrypoint_parity` | 100 | 0 | front doors agree + servewiring verdicts track the real defaults |

## Token-defaults-debt work-list

No token-defaults-debt: every stacking saver fak can safely default is on out of the box, honestly noted, and locked against regression. The lone off-by-default lever (`ctxview`, the opt-in planned view) is correctly gated behind a watched-live witness — the tracked next default to turn on once that gate clears. 🎉

---
title: "FrontierSWE scoring parity: the Go runtime vs the Python oracle"
description: "How internal/frontierswe's Go scorer (ExtractScore + GatedScore) maps field-for-field onto Proximal Labs' scripts/score_from_reward.py, the published leaderboard oracle, with the committed reward.json parity fixtures and the one known divergence (anti-cheat zeroing)."
---

# FRONTIERSWE-SCORING-PARITY — the Go scorer vs the published oracle

> A fak FrontierSWE run is only comparable to the leaderboard if it produces the
> **same number**. FrontierSWE ships a reference scorer,
> [`scripts/score_from_reward.py`](https://github.com/Proximal-Labs/frontier-swe/blob/main/scripts/score_from_reward.py),
> that turns a trial's `reward.json` into the published leaderboard score. This
> doc records how the Go port in `internal/frontierswe/score.go` maps onto that
> oracle, so the port can be asserted bit-for-bit on committed fixtures.

The oracle stays the source of truth. The Go scorer is the **runtime** — it lets
a fak run self-score with no Python dependency — and `score_test.go` pins it to
the oracle by replaying the oracle's own output over a corpus of committed
`reward.json` fixtures.

## Two steps, mirrored field-for-field

| Python (oracle) | Go (`internal/frontierswe/score.go`) |
| --- | --- |
| `extract_score(reward_data, task, ssim_threshold=0.99) -> (correctness, speedup\|None)` | `ExtractScore(reward, task) (float64, *float64)` / `ExtractScoreSSIM(reward, task, ssim)` |
| `compute_gated_score(correctness, speedup, task) -> score` | `GatedScore(correctness, speedup, task) float64` |
| `category_of(task)` over the three name lists | `CategoryOf(name)` over `catalog.go` (cross-checked in `score_test.go`) |

`ExtractScore` uses the library default `ssim_threshold = 0.99` (the value
`extract_score`'s signature defaults to). The reference **CLI** defaults to
`0.95`; pass `ExtractScoreSSIM(..., 0.95)` to reproduce a CLI run at that
threshold. Both thresholds are exercised by fixtures (`revideo_lowssim_*`).

## The gate formulas (compute_gated_score)

| Family / task | Gate |
| --- | --- |
| implementation | `correctness` (unchanged) |
| performance | `0.5 * correctness`, or `0.5 + 0.5 * speedup` once `correctness == 1.0` |
| ml_research | raw reward; `frogsgame-rl` is `board_count / 500` |
| notebook-compression (special) | `speedup` if fully correct, else `0` |
| libexpat-to-x86asm (special) | **uncapped** `0.5 + 0.5 * speedup` once fully correct, else `0.5 * correctness` |

The `libexpat-to-x86asm` branch is checked **before** the generic performance
branch in both the Python and the Go, because its full-correctness payoff is
uncapped (a speedup `> 1.0` pushes the score above `1.0`) — `libexpat_full`
fixture: `correctness = 1.0`, `speedup = 0.9` -> score `0.95`; a `speedup = 3.0`
gives `2.0`.

The `libexpat-to-x86asm` partial-correctness reconstruction uses the per-module
weight table (mirroring `_LIBEXPAT_MODULE_WEIGHTS`):

| module | weight |
| --- | --- |
| smoke_tier | 1 |
| basic_tests | 3 |
| ns_tests | 2 |
| misc_tests | 1 |
| alloc_tests | 2 |
| nsalloc_tests | 1 |
| acc_tests | 0 (excluded) |

`correctness = (sum over modules of (passed/total) * weight) / sum(weights)`.

## Parity fixtures

`testdata/frontierswe/reward/*.json` are committed `reward.json` fixtures — at
least one per category, both special cases, and every non-trivial `extract_score`
path (subscore vs test-count, hard-fail partial credit, gate-passed speedup,
token-match-rate, parity-pass-rate, the `a or b or 0.0` short-circuit, …). The
sidecar `testdata/frontierswe/reward/expected.json` records, per fixture, the
**oracle's** `correctness`, `speedup`, and gated `score`, produced by running
`scripts/score_from_reward.py` on each fixture. `TestGatedScoreParity` asserts
the Go scorer reproduces all three for every fixture; `TestCategoryListsMatchOracle`
and `TestGateFormulas` pin the category lists and gate arithmetic to the oracle
independently of any fixture.

## Known divergence: anti-cheat zeroing

The reference `compute_gated_score(correctness, speedup, task)` has **no
anti-cheat parameter** — the upstream pipeline applies the anti-cheat zeroing
(a trial flagged in `scoring/anticheat.json` scores `0`) as a separate step
*outside* the function this script exposes. The Go port keeps the pure gate
faithful to the oracle and layers the zeroing on top:

- `GatedScore(correctness, speedup, task)` — exact port of `compute_gated_score`,
  no anti-cheat. This is what the parity test asserts against the oracle.
- `GatedScoreAntiCheat(correctness, speedup, task, antiCheatFlagged)` — `0` when
  `antiCheatFlagged`, else `GatedScore(...)`. This is the runtime's composition
  point.
- `Score(reward, task, antiCheatFlagged)` — the end-to-end convenience
  (extract -> gate -> anti-cheat) a fak run calls to self-score a trial.

The flag itself comes from reading `scoring/anticheat.json` for the trial, which
is a run-time concern outside this scorer (the scorer takes the resolved boolean).
This is the one and only place the Go runtime intentionally does **not** match the
single-function oracle, and it is additive: with `antiCheatFlagged = false`,
`Score` equals the oracle on every fixture run at the library-default SSIM.

## Score-parity gate before any TTS claim

The FrontierSWE value claim is "same quality, faster time-to-solution." C11 makes
that mechanical with `ScoreParity(raw_trials, fak_trials)`: the fak arm may not
emit a TTS-win claim unless its trial distribution is at least as good as raw.

The predicate is distribution-level, not a single lucky trial:

- `fak.avg_score >= raw.avg_score`
- `fak.best_score >= raw.best_score`
- `fak.correct_count >= raw.correct_count`, where correct means `correctness == 1.0`
- if raw has full-correct speedup trials, `fak.avg_speedup >= raw.avg_speedup` and
  `fak.best_speedup >= raw.best_speedup`

All comparisons use the gate tolerance recorded in the
`fak.frontierswe.score-parity.v1` report. A failure emits the closed refusal token
`FRONTIERSWE_SCORE_PARITY_FAILED`, so a faster-but-worse fak run is refused rather
than silently reported.

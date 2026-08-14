---
title: "Dispatch-worker context-budget trap: the derived-budget fix and its measured-baseline residual (2026-07-08)"
description: "The #2972 crash-loop where the claude dispatch worker's guard budget (48000) sat below the workers' ~62K irreducible baseline prompt, so every worker was born over-budget and drained BUDGET_CONTEXT_EXHAUSTED on turn 1. Records the shipped derived-budget fix (min(baseline*2, HardContextCap-OutputReserve) = 124000), why a flat constant goes stale, and the #3522 follow-up that turned the frozen 62000 baseline into a MEASURED (floored, raising-only) one that sizes the launch constituents readable at launch — closing the organic-orientation-growth hole while the runtime-attribution witness for the full dynamic prompt remains the last checkable step."
slug: dispatch-worker-context-budget-trap
date: 2026-07-08
---

# Dispatch-worker context-budget trap (2026-07-08)

This is the tracking note referenced by commit `8a0fcffbb` (`fix(dispatchworker):
derive claude guard context-budget from ctxplan window, not a flat constant (fak
cmd)`). It records the trap, the shipped fix, and the one residual the fix does
**not** close — so the follow-up is witnessed, not lost.

## The trap (#2972)

A dispatch claude worker is launched as `claude -p /dos-dispatch-loop --lane <lane>`
fronted by `fak manage` (`cmd/dispatchworker/guard.go`). Guard is handed
`--context-budget-tokens N`, which seeds the per-session `ContextTokensLeft`.

The load-bearing semantics (`internal/session/usage.go` `DebitUsage`): **every turn
debits that turn's ENTIRE resident context window** (prompt + cache_read +
cache_creation — the whole window, not the newly-added delta) from the budget. When
it reaches `<= 0` the session drains with `BUDGET_CONTEXT_EXHAUSTED`. So for a
single-issue `-p` worker the budget behaves as a **per-turn resident-window ceiling**:
if turn 1's window already exceeds the budget, the worker is *born over-budget* and
never runs a productive turn.

The old value was a flat `48000` — a compaction shed-line mis-wired as a drain
ceiling. It sat **below** the workers' ~62K irreducible baseline prompt:

- the issue body,
- AGENTS.md / llms.txt / CLAUDE.md orientation,
- injected fleet memory,
- the ~40K `startup.json` `route` blob,
- Claude Code's own system prompt.

Result chain: turn-1 window (~62K) > budget (48000) → `BUDGET_CONTEXT_EXHAUSTED` on
turn 1 → guard's 2 restarts burned → raw 409 → child exit 1 → `CHILD_CRASH`. Fleet
ship rate collapsed to **2–9%**.

> **SUPERSEDED (turn-starvation).** The derivation recorded in the next section —
> `min(baseline × 2, HardContextCap − OutputReserve)` = 124000 — was itself defective and
> has been replaced. It cured the *birth* wall but created a *turn* wall: the seeded
> budget is a CUMULATIVE allowance (`internal/session/usage.go:99` debits each turn's
> ENTIRE resident window), so `baseline × k` funds at most `k` turns, and clamping that
> cumulative total to the per-turn window ceiling made the clamp binding for every
> `k ≥ 3` (`min(62000×k, 168000) = 168000`), pinning every child at ~2 turns no matter
> what the factor was. Live witness
> `.dispatch-runs/resolve-5103-20260726-022520.log`: 6 turns at ctx 68.4k→83.2k,
> `context_tokens=124000`, `restart_exhausted count=3
> dominant_cause=BUDGET_CONTEXT_EXHAUSTED` at 5m42s of a 29m runway → 409 → exit 1,
> reproduced on 120/120 worker witnesses as `CLAIM_NO_COMMIT`. The shipped derivation is
> now `max(HardContextCap − OutputReserve, baseline) × claudeGuardTurnsPerEpoch`
> (= `max(168000, 62000) × 12` = **2016000**): the window ceiling bounds the PER-TURN
> resident, where it is dimensionally correct, and the turn count scales the cumulative
> total. The goldens below (124000, `budget <= ceiling`, strict monotone-in-baseline)
> are superseded by turn-unit assertions in the same two tests.
>
> Corollary recorded here because it cost real debugging time: the per-turn stderr nudge
> renders `ctx:<resident>/<compact-history-budget>` (`internal/gateway/debug_stats.go`
> `formatCompactionBudgetNudge`) — the denominator is the 96000 COMPACT shed-line, never
> the session budget. A worker therefore reads `ctx:83.2k/96.0k dist:12.8k-to-compact`
> ("compaction is close") on the very turn its unrelated cumulative budget kills it, and
> the compaction fold correctly logs `bailed: under_budget` because it compares the
> per-turn *suffix after the cache anchor* (`internal/agent/anthropic_compact.go:372`)
> against 96000. The two "budgets" were never on the same scale.

## The shipped fix (`8a0fcffbb`) — superseded, see above

Replace the flat constant with a launch-time **derivation** so no single flat value
can silently fall below the baseline the next time the baseline grows:

```
budget = min(claudeGuardBaselineTokens * claudeGuardBirthHeadroomFactor,
             HardContextCap - OutputReserve)
       = min(62000 * 2, 200000 - 32000)
       = min(124000, 168000) = 124000
```

- Grows with the baseline (baseline × 2 birth-headroom → a worker is born with the
  whole baseline again in headroom; never a birth wall again).
- Shrinks with the model window (clamped to `HardContextCap - OutputReserve`, the
  effective ceiling from `internal/ctxplan.GenericTurnEnvelope()` — the long-context
  doctrine seam, `docs/long-context-defaults.md`: the advertised window is a hard
  CAP, never a raw target). Always a runaway backstop.

Wiring:
- Go: `cmd/dispatchworker/guard.go` — `deriveClaudeGuardContextBudget`,
  `claudeGuardContextBudgetTokens`, constants `claudeGuardBaselineTokens = 62000`,
  `claudeGuardBirthHeadroomFactor = 2`.
- Python mirror: `tools/dispatch_worker.py` `claude_guard_context_budget_tokens`
  hand-mirrors the four constants (it cannot import the Go `ctxplan` package) and
  must stay numerically identical.
- Golden parity: `TestClaudeGuardContextBudgetDerivation` (Go) and
  `test_claude_guard_context_budget_derivation_matches_go` (Python) both pin `124000`
  and assert `budget > baseline` (birth-safe), `budget <= ceiling` (runaway backstop),
  monotone-in-baseline, and the exact CLI flag surface. The adequacy floor in
  `TestGuardWrapClaudeFrontsWithFakGuardAnthropic` is `>= 62400`.

## The measured-baseline seam (shipped — #3522)

The follow-up replaces the *sole* frozen baseline with a **measurement** that is
**floored** at the old constant and can only ever **raise** the baseline — never lower
it. `claudeGuardBaselineTokens = 62000` is retained as a FLOOR, not the answer:

```
measured  = Σ approxTokensFromBytes(bytes(constituent))   # (bytes+3)/4, the ctxplan ruler
baseline  = max(measured, claudeGuardBaselineTokens)       # floor: never below the shipped value
budget    = min(baseline * 2, HardContextCap - OutputReserve)
```

- **Constituents sized at launch:** the orientation files a self-claiming lane worker
  actually loads from the workspace root (`AGENTS.md`, `llms.txt`, `CLAUDE.md`), a
  workspace-root `MEMORY.md` when a repo keeps one, and the `route` blob when the
  launcher names its path via `DISPATCH_STARTUP_BUNDLE`. An absent/unreadable file
  contributes nothing (the degenerate guard). Caveat: the *real* injected fleet memory
  lives in the per-project claude memory dir (`…/projects/<ws>/memory/MEMORY.md`), off
  the workspace root and not portably derivable at launch, so in the common fleet
  layout `MEMORY.md` is absent and floor-covered — it is measured only for a repo that
  actually keeps a root `MEMORY.md`.
- **Raising-only, floored:** a degenerate or partial measurement (empty inputs,
  unreadable files, an empty workspace) sums below the floor, so `max(…, 62000)` keeps
  today's exact 124000 — the #2972 born-over-budget invariant holds by construction.
  But if the *orientation/memory* constituents organically grow PAST the floor, the
  baseline (and budget) track them automatically instead of silently outgrowing a
  frozen constant. That is the specific hole this closes.
- **Observable:** a claude launch record now carries `guard_baseline_tokens` and
  `guard_context_budget_tokens` (payload + `render`), so drift is a visible number in
  `.dispatch-runs/` rather than an argv int nobody reads.
- **Wiring:** Go `cmd/dispatchworker/guard.go` — `approxTokensFromBytes`,
  `measureLaunchBaselineTokens`, `resolveClaudeGuardBaseline`,
  `gatherLaunchConstituentBytes`, `measuredClaudeGuardBaseline`,
  `claudeGuardContextBudgetTokens(workspace, env)`. Python mirror
  `tools/dispatch_worker.py` (same names, snake_case). Parity golden:
  `TestMeasureLaunchBaselineFloorsAndTracks` (Go) and
  `test_measure_launch_baseline_floors_and_tracks` (Python) — same fixtures, same
  integers; the derivation goldens still pin the hermetic 124000 default.

### Why this is NOT the rejected static guard

An earlier draft of this note rejected "a CI test that sums the on-disk repo files and
**asserts the sum stays under** `claudeGuardBaselineTokens`" — because those files are
only ~15–20K of the ~62K baseline, so a guard that reads as "the baseline is guarded"
while witnessing a fraction manufactures false confidence. The shipped seam is the
inverse of that failure mode and does **not** reintroduce it:

- It **never asserts** the measured sum bounds the real prompt. It floors AT 62000 and
  only raises — the floor remains fully authoritative for every dynamic term the sum
  cannot see, so no false "fully guarded" claim is made.
- Its job is not "prove the baseline is safe" (impossible from repo files alone) but
  "stop the *stable* constituents from silently outgrowing the budget while the floor
  covers the rest." A fraction measured additively-upward is strictly safer than the
  frozen constant; a fraction measured as an *upper bound* was the trap.

## The residual — still open

The seam sizes only the constituents *readable at launch*. The dominant DYNAMIC terms —
the per-issue body, Claude Code's own system prompt, and the `route` blob unless a
launcher plumbs `DISPATCH_STARTUP_BUNDLE` — are still covered only by the floor. So the
full-prompt witness is not yet in place: organic growth of a *dynamic* term past 124000
would still re-introduce the turn-1 trap and still surface as a generic `CHILD_CRASH`.

## Next checkable steps (in preference order)

1. **Runtime attribution (the real "measure the FULL launch prompt" fix).** The guard
   ALREADY measures the real launch prompt — `DebitUsage` sees turn 1's full resident
   window every run. What's missing is *attribution*: make a turn-1 drain
   (`ContextTokensLeft` exhausted before any productive turn) emit a DISTINCT,
   attributed reason (`real_baseline > budget`, with both numbers) instead of a
   generic `BUDGET_CONTEXT_EXHAUSTED` → `CHILD_CRASH`. Then a re-introduced trap is
   loud and self-explaining in the guard decision journal
   (`.dispatch-runs/guard-audit/`) instead of a silent ship-rate collapse. Witness:
   a hermetic session whose turn-1 window exceeds the budget produces the new reason.
2. **Plumb `DISPATCH_STARTUP_BUNDLE` from the dispatch tick.** The seam already folds
   the `route` blob when its path is named in the environment; wiring the dispatch tick
   (which writes the sidecar) to export that path moves the ~40K dominant static-ish
   term from floor-covered to measured. Witness: a launch whose bundle exceeds the
   floor raises `guard_baseline_tokens` above 62000.
3. **Offline baseline-floor refresh.** Sample real guarded workers' turn-1 resident
   windows from the guard decision journal (or `tools/ctxwin.py` over worker sessions)
   and refresh the `claudeGuardBaselineTokens` FLOOR from the p95 on a cadence. Witness:
   a regeneration script + a note like `CTXWIN-CONTEXT-WINDOW-BASELINE-*`.

Status: the frozen baseline is now a **floored, raising-only measurement** (#3522,
shipped and green) — the organic-orientation-growth hole is closed. Three checkable
steps remain (above); the full-dynamic-prompt runtime-attribution witness (step 1) is
the highest-preference of them.

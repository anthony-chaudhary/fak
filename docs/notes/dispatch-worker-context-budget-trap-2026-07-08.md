---
title: "Dispatch-worker context-budget trap: the derived-budget fix and its measured-baseline residual (2026-07-08)"
description: "The #2972 crash-loop where the claude dispatch worker's guard budget (48000) sat below the workers' ~62K irreducible baseline prompt, so every worker was born over-budget and drained BUDGET_CONTEXT_EXHAUSTED on turn 1. Records the shipped derived-budget fix (min(baseline*2, HardContextCap-OutputReserve) = 124000), why a flat constant goes stale, and the open residual: the 62000 baseline is still a frozen hand-measured constant that nothing witnesses against the real launch prompt."
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
fronted by `fak guard` (`cmd/dispatchworker/guard.go`). Guard is handed
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

## The shipped fix (`8a0fcffbb`)

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

## The residual — NOT closed by the fix

`claudeGuardBaselineTokens = 62000` is still a **frozen, hand-measured constant**.
Nothing measures the *real* launch prompt and reconciles it against this assumption.
Organic growth of any baseline contributor (a longer AGENTS.md, a fatter `route`
blob, more injected memory, a bigger Claude Code system prompt) past the derived
`124000` budget silently re-introduces the exact turn-1 birth trap — and it would
again surface only as a generic `CHILD_CRASH`, not as "born over budget."

The golden tests do **not** catch this: they witness the *arithmetic* (budget vs the
*assumed* baseline), not the *assumed baseline vs the real prompt*. The one input
that matters most is the one input nothing checks.

## Why a naive static guard is NOT the fix (rejected approach)

Tempting: a CI test that sums the on-disk repo-file contributors (CLAUDE.md +
AGENTS.md + llms.txt) as tokens (≈4 chars/tok) and asserts the sum stays under
`claudeGuardBaselineTokens`. **Do not ship this as a closure.** Those static files
are only ~15–20K of the ~62K baseline; the dominant contributors — the ~40K dynamic
`route` blob, the per-issue body, and Claude Code's own system prompt — are *not*
repo files and are invisible to a static sum. A guard that witnesses a fraction of
the baseline while reading as "the baseline is guarded" is worse than no guard: it
manufactures false confidence over exactly the terms that actually move.

## Next checkable steps (in preference order)

1. **Runtime attribution (the real "measure the launch prompt" fix).** The guard
   ALREADY measures the real launch prompt — `DebitUsage` sees turn 1's full resident
   window every run. What's missing is *attribution*: make a turn-1 drain
   (`ContextTokensLeft` exhausted before any productive turn) emit a DISTINCT,
   attributed reason (`real_baseline > budget`, with both numbers) instead of a
   generic `BUDGET_CONTEXT_EXHAUSTED` → `CHILD_CRASH`. Then a re-introduced trap is
   loud and self-explaining in the guard decision journal
   (`.dispatch-runs/guard-audit/`) instead of a silent ship-rate collapse. Witness:
   a hermetic session whose turn-1 window exceeds the budget produces the new reason.
2. **Offline baseline refresh.** Sample real guarded workers' turn-1 resident windows
   from the guard decision journal (or `tools/ctxwin.py` over worker sessions) and
   refresh `claudeGuardBaselineTokens` from the p95 on a cadence — converting the
   hand-measured constant into a measured one. Witness: a regeneration script + a note
   like `CTXWIN-CONTEXT-WINDOW-BASELINE-*` that sources the number.

Status: **not yet** for the residual — the derived-budget fix is shipped and green;
the measured-baseline witness above is the remaining, checkable work.

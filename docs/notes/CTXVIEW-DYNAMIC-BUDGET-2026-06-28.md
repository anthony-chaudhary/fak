---
title: "Dynamic ctxview budget: 8000 is below the agentic floor — scale the window to the model (2026-06-28)"
description: "Why the flat --ctx-view-budget 8000 default is the wrong absolute number for agentic sessions, the [faithfulness floor .. economic ceiling] band the budget actually lives in, and a workload-adaptive policy that scales the O(1) resident target to the model's context window (with a prefix floor and an economics-aware control-loop follow-on). The next-steps ladder for the ctxview epic."
slug: ctxview-dynamic-budget
date: 2026-06-28
---

# Dynamic ctxview budget (2026-06-28)

We just flipped the ctxplan O(1) context planner ON by default at a **flat
`--ctx-view-budget 8000`** (commit `89386943`). That default is the right *shape* (a
bounded resident view) but the wrong *absolute number* for agentic workloads. This note
records why, and the smart way to make the window target dynamic.

## Why a flat 8000 is wrong

The budget is a hard cap on the **total resident message-history tokens** of the planned
view (`internal/ctxplan/plan.go:202` `Optimize`). The **pins** — system prompt, the active
goal, and the first/last user turns (`internal/agent/ctxplan_seam.go:171` `messagesToStore`)
— are charged FIRST and are non-negotiable; if they exceed the budget the plan sets
`OverBudget=true` and keeps *only* the pins, eliding all middle history
(`plan.go:249`). It never errors — it fails open to the minimum viable view.

Two consequences make 8000 a poor agentic default:

- **It is far below the economic ceiling.** The [O(1) context-window
  economics](../explainers/o1-context-window-economics.md) result is a near-identity: a
  reconstructed window beats a warm prefix cache **exactly while it is smaller than the
  cache's effective discount** — ≈ **11.8% of the billed prompt**, which on heavy real
  Claude Code sessions (mean billed prompt ~284K tok/turn) is **~34K tokens**. So you can
  keep ~34K resident and *still* be cheaper than the warm cache. 8000 spends a quarter of
  that headroom and sheds the rest of the agent's working set for no economic reason.
- **It leaves almost no recency reserve.** Once a session's history passes 8000, the
  planner keeps the pins + the few most-relevant spans and stubs everything else. The
  agent's *recent* tool results and reasoning — the working set it actually reasons over —
  get stubbed on the wire (recall is exact via page-fault, but a stub the model can't see
  this turn is a faithfulness risk and a page-fault round-trip). Agentic loops need a
  healthy recent window, not the bare prefix.

The single flat number honors neither bound. **The budget lives in a band:**

```
floor (faithfulness)            target                       ceiling (economics)
  pins + recency reserve   <   what we keep resident   <   ~min(0.12·billed_prompt, ctx_window)
  (system+goal+recent)         (scales with the model)      (stay cheaper than warm cache)
```

## The smart policy: scale to the model, floor on the prefix

Replace the flat number with a per-turn **derived** budget:

```
effectiveBudget(turn, model) = clamp(
    target = budgetFraction × contextWindow(model),   // scales with model capability
    floor  = max(configuredFloor, pinnedTokens + recencyReserve),  // never shed the working set
    ceil   = contextWindow(model) − genReserve,        // never crowd out generation
)
```

- **`contextWindow(model)`** — for the in-kernel path, `Config.ContextSizeConfig().MaxContext`
  (the gguf `n_ctx`); for the **Anthropic/OpenAI passthrough**, a small model-id→window map
  (`claude-* → 200000`/`1000000`, `gpt-* → 128000`, …), defaulting conservatively. This map
  is reusable infrastructure (KV sizing, vcache).
- **`budgetFraction`** — the one real tunable. Anchored on the **economic crossover (~0.12)**
  it gives ~24K for a 200K Claude window: **3× the current working set, still cost-positive
  on the heavy sessions where it matters** (beats the warm cache once the billed prompt
  passes ~200K). A faithfulness-leaning operator can raise it toward 0.25–0.5; the ceiling
  keeps it bounded.
- **`configuredFloor`** — keep `--ctx-view-budget N` as an explicit override (a fixed N still
  works; `auto` selects the dynamic policy; `0` stays off). So this is strictly better than
  today: never smaller than 8000, larger where the model can afford it.

Worked numbers (fraction 0.12, floor 8000):

| model window | flat today | dynamic target | effect |
|---|---:|---:|---|
| 32K local | 8000 | ~8000 (floored) | unchanged on small models |
| Claude 200K | 8000 | ~24K | 3× working set, cost-positive on heavy sessions |
| Claude 1M | 8000 | ~120K | tracks the bigger window |

The crucial safety property is unchanged: pins always stay, the planner only ever shortens,
and recall is exact — so raising the target only ever *adds* faithful context.

## Next-steps ladder (the ctxview epic)

1. **Dynamic budget v1 — context-window-scaled, prefix-floored.** The policy above. Needs:
   the model-id→window map + `MaxContext` carried onto the gateway `Server`; the derive at
   `maybePlanMessages` (`internal/gateway/gateway.go:1055`); `--ctx-view-budget auto` (default)
   vs an explicit N. Witness: re-run `cmd/ctxplanbench` at the derived budgets; the existing
   `TestCtxViewHTTP` passthrough byte-identity must still hold.
2. **`holds_new_context` as a live SLO.** Port the explainer's `holds_new_context` flag to Go
   from the existing `Plan.OverBudget` + a per-turn "genuinely-new tokens" count, and surface
   it on `/metrics` + the `--debug-stats` line. This makes budget *adequacy* observable on
   real traffic — the control signal for everything below.
3. **Economics-aware control loop.** Read `cacheobs.Default.Snapshot()` (reuse ratio, billed
   prompt) + the per-session `OverBudget`/fault rate at plan time and auto-tune the fraction:
   raise toward the crossover when faults are high, settle when reuse saturates. The budget
   self-tunes to the workload instead of being guessed.
4. **Forecast-derived (relevance-threshold) budget.** Use the benefit-threshold knapsack
   (`internal/ctxplan/plan.go:313` already skips `Benefit<=0`) so the view = pins + every
   candidate above a relevance threshold, bounded by the ceiling — the budget becomes an
   *output* of relevance, not a number to guess.
5. **Faithfulness eval (the load-bearing quality witness).** The whole O(1) thesis assumes a
   bounded view lets the agent produce the same turn. Add a task-success eval (a SWE-bench /
   terminal-bench slice) comparing planned-view vs full-history agent success — the quality
   axis the cost/recall witnesses do not cover, and the gate before pushing the budget
   *smaller*.
6. **(Stretch) In-kernel residency at the dynamic budget.** Wire the derived budget into the
   `kvmmu` residency bridge (`FAK_INKERNEL_KVMMU`) so the compute-side O(1) residency tracks
   the same target — the regime-E win.

## Honest fences

- The `budgetFraction` default is a judgment call on the cost↔fidelity tradeoff; v1 should
  ship a defensible anchor (the ~0.12 crossover) and let the control loop (#3) earn a smaller
  or larger one from live signal, not a guess.
- `contextWindow` for the passthrough is a model-id heuristic until a provider advertises it;
  an unknown model falls back to the configured floor (no worse than today).
- None of this is a quality claim — it improves the *budget*, not the planner's forecast or
  faithfulness; that is what eval #5 exists to establish.

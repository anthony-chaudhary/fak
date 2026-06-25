---
title: "The compounding benefits of a saved call: why one avoided tool call pays back four times, then again on the horizon"
description: "fak's headline accounting prices one quantity — turns saved — four ways (tokens, dollars, latency). That undercounts. A single avoided or cheapened tool call discharges across four ORTHOGONAL budgets at once (local CPU, GPU/prefill, context window, wall-clock), and then a fifth effect compounds on top: the budget it returns extends how many EFFECTIVE calls the session can still make. This note formalizes both — the multi-axis discharge and the horizon multiplier — grounds every term in a real seam, and fences measured from modeled."
slug: compounding-benefits-of-a-saved-call
keywords:
  - avoided tool call
  - effective tool call
  - agent horizon
  - context budget
  - compounding savings
  - turn tax
  - work elimination
  - long-horizon agent
date: 2026-06-25
---

# The compounding benefits of a saved call

*Who this is for: anyone reasoning about what fak is actually worth on a long agent run, and why the worth is larger than the headline "turns saved" number — without inventing a number to say so. Prerequisite: a rough grasp of the [loop ladder](engineering-is-building-loops.md) and the [O(1) context economics](o1-context-window-economics.md). You will come away able to state the two compounding structures precisely, name the seam each rides on, and say exactly where the claim is measured and where it is modeled.*

**Short version.** fak's shipped accounting (`internal/turnbench`, `Net`) takes one integer — `turns_saved` — and prices it four ways: tokens, dollars, latency. That is honest but it *under-models the benefit in two specific ways*, and both are the question the user posed.

1. **One saved call discharges across orthogonal budgets, not one budget in four currencies.** An avoided tool call does not "cost −X tokens." It simultaneously returns local CPU (the in-process adjudication or spawn it never ran), GPU/prefill FLOPs (the forward pass that never happened), a context-window slot (the result that never entered the window), and wall-clock (the round-trip the loop never blocked on). These are *separate accounts with separate ceilings*. The flat `Net` collapses three of them into the token price; the true discharge is a **vector**, and the binding constraint on a given run is whichever account is scarcest — usually not dollars.

2. **The returned budget buys more horizon, and that is multiplicative.** Context and dollars are finite *per session*. The number of useful calls a session can make before it must compact, reset, or stop is `budget / effective_cost_per_call`. fak pushes on **both** ends of that ratio — it shrinks the denominator (each call is cheaper) *and* refills the numerator (containment and reuse hand budget back to the pool). Numerator-up and denominator-down compound: the horizon gain is the *product*, not the sum, of the per-call savings and the budget-recovery. This is the "longer-horizon work, and faster work at the same horizon" the question names, made precise.

The rest of this note builds both claims from the seams and fences them hard. The one-line thesis: **fak's value is not a discount on a turn; it is a discount on a turn paid out of four budgets, reinvested into horizon.**

---

## 1. The flat model fak ships, and exactly what it leaves on the table

The turn-tax harness is the honest core. For a replayed trace it classifies every call the kernel saw (`ClassBreakdown` in `internal/turnbench/turnbench.go`): a grammar repair, a vDSO local serve (pure / content-cache / static), a quarantine, a deny, or a plain pass. The turns the baseline pays and fak does not is `turns_saved = grammar + vdso` (`ClassBreakdown.turnsSaved`), split honestly into **forced** turns the baseline *demonstrably* re-issues (a duplicate read, an aliased call that errors and is re-prompted) and **elision** turns for optional calls a stronger model could have skipped (`forcedTurns` / `elisionTurns`). That split is the project's anti-overclaim discipline and this note keeps it.

Then `netFor` prices it (`internal/turnbench/turnbench.go`):

```go
func netFor(turns int, cm CostModel) Net {
    return Net{
        TurnsSaved:     turns,
        TokensSaved:    turns * cm.tokensPerTurn(),
        DollarsSaved:   float64(turns) * cm.dollarsPerTurn(),
        LatencySavedMs: float64(turns) * cm.ModelTurnLatencyMs,
    }
}
```

Read that closely. It is *one* count, `turns`, multiplied by *one* per-turn constant in each unit. The structure it encodes is "a saved turn is worth `tokensPerTurn` tokens **and** `dollarsPerTurn` dollars **and** `ModelTurnLatencyMs` ms," and those three are the *same saving* expressed in three currencies — you would never add them, because dollars *are* tokens at a price and latency *is* turns at a round-trip. The model is a scalar saving with three exchange rates.

That is correct as far as it goes, and it is deliberately conservative. What it cannot express is the two things this note is about:

- It has **no axis for local compute** at all. The CPU the kernel did or did not burn to make the decision is invisible to `Net` — yet it is the axis the boundary-tax sentinel (the ~2,849× in-process-vs-spawned number) is entirely about. A saved *spawn* is a real local-CPU saving that `Net` never books.
- It has **no notion of a finite per-session budget**, so it cannot represent that the saving *returns* something that lets the session do more. Every saved turn is priced as a one-time rebate, never as recovered headroom. The horizon effect is structurally absent.

Both omissions are *safe* — `Net` under-claims, which is the right direction. But "the benefit is larger than `Net` says, and here is the structure of the part it omits" is a true and useful statement, and it is what follows.

---

## 2. Claim 1 — a saved call is a vector, not a scalar

A tool call, when it actually runs, draws down four distinct resources. Avoid it, or serve it locally, or contain its result, and you credit each of the four — but by *different amounts from different seams*, and against *different ceilings*. Here is the discharge, one row per account.

| account | what a *run* call draws | what fak's lever returns | the seam | measured or modeled |
|---|---|---|---|---|
| **local CPU** | the adjudication cost + (in a hooked harness) a process spawn per gate | an in-process decide instead of a spawn; a vDSO hit instead of a dispatch path | `Adjudicator.Adjudicate`, `vdso.Lookup` at `kernel.go` Submit; baseline `bench.MeasureSpawnedBaseline` | **measured** (M3: ~362 ns decide; ~2,849× vs spawned hook) |
| **GPU / prefill** | a forward pass over the call's prompt + the new result's tokens, attention O(L²) in context length | the forward pass that never runs (elided/served call) + the prefill never paid for a result that never enters context | `Session.Prefill` / `kvcache.go`; result-elision via `vdso`; ultra-long-context floor `geometry.go` | **measured** for the reuse arms (B/C 2.4–2.7× vs tuned, at T=8/16 on SmolLM2; the 50×5 headline run is still pending); **modeled** for the ultra-long floor (geometry, no model) |
| **context window** | one result's tokens permanently occupy a window slot for the rest of the session | a quarantined/paged-out result costs a sub-2KB pointer, not its full body; an O(1) view holds the window flat | `ctxmmu.Admit` (page-out stub), `ctxplan.Optimize` (bounded view) | **measured** as a rate (pollution rate, resident-token compression on real transcripts); the window→horizon link is **modeled** |
| **wall-clock** | the loop blocks on a model round-trip (~seconds) before it can act on the result | a locally-served or elided call returns without the round-trip | `ModelTurnLatencyMs` in the cost model; vDSO serve path | **modeled** (a knobbed round-trip constant, never a wall-clock measurement) |

The load-bearing word is **orthogonal**. These are not one saving in four units. They are four *different* budgets, each with its own ceiling, and a single saved call credits all four at once:

- **The CPU account and the GPU account are paid to different silicon.** Saving a spawn helps a CPU-bound orchestrator host; saving a forward pass helps a GPU-bound serving box. A run that is GPU-starved and CPU-idle gets nothing from the first column and everything from the second. `Net`, pricing only through tokens, cannot tell you which.
- **The context account has the hardest ceiling and the least elastic price.** You can buy more dollars. You cannot buy more context window mid-session — it is fixed by the model. A result that bloats the window is not "expensive," it is *budget you cannot get back without a compaction*, which is why the context axis is the one that gates horizon (§3).
- **The wall-clock account is the one a human or a downstream loop actually waits on.** Two runs with identical token bills but different round-trip counts feel completely different to the operator and finish at different times.

The honest consequence: **the binding benefit of fak on a given run is whichever of these four accounts is scarcest, and it is rarely dollars.** A long local agent on a laptop is context- and wall-clock-bound; a fleet on a rented GPU is prefill-bound; a hooked CI harness is CPU-bound on the gate itself. The flat `Net` reports a dollar figure for all three, which is exactly the wrong axis for each. Reporting the *vector* — even with three of its four entries modeled — tells a reader which lever matters for *their* bottleneck.

### Why this is not just "four benefits" — the discharge is from one event

The subtle part, and the reason "compounding" is the right word and not "list": all four credits come from **the same single adjudication decision**. The kernel decides once — at `Submit`, before any engine or network — whether these bytes may enter the model's attention. That one verdict is what simultaneously (a) skips the spawn, (b) skips the forward pass, (c) keeps the slot out of the window, and (d) skips the round-trip. This is the [inference-front-end lens](engineering-is-building-loops.md#loops-all-the-way-down) restated as economics: *one decision, enforced once, discharges across every downstream budget the decision would have committed.* A serving engine that sees only the wire bytes has already lost three of the four accounts — it cannot decline to spawn, cannot decline to prefill before the prompt arrives, cannot keep a result out of a window it does not own. The kernel can, because it is upstream of all four.

---

## 3. Claim 2 — the horizon multiplier (the part the question is really about)

Now the compounding. Define, for one session under a binding budget `B` (pick the scarce account from §2 — for a long agent it is almost always the context window, secondarily dollars):

```
effective_horizon  =  B  /  effective_cost_per_call
```

`effective_horizon` is the number of *progress-making* calls the session can still make before it hits the wall and must compact, reset, or stop. "Effective" excludes the calls that buy nothing — a re-issued duplicate read, an aliased retry, a turn spent re-reading context that fell out of the window. Those are exactly the calls fak's turn-tax levers delete. So fak moves *both* terms of the ratio, and they multiply.

**Denominator — each call costs less.** Every lever in §2 lowers `effective_cost_per_call`: a vDSO hit costs a pointer instead of a forward pass; a grammar repair costs a re-store instead of a re-prompt round-trip; an O(1) view costs a bounded prefill instead of an O(L²) re-prefill. Call the factor `d < 1` (cost multiplier per call after fak).

**Numerator — budget is returned to the pool, not just spent more slowly.** This is the move `Net` cannot see. Containment and reuse *give budget back*:

- A **quarantined or paged-out result** does not consume its window slot — it consumes a sub-2KB pointer (`ctxmmu.Admit`). The difference is window budget *returned to `B`*, available for a future real call. Across a long session this is the dominant recovery, because [92% of context is tool results, most of them stale](o1-context-window-economics.md).
- A **bit-exact KV eviction** (`kvcache.go` `Evict`, `max|Δ|=0`) removes a span from the *middle* of a kept run and re-RoPEs the survivors, so the freed positions are genuinely reusable, not just logically forgotten. A shared-slot engine cannot do this — it can only append — so for it the numerator only ever shrinks.
- A **session reset with sound carryover** (`session.Recontinue`, `internal/sessionreset`) is the explicit "refill `B` to full, keep the durable facts" operation — the human-like move of starting fresh while carrying what matters.

Call the budget-recovery factor `r > 1` (the effective `B` is `r·B` over the session because spent budget keeps coming back).

Then:

```
effective_horizon_fak     r·B / (d · c)        r
------------------------ = ------------- = --------- = r / d        ( > r, and  > 1/d )
effective_horizon_naive     B / c              d
```

The horizon gain is `r/d` — the **product** of budget-recovery and per-call-cheapening, not their sum. A modest `d = 0.7` (each call 30% cheaper) and a modest `r = 1.5` (budget effectively recovered half-again over the session) is not a 1.8× horizon (`1 + 0.5 + 0.3`); it is `1.5 / 0.7 ≈ 2.1×`. The two levers reinforce because cheaper calls spend the recovered budget more slowly, and recovered budget gives the cheaper calls more runway — each makes the other worth more.

This is the precise statement of the user's two intuitions:

- *"More real / effective tool calls allows for longer-horizon work"* — that is the numerator: `r` raises how many effective calls fit, by recovering the budget the wasted calls and bloated results would have burned.
- *"...and faster work at the same horizon"* — that is the denominator: `d` lowers the cost of each call, so a *fixed* horizon completes in less wall-clock and less spend.

And the compounding is why you cannot get this by optimizing one lever in isolation. A pure caching layer moves `d` and leaves `r = 1`. A pure compaction tool moves `r` and leaves `d = 1`. fak's bet — the [loops-all-the-way-down](engineering-is-building-loops.md) assembly — is that one kernel holding the adjudication decision moves both, and the value is their product.

### The honest fence on Claim 2

`r/d` is a **model**, and it inherits every caveat the O(1) economics doc carries, plus one of its own:

- **`d` is partly measured, partly modeled.** The vDSO/grammar turn deletions are real kernel events (`turns_saved` is measured on a replayed trace). The per-call *cost ratio* in tokens is measured; in dollars and latency it rides the knobbed `CostModel`. In local CPU it is measured (the boundary tax). So `d` is a blend, and a reader must price it on *their* scarce axis.
- **`r` is the softest term and must not be quoted as a number.** It depends on the workload's staleness profile — how much of the context is reclaimable without losing a fact the agent needs. The repo *measures the inputs* to `r` (pollution rate, resident-token compression, the demand-page fault rate that flags when reclamation went too far) but does **not** ship a measured `r` for a real task, because doing so soundly requires a task-success eval proving the reclaimed budget did not cost an answer. Until that eval exists, `r` is a structural argument, not a figure. **Do not publish a horizon multiplier as a headline number.** Publish the structure and the measured inputs.
- **The whole thing assumes faithfulness.** A horizon you bought by evicting a span the agent later needed is not a horizon gain — it is a demand-page fault (best case, you pay it back) or a wrong answer (worst case). This is the same load-bearing faithfulness assumption `internal/ctxplan` exists to establish, and the same one the O(1) cost result rests on. A horizon win on a window that breaks the agent's reasoning is not a win.

---

## 4. Where the compounding does and does not hold

The model is sharpest where the scarce budget is real and the levers are sound. It is weakest — and must be fenced — where either fails.

**It holds strongly when:**
- The session is long enough that the O(T²) re-prefill of the naive arm dominates, so the denominator gap widens with T (the [session value stack](../benchmarks/SESSION-VALUE-STACK-RESULTS.md) shows exactly this at its measured points — T=8/16 on SmolLM2: A/C grows with turns, 11.2× → 14.5×, while B/C holds steady at 2.4–2.7×; the realistic-model 50×5 headline is still a pending live run).
- Most of the context is stale tool results, so `r` has real budget to recover. This is the common shape of long tool-use loops, not an edge case.
- fak owns the engine, so eviction is a cheap bit-exact cache op rather than a re-prefill — then the denominator win and the numerator win are *the same operation* (evict a span = cheaper next call AND recovered budget), which is the tightest form of the compounding.

**It weakens or inverts when:**
- The task genuinely needs most of the history every turn (`r → 1`: nothing reclaimable) — then you are above the O(1) crossover and the numerator lever is dead. fak still moves `d`, but the multiplier collapses toward `1/d`.
- The scarce budget is dollars on a hosted API whose prompt cache already harvests most of the prefix saving — measure with `tools/session_audit.py` before claiming a dollar win; the harness cache may already own it.
- The levers fire on the wrong axis for the bottleneck: a vDSO hit saves a forward pass, which is worth nothing on a run that is CPU-bound on the orchestrator and GPU-idle. The vector in §2 is the guard against quoting the wrong account.

**It is structurally unavailable to a shared-slot engine.** The numerator lever (`r > 1`) needs per-agent KV ownership — the ability to evict a span from the middle of *one* agent's run and re-RoPE the survivors bit-exact. A PagedAttention/RadixAttention pool shares cells across requests and cannot do this without forking the pool. So the compounding is not a tuning fak happens to have; it is downstream of the one structural property (per-agent addressable KV) that the [inference-front-end lens](engineering-is-building-loops.md) names as the un-commoditized moat.

---

## 5. What this changes about how to report fak's value

Three concrete shifts, each honest:

1. **Report the saving as a vector, not a dollar figure.** `Net` is a scalar with three exchange rates; the real saving is four accounts with four ceilings. The smallest honest upgrade is to surface the *local-CPU* account `Net` omits entirely (the boundary tax is already measured) and to label which account is binding for the run's profile. A reader on a laptop cares about context and wall-clock; a reader on a GPU fleet cares about prefill. One dollar number serves neither.

2. **Frame the long-run benefit as horizon, not total spend.** "fak saved $X over the session" is the weakest true thing you can say, because the harness cache may already own most of it. "fak let the session make `N` more effective calls before it had to reset" is the strong true thing, and it is the thing a long-horizon agent author is actually budgeting against. The seam that would measure `N` is the [agentic lifecycle KPIs](../notes/AGENTIC-LOOP-KPIS-2026-06-25.md) — specifically the session-layer KPIs (resume warmth, promotion rate, core-image size) that today are guarded, not standing.

3. **Keep the multiplier as structure, ship the inputs as numbers.** The product `r/d` is the right *mental model* and the wrong *headline*. Publish `d`'s measured parts (boundary tax, turns_saved, B/C reuse) and `r`'s measured inputs (pollution rate, resident-token compression, demand-page fault rate) — each a real event — and let the reader compose the multiplier for their workload. This is the same discipline that corrected the webbench number from "measured 9.7×" to an honest modeled floor: the structure is real, the single number would be invented.

---

## 6. The one-paragraph version

A tool call draws on four separate budgets — local CPU, GPU prefill, context window, wall-clock — and fak's single upstream adjudication decision can decline to spend any of them, so one avoided or cheapened call pays back from all four accounts at once, against whichever ceiling is actually binding (rarely dollars). That saving then compounds, because the budget it returns to a finite session — chiefly context window, via containment and bit-exact eviction — extends how many *effective* calls the session can still make: `effective_horizon = budget / effective_cost_per_call`, and fak pushes the denominator down (cheaper calls) while pushing the numerator up (recovered budget), so the horizon gain is their product `r/d`, not their sum. The per-call discharge is largely measured; the horizon multiplier is a model whose *inputs* are measured but whose *single number* is deliberately not published, because quoting it soundly needs a task-success eval that the budget you reclaimed did not cost an answer. The shipped `Net` accounting is a conservative scalar that under-models both effects — which is the safe direction, and the gap this note names.

## Reproduce / read next

```sh
# the flat Net this note extends — the measured turn-tax and its four-way price
go test ./internal/turnbench/...
fak turntax --suite turntax-airline     # the safety floor: 1 injection->0, 1 destructive->0

# the denominator's measured inputs
fak benchmarks run kernel-latency       # the local-CPU account (boundary tax)
python tools/ctxcost.py crossover        # the context account's O(1) economics

# the numerator's measured inputs
python tools/session_audit.py            # is the harness cache already harvesting the dollar saving?
```

- [Engineering is building loops](engineering-is-building-loops.md) — the loop ladder and the "one decision, every ring" frame this note prices.
- [The O(1) context window economics](o1-context-window-economics.md) — the measured crossover that grounds the context account and the faithfulness fence.
- [Session value stack results](../benchmarks/SESSION-VALUE-STACK-RESULTS.md) — the B/C reuse number behind `d` (measured at T=8/16; 50×5 headline pending).
- [Internal benchmark KPIs](../notes/AGENTIC-LOOP-KPIS-2026-06-25.md) — the lifecycle seams that would turn the horizon model into a standing measurement.

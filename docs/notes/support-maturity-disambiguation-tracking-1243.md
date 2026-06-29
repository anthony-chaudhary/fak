---
title: "EPIC #1243 — Support-maturity disambiguation: a witnessed level ladder + a dev-regime router"
description: "Splits the one fuzzy word 'supported' into a closed, witnessed maturity ladder (none → loads → runs → correct → optimized → SOTA-parity → beyond-SOTA) and a router that turns each cell's rung into the right dev-regime, time-horizon, and tooling — so a 10-step 'make it work' is never confused with a 10,000-step 'make it fast'. Extends internal/covmatrix; ships no code in this note."
---

# EPIC #1243 — Support-maturity disambiguation

> **Status:** OPEN · planning + research only · roll-up tracking note for epic
> [#1243](https://github.com/anthony-chaudhary/fak/issues/1243). **This note ships no code;**
> its children (#1244–#1257) are the build-out, each extending a named package with its own witness.
> **Lanes:** spans `covmatrix`, `compute`, `supportmaturity` (new), `scorecard`, `rsiloop`,
> `shipgate`, `docs`.
> **Sits above (does not duplicate):** the Parity tracks [#307](https://github.com/anthony-chaudhary/fak/issues/307)
> (Model-Support) · [#305](https://github.com/anthony-chaudhary/fak/issues/305) (GPU/Backend) ·
> [#303](https://github.com/anthony-chaudhary/fak/issues/303) (Testing/Quality) ·
> [#301](https://github.com/anthony-chaudhary/fak/issues/301) (Foundation) — they do the per-cell
> *parity work*; this is the *instrument* that measures where each cell is and routes the work.
> **Dual of:** the [concept-disambiguation scorecard](../concept-disambiguation-scorecard/README.md)
> (disambiguates *names*); this disambiguates *support levels*.
> **Cross-link (a dual, not a merge):** [#1178](https://github.com/anthony-chaudhary/fak/issues/1178)
> random time horizons tracks *session-dormancy* horizon (time **away**); this epic tracks the
> *work/effort* horizon (steps to **advance a rung**). Two different clocks, kept apart.

## 0. Child map

| Slug | Issue | Rung | Title |
|---|---|---|---|
| **C1** | [#1244](https://github.com/anthony-chaudhary/fak/issues/1244) | E0 · keystone | define the closed M0–M7 ladder enum unifying `covmatrix.Support` + `CorrectnessClass` + preflight verdicts |
| **C2** | [#1245](https://github.com/anthony-chaudhary/fak/issues/1245) | E1 | per-rung witness binding + shipgate-gated promotion + drop-on-regression |
| **C3** | [#1246](https://github.com/anthony-chaudhary/fak/issues/1246) | E2 | grade `covmatrix` into a `support_maturity_debt` scorecard (`pkg/scorecard.Fold` + control-pane row) |
| **C4** | [#1247](https://github.com/anthony-chaudhary/fak/issues/1247) | E2 | per-cell declared TARGET rung; debt = Σ(target − current) only where the regime expects higher |
| **C5** | [#1248](https://github.com/anthony-chaudhary/fak/issues/1248) | E3 | the cross-support tensor — (family × backend × precision) + complete-model-support coverage face |
| **C6** | [#1249](https://github.com/anthony-chaudhary/fak/issues/1249) | E3 | score non-model FEATURES (cache tiers, attention variants, serving features) on the same ladder |
| **C7** | [#1250](https://github.com/anthony-chaudhary/fak/issues/1250) | E4 · the handle | the rung→regime router (R0 explore / R1 prototype / R2 optimize / R3 production) |
| **C8** | [#1251](https://github.com/anthony-chaudhary/fak/issues/1251) | E4 | the work/effort-horizon axis (step-budget per regime); cross-link the #1178 dormancy dual |
| **C9** | [#1252](https://github.com/anthony-chaudhary/fak/issues/1252) | E5 | router emits a per-cell next-action routed to the right loop (idea-scout / dispatch / rsiloop / self-tax) |
| **C10** | [#1253](https://github.com/anthony-chaudhary/fak/issues/1253) | E5 | an R2 cell's optimization IS a long-running `rsiloop` with `shipgate` keep/revert toward its target rung |
| **C11** | [#1254](https://github.com/anthony-chaudhary/fak/issues/1254) | E6 | a `fak support` verb folding the scorecard + router into one per-cell read-out |
| **C12** | [#1255](https://github.com/anthony-chaudhary/fak/issues/1255) | E6 | regenerate `HARDWARE-MATRIX.md` from the scorecard (no hand-typed cells) + living authority row |
| **C13** | [#1256](https://github.com/anthony-chaudhary/fak/issues/1256) | X | the honesty fence — no self-reported promotion, rung can drop, WITNESSED/OBSERVED/MODELED labels |
| **C14** | [#1257](https://github.com/anthony-chaudhary/fak/issues/1257) | X | add support/coverage/correctness/parity/maturity to the concept-disambiguation-scorecard |

## 1. The problem, in one sentence

The word **"supported"** is a single fuzzy token that collapses an entire ladder — *none /
loads / runs / correct / optimized / SOTA-parity / beyond-SOTA* — into one undifferentiated
green cell.

> "Supported" tells you a path exists.
> **It does not tell you whether the path merely runs, runs *correctly*, runs *fast*, or beats
> the best in the world — and that difference is exactly what decides how you should work on it.**

That fuzz **misroutes agentic effort**, which is the operational cost. The same word points a
10,000-step optimization (RSI) loop at something that is actually at *loads-but-doesn't-decode*,
and it ships a *correct-but-slow* path to users as if it were production-ready. The fix is not
more benchmarks; it is to **stop overloading one word** — to position every cell on a graded,
witnessed ladder, and to **derive from that rung the right regime, time-horizon, and tooling**.

## 2. Why now — a named gap, not a new idea

`internal/covmatrix` already classifies **14 model families × 4 backends = 56 cells** into a
four-state `Support` enum and folds it into the live `coverage-matrix` scorecard:

| `covmatrix.Support` | Means | Debt? |
|---|---|---|
| `UNDEFINED` | reachable with neither a fence nor a witness — a silently-wrong path | **yes** (`growth_debt`) |
| `FENCED` | the accelerated path panics *honestly* (`requirePreNorm`/`requireGLMDsaSession`) | no — honest stop |
| `PROOF-PATH-ONLY` | correct on the scalar CPU path, no CI oracle on the accelerated path | no |
| `SUPPORTED` | runs **and** has a CI-runnable witness | no |

This is the **horizontal safety axis**: *is the path present and honest, or silently wrong?* It
is real and load-bearing. But notice what `SUPPORTED` hides: a cell that *barely runs* and a cell
that *beats llama.cpp* are **the same color**. covmatrix's own green cell is the overloaded word
this epic exists to split.

What no surface grades today is the **vertical maturity inside "supported"** — correct →
optimized → parity → beyond-SOTA — each *witnessed*; and nothing **routes** a cell's level to the
right dev process. `docs/HARDWARE-MATRIX.md` is the closest rollup, but it is a hand-typed prose
snapshot that drifts. This epic is the missing instrument. It is **not greenfield**: it extends
covmatrix's cell space and `Support` enum, folds through the existing `pkg/scorecard` control
pane, and reuses `rsiloop`/`shipgate` for the optimization regime.

## 3. What already exists (the substrate this epic unifies)

The honest starting point. None of this is rebuilt; it is extended, ordered, and routed. The epic's
job is to **collapse N parallel "level/tier/class" vocabularies into one ordered ladder**, then add
the regime router on top.

| Surface | What it already gives us | What's missing |
|---|---|---|
| `internal/covmatrix` `Support` (`covmatrix.go:42`+) | 56 (family × backend) cells; `UNDEFINED/FENCED/PROOF-PATH-ONLY/SUPPORTED`; `growth_debt`; `accelerated_coverage`; `OracleInCI`; `StaleProofPath`/`StaleUnwitnessed` | "supported" is one bucket — no rung *inside* it; no precision axis; no regime |
| `internal/compute` `CorrectnessClass` (`compute.go:206`) | `Reference` (bit-exact) vs `Approx` (device FP-order) | a correctness *gate*, not a position on a maturity ladder |
| `internal/compute` `Backend.Tier()` (`compute.go:312`) | runtime capability probe (`scalar`/`avx512`/`sm89`/`sm80`) | a hardware label, not folded into a support level |
| `internal/compute` `Caps` (`compute.go:234`) | optimization flags (`FusedAttn`, `FusedFFN`, `GraphCompile`, …) | advertises a fast path exists — never tied to a *witnessed* speedup |
| `internal/compute` `Dtype` / `KVPrecision` (`compute.go:50`, `kvprecision.go:28`) | precision tiers (F32…Q4_K; KV F32 vs Q8) | a missing tensor axis — support is keyed (family × backend) only |
| `internal/ggufload` preflight (`preflight.go:28`) | `PreflightReady` + 3 `REFUSE_*` verdicts; `FitOK`/`FitUnknown`/`FitTooBig` | a load-time gate, not a rung (it *is* the M2 witness) |
| `internal/model` `UnsupportedArchError` (`arch_support.go:29`) + fences (`kv.go:151`) | typed refusal for un-implemented archs; honest panics over silent divergence | the M0/M1 witnesses — present but uncollated |
| `internal/turnbench` parity (`parity.go:64`) | model card class (`frontier-hosted`/`local-*`/`sota-local-baseline`) + capability/safety/cost parity | the M6/M7 witness — present but not bound to a ladder |
| `BENCHMARK-AUTHORITY.md` | every speed claim traced to a commit + artifact | the M5/M6/M7 *evidence source*, never read by a support scorer |
| `pkg/scorecard` `Fold` + `tools/scorecard_control_pane.py` | the `*_debt` → grade fold + pinned-baseline CI ratchet | no `support_maturity_debt` row registered |
| `internal/rsiloop` + `internal/shipgate` | propose → verify-correct → measure-faster → KEEP/REVERT/ESCALATE; `EvidenceClass`; non-author witness | the engine of the *optimize* regime — never pointed at a ladder rung as its promotion target |
| `internal/bgloop` / `internal/loopmgr` | in-kernel loop runtime + durable lifecycle ledger | the substrate of the *production* regime's continuous gates |

**The unification (the net-new ordering).** Every row above is some notion of "how far along is
this." Today they are N parallel vocabularies a reader must hold in their head. The ladder is the
**single total order** they lower into:

| Rung | Name | The existing witness that PROVES it |
|---|---|---|
| **M0** | none / undefined | `covmatrix` `UNDEFINED`; arch absent → `UnsupportedArchError` |
| **M1** | fenced (honest) | `covmatrix` `FENCED` — `requirePreNorm`/`requireGLMDsaSession` honest panic |
| **M2** | loads / preflight-ready | `ggufload` `PreflightReady` |
| **M3** | runs (proof-path) | `covmatrix` `PROOF-PATH-ONLY` / supported-but-no-oracle |
| **M4** | correct (witnessed) | `covmatrix` `OracleInCI=true` (not `StaleUnwitnessed`); `CorrectnessClass.Reference` gate |
| **M5** | optimized | `Caps` fast-path advertised **and** a committed bench >1× vs the reference path |
| **M6** | SOTA-parity | `turnbench` `sota-local-baseline` parity / `BENCHMARK-AUTHORITY` ≥ ~0.95× vs llama.cpp |
| **M7** | beyond-SOTA | `BENCHMARK-AUTHORITY` > 1× vs the best external |

The ladder **preserves covmatrix's honesty discipline**: M1 `FENCED` is an honest stop (*not* debt);
only M0 `UNDEFINED` is debt. And a rung is non-forgeable: a cell rises **only on a non-author
witness** (the `shipgate` rule), and **drops** when its bound witness regresses (a red oracle
demotes M4 → M3).

## 4. Survey of the art — and what we steal from each

How the engineering world separates "it exists" from "it's done" and routes effort accordingly.

| External practice | What it buys | Imported as |
|---|---|---|
| **Technology Readiness Levels** (NASA TRL 1–9) | one ordinal scale from "basic principles" to "flight-proven", each rung with an *exit criterion* | the M0–M7 ladder with a **witness** as each rung's exit criterion |
| **Capability Maturity Model** (CMM/CMMI) | maturity as a *process* property, not a feature checkbox | the regime router — maturity dictates which *process* applies |
| **Conformance / support matrices** (Khronos CTS, SQL conformance levels, "can I use") | a feature × target grid with graded, tested support | the cross-support tensor (family × backend × precision), cells *tested*, not asserted |
| **Crawl/walk/run rollout staging** | different rigor and ownership per stage | R0/R1/R2/R3 with per-regime tooling + who-operates |
| **Definition-of-Done gates** (agile) | "done" is a witnessed predicate, not a claim | every rung promotion is `shipgate`-gated |
| **SLO/error-budget operation** (SRE) | production is *continuous defense*, not a one-time pass | R3 = the self-tax #1147 regression gate + `bgloop` continuous loops |
| **Effort estimation by phase** (cone of uncertainty) | the *variance* of effort collapses as maturity rises | the work-horizon axis: R0 abandon-cheap ≪ R2 ~1k–10k bounded steps |

**Shortlist worth stealing first:** TRL's *exit-criterion-per-rung* (our witness binding) and CMM's
*maturity-selects-process* (our regime router). The rest fak already has in embryo (covmatrix is a
conformance matrix; rsiloop/shipgate is a DoD gate; #1147 is the SLO defense).

## 5. The two planes

### Plane A — the support-maturity ladder (where each cell sits)

The M0–M7 ladder of §3, witnessed not claimed. Its output per cell is `(rung, witness, provenance
label)`. The scorecard folds it: `support_maturity_debt` = Σ over cells of `(target_rung −
current_rung)` **only** where the cell is below its declared target (C4) — so a *correct-but-slow*
cell whose target is M4 is **zero debt and honestly so**, and the debt names exactly the cells whose
*regime* expects more. Coverage = % of declared cells at-or-above their target rung. The fold rides
the existing control-pane ratchet (C3).

### Plane B — the dev-regime / time-horizon router (how to work on each cell)

Derived deterministically from the rung. This is the **instant handle** — the answer to "I'm looking
at cell X; what kind of work does it want, how long should it take, and what tool runs it?"

| Regime | Rungs | The question | Horizon (steps) | Tooling / loop | Who operates |
|---|---|---|---|---|---|
| **R0 explore** | M0–M1 | "can this even work?" | short, high-variance, **abandon-cheap** | idea-scout, research notes, scouts | human + scout |
| **R1 prototype** | M1–M3 | "make it run end-to-end / get a CI oracle" | ~10–100 | dispatch worker, get-to-green, tests | dispatch fleet |
| **R2 optimize** | M4–M5 | "make it fast / better" | **~1,000–10,000**, long-running | `rsiloop` + `shipgate`, kernel/compiler tooling, benches | autonomous RSI loop |
| **R3 production** | M6–M7 | "keep it that way for users" | continuous / forever | self-tax #1147 gate, SLOs, `bgloop`, UX | gate + on-call |

The mapping is total and deterministic, so the router can emit, per cell:
`{regime, step-budget, recommended tooling, report style, next-action}` (C7/C8), and route that
next-action to the *right loop* (C9/C10). **This is the literal mechanization of the founding ask:
"10 steps to get it working but 10,000 to optimize → different approaches needed."** A cell does not
get an RSI optimization loop until it is M4-correct; it does not get an SLO gate until it is
M6-parity. The regime is read off the rung, not guessed.

## 6. Definition of Done (epic-level — every item WITNESSED, no self-report)

The epic closes when **all** hold, each with a third-party-rederivable witness:

1. **The ladder is closed and lossless.** M0–M7 is an ordered Go enum; every `covmatrix.Support`
   value and every `ggufload` preflight verdict lowers into exactly one rung. *Witness:* a totality
   test over both enums. *(C1)*
2. **Every rung is bound to a witness, and promotion is non-forgeable.** A cell rises only on a
   non-author witness (`shipgate`); a rung drops on witness regression. *Witness:* a planted
   oracle-red demotes a fixture cell; an author-only bench-win does not promote it. *(C2, C13)*
3. **A scorecard folds it.** `support_maturity_debt` + coverage + grade via `pkg/scorecard`,
   registered in the control pane, with declared per-cell targets so the debt is honest. *Witness:*
   the snapshot regenerates deterministically; `--check` reds on a real regression, not on an
   honestly-M4 cell. *(C3, C4)*
4. **The cross-support tensor is real.** (family × backend × precision) cells + a
   complete-model-support coverage face; non-model features scored on the same ladder. *Witness:* the
   tensor enumerates the cross cells; no cell silently `UNDEFINED`. *(C5, C6)*
5. **The regime router is total and routes work.** Each rung yields {regime, step-budget, tooling,
   next-action}, and that next-action dispatches to the right loop. *Witness:* a golden cell at each
   rung emits the expected regime + routed action. *(C7, C8, C9, C10)*
6. **One read-out, one generated matrix.** `fak support` folds the plane into a per-cell
   `rung · regime · target · next-action · witness`; `HARDWARE-MATRIX.md` regenerates from the
   scorecard. *Witness:* the verb is golden-tested; a stale matrix cell reds the freshness check.
   *(C11, C12)*
7. **The words themselves are disambiguated.** support / coverage / correctness / parity / maturity
   position as `crystal` in the concept-disambiguation scorecard. *Witness:* the five rows resolve;
   disambiguation-debt does not rise. *(C14)*

**Explicit non-goals / fences.** (a) This is the **instrument**, not the parity work — #307/#305/#303/#301
close the per-cell gaps; this measures and routes them. (b) A rung is *an envelope with a witness*,
not a verdict of failure — an honestly-M4 cell at an M4 target is **done**, not behind. (c) The
work/effort horizon here (steps to advance a rung) is the **dual** of #1178's session-dormancy
horizon (time a session is away) — cross-linked, never blended into one clock.

## 7. Sequencing

**C1 unlocks everything** (no ladder ⇒ no rung ⇒ no scorecard ⇒ no router). **C2** (witness binding +
promotion rule) gates **C3** (scorecard) and **C13** (the honesty fence). **C3** unlocks **C4/C5/C11/C12**.
**C7** (regime router) is independent of the scorecard and unlocks **C8/C9** and feeds **C11**. **C10**
(rsiloop binding) follows **C7/C9**. **C6** and **C14** are largely independent. The honest **MVP slice is
C1 + C2 + C3 + C7**: a closed witnessed ladder, a debt fold, and the regime handle — the first point at
which "supported" stops being one fuzzy word and becomes a graded, routed answer.

## 8. A worked cell (illustrative — the shape is the contract, not a benchmark claim)

Reading one (family × backend × precision) cell through the instrument, today's witnesses in hand:

| Cell | Bound witness | Rung | Declared target | Debt | Regime | Next-action |
|---|---|---|---|---|---|---|
| Llama × cpu × Q8_0 | `OracleInCI=true`, bench >1× f32 | **M5 optimized** | M5 | 0 | R2→R3 boundary | hold; watch self-tax #1147 |
| Qwen3.6-27B × metal × Q4_K | first-2-token oracle match; no committed speedup row | **M4 correct** | M6 | 2 | **R2 optimize** | open an `rsiloop` toward M5 (the 10k-step lane) |
| GPT-NeoX × cuda × * | `FENCED` (ParallelResidual ≠ PreNorm) | **M1 fenced** | M3 | 2 | **R1 prototype** | dispatch-worker: implement the path to a CI oracle |
| Falcon × vulkan × * | `UNDEFINED` (no fence, no witness) | **M0 undefined** | M1 | growth_debt | **R0 explore** | idea-scout: is this path worth a fence at all? |

The same instrument, read four ways, yields four *different kinds of work* with four *different
horizons* — which is the entire point. The Qwen3.6 cell is not "unsupported" and is not
"production"; it is **M4-correct, R2-optimize, ~10k steps of rsiloop away from M5** — a sentence the
word "supported" could never say.

# Generation-Aware RSI Keep/Revert Gate

The RSI loop keeps a self-improvement candidate only if a witness the candidate's
author did **not** write confirms a strict gain; otherwise it reverts. Today that
keep-bit optimizes **one scalar metric against latest `main`** — the *immediate
local horizon*. This memo defines how the keep/revert gate should also consider a
**generation-specific objective function** and the **witness strength** each
horizon demands, so the loop can optimize against the right horizon instead of
only the tick-local number. It answers issue
[#1657](https://github.com/anthony-chaudhary/fak/issues/1657)'s *Why* — "RSI
should optimize against the right horizon, not just immediate local metrics" —
under epic [#1625](https://github.com/anthony-chaudhary/fak/issues/1625). Closes
[#1657](https://github.com/anthony-chaudhary/fak/issues/1657).

Generation stream: `gen/second-next`. This is an **architectural design memo +
compatibility policy** for the keep-bit — the second-next proof bar (simulation /
compatibility policy / a dependency edge with a demotion criterion), **never
default runtime exposure**. No line of the shipping keep-bit changes here; the
runnable extension is named below as promotion evidence, behind an explicit gate.

## Where this sits (keep the keep-bit grain distinct)

This is the **ship-gate grain** — how the automated keep-or-revert decision
adjudicates one self-improvement candidate. Keep it distinct from the three
existing generation artifacts so they compose instead of overlapping:

| Artifact | Question it answers | Grain |
|---|---|---|
| [Generation Readiness Gates](../docs/notes/GENERATION-READINESS-GATES-2026-06-30.md) (#1644) | *Is the evidence strong enough for an operator to promote this item's horizon?* | one item, at promotion |
| [Multi-Generation Portfolio Optimizer](generation-portfolio-optimizer.md) (#1652) | *Given the whole labeled portfolio, which mix should attention go to?* | the whole portfolio, at dispatch planning |
| [Generation Lifecycle Simulation](../docs/notes/GENERATION-LIFECYCLE-SIMULATION-2026-07-04.md) (#1656) | *How do promote/demote/retire/park move items between streams over time?* | one item, over a timeline |
| **This gate (#1657)** | *When the RSI loop measures a candidate, what objective function and witness strength decide KEEP vs REVERT for its horizon?* | one candidate, at the keep-bit |

The canonical stream / evidence / orthogonality definitions this gate scores
against live in [`docs/generation.md`](../docs/generation.md); this file is the
checkable design, not a second source of truth. The witness-strength axis here is
deliberately the **keep-bit projection** of the same evidence classes those
artifacts use, so a "strong enough to promote" call (#1644) and a "strong enough
to keep" call never disagree on what counts as a witness.

## The keep-bit today (what actually decides KEEP)

Grounded in [`internal/shipgate/shipgate.go`](../internal/shipgate/shipgate.go)
and [`internal/rsiloop/rsiloop.go`](../internal/rsiloop/rsiloop.go):

- `shipgate.Evaluate(Witness)` KEEPs iff, for the candidate's proven
  `EvidenceClass` profile, **(1)** a strict directional gain in one scalar
  `Metric` (`After` beats `Before` under `LowerBetter`), **(2)** `SuiteGreen`,
  and **(3)** `TruthClean` — every field measured, none from the candidate's
  say-so (the *non-forgeable keep-bit*).
- The only graduation today is `EvidenceProfile`, indexed by `EvidenceClass`:
  `ClassFull{gain,suite,truth}`, `ClassDocsOnly{truth}`,
  `ClassProofCarrying{gain,truth}`. Narrower classes drop only a signal the
  harness has **proven** irrelevant; they never add a forgeable input.
- The class is proven by `Harness.Classify(c)`, **never** asserted by the
  candidate; a nil seam or an unprovable narrowing pins `ClassFull` (the strict
  all-three rule). `ClassifyPaths` is the worked example: docs-only *proven from
  touched paths*, not claimed.
- The baseline is re-derived from `main` every `Run` (`Harness.BaselineMetric`),
  so a keep is always a gain over **latest main** — precisely the "immediate
  local metric" the issue wants to generalize.

Two facts make this extensible without weakening it. First, the profile is
already a **witness-strength selector** — the seam a horizon axis rides. Second,
`Measurement.Score *Scorecard` is an **additive telemetry side-channel the
keep-bit deliberately ignores** (`Evaluate` reads only `Metric`, `SuiteGreen`,
`TruthClean`) — the inert place a generation objective *readout* can land first,
before any gate consumes it.

## The gap: one horizon, one objective

The keep-bit's objective is fixed: *strict gain in a tick-local scalar vs main*.
That is exactly right for a `gen/now` candidate and exactly wrong for a
`gen/second-next` one, whose value is **optionality preserved under a
compatibility or simulation witness**, not a number that moves this tick. Under
today's gate a second-next architectural bet that ships a passing simulation and
an additive-only ABI edge but no scalar gain is **REVERTED** — the loop optimizes
it away for failing an objective that was never its horizon's objective.

## The design: a horizon-indexed objective + witness profile

Extend the keep-bit with two horizon-indexed choices, **composed with** (not
replacing) the existing `EvidenceClass` profile:

1. **Objective function** — *which* KPI/witness the horizon optimizes.
2. **Witness-strength profile** — *which* measured signals must corroborate it.

Both are indexed by a **harness-proven horizon**, resolved the same
non-forgeable way `EvidenceClass` is: read from artifacts the candidate did not
author — its issue's `gen/*` label + milestone (from `gh`), or a
`Generation:` commit sidecar the loop reads from git, **never** a field the
candidate sets on itself. A `Harness.ClassifyGeneration(c) Horizon` seam mirrors
the existing `Harness.Classify`.

| Proven horizon | Objective function (what "better" means) | Keep witness required | Metric-gain? |
|---|---|---|---|
| `gen/now` | Strict gain in an immediate trunk KPI vs latest `main` (the legacy objective, unchanged). | gain **and** suite **and** truth | **yes** |
| `gen/next` | Gain in a *foundation* KPI that may live behind a default-off gate; the gate's own test is part of `SuiteGreen`. | gain **and** suite **and** truth, where the suite exercises the gated path | **yes** |
| `gen/second-next` | **Optionality preserved under a compatibility/simulation witness** — a passing simulation *or* an additive-only ABI / no-in-place-edit schema check (per [`docs/generation-abi-compatibility-policy.md`](../docs/generation-abi-compatibility-policy.md)), **plus a named demotion criterion**. No tick-local scalar gain is required. | (simulation- or compat-witness) **and** truth **and** demotion-criterion-present | **no** |
| `gen/future` | None — research/memo work is **not an RSI keep-bit candidate**. It never enters propose→measure→keep-or-revert; it is a planning artifact scored by #1652/#1644. | n/a (out of loop by construction) | n/a |

Mechanically this is one new profile map keyed by horizon, folded into `Evaluate`
alongside `ProfileFor`, plus a per-horizon **objective selector** that names which
measured witness fills the "gain" slot (a scalar delta for `now`/`next`, a
simulation/compat pass for `second-next`). The `Profile{needGain,needSuite,
needTruth}` shape already expresses "drop a signal the horizon proved
irrelevant"; the second-next profile is `{needGain:false, needSuite:false,
needTruth:true}` **augmented** by a non-forgeable compat/sim witness that replaces
`needGain` — the same "narrower class drops only proven-irrelevant signals" rule,
one horizon deeper.

### The forgeability fence (fail-closed)

Dropping the strict-metric-gain requirement for `gen/second-next` is safe **only**
when both the horizon *and* its replacement witness are non-forgeable. So:

- If `ClassifyGeneration` cannot **prove** the horizon from unauthored artifacts,
  it pins `gen/now` semantics → `ClassFull` → today's strict all-three keep. A
  candidate cannot lower its own bar by *claiming* `gen/second-next`.
- If the second-next compat/simulation witness is not itself measured (a real
  sim run, a real ABI-additive check), the horizon-aware keep **fails closed** to
  a REVERT — a self-asserted "this preserves optionality" is never a keep.

This preserves the one rule the whole package exists for: **the keep-bit's inputs
are measured, never the candidate's word.** Generation adds *which* objective and
*which* witness, never a forgeable one.

## Orthogonality (the gate changes none of the three)

- **Priority.** A horizon selects an objective function, not urgency. A
  `gen/second-next` candidate that KEEPs on a compat witness is not thereby
  higher-priority than a `gen/now` one that KEEPs on a throughput gain; it cleared
  *its horizon's* bar. The breaker/escalation logic
  (`shipgate.Gate`, consecutive non-keeps → `ESCALATE`) is unchanged and
  horizon-blind.
- **Shared trunk.** The keep-bit already runs candidates in an isolated worktree
  and **never mutates `main`** (`ApplyInWorktree`); landing a kept change is a
  separate path-scoped commit on `main`. A horizon label authorizes no branch, no
  worktree escape, no stale-trunk exception. Every stream still lands through
  `main`, by explicit path, DCO-signed, `(fak <leaf>)`-stamped.
- **Runtime feature gates.** This is planning metadata for *what evidence keeps a
  candidate*, not a runtime exposure switch. The generation objective *readout*
  lands first in the inert `Measurement.Score` telemetry (which `Evaluate`
  ignores); consuming it in the keep-bit is a **default-off, operator-enabled**
  gate. A `gen/next` candidate may still ship inert behind its own default-off
  feature gate and be kept on its foundation objective — that is correct posture,
  not a contradiction.

## Worked keep/revert readout (the planning witness)

Four candidates through the gate — before (today's single objective) and after
(horizon-indexed), showing the second-next bet flip to KEEP while a **forged
label stays REVERT**. This is the before/after the issue's *Witness* section asks
of a planning artifact.

```text
                              measured witness                     today      horizon-aware
C1 gen/now   proven   scalar +3% vs main, suite✓, truth✓           KEEP       KEEP   (now objective, unchanged)
C2 gen/next  proven   foundation KPI +1%, gated-path suite✓, truth✓ KEEP      KEEP   (next objective, gated suite)
C3 gen/2nd   proven   NO scalar gain; sim✓ + ABI additive-only✓;
                      truth✓; demotion criterion named             REVERT     KEEP   (2nd objective: optionality+compat)
C4 gen/2nd   CLAIMED  NO scalar gain; "preserves optionality"
                      (self-asserted, no sim/compat witness)       REVERT     REVERT (fail-closed: horizon/witness unproven)
```

Reading it: today the gate has one objective, so C3 (a real architectural option
with a passing simulation and an additive-only ABI edge) is optimized away for
lacking a tick-local number it was never meant to move — the exact "immediate
local metric" failure the issue names. Horizon-aware, C3 KEEPs on *its* objective
while C1/C2 are untouched, and C4 — same horizon *claimed* but no measured
compat/sim witness — stays REVERT. The generalization strictly *widens* what can
be kept for proven higher-horizon work and **narrows nothing** for `gen/now`,
because an unproven horizon collapses to today's strict rule.

## Machine-readable schema

An agent or a future `fak` verb emits one object per keep-bit evaluation — the
machine form of the row above, landing first in the inert `Measurement.Score`
side-channel before any gate reads it.

```json
{
  "schema": "fak-generation-keepbit/1",
  "candidate": "C3",
  "horizon": "gen/second-next",
  "horizon_proven": true,
  "objective": "optionality_under_compat_witness",
  "witness": {
    "metric_gain": null,
    "simulation_pass": true,
    "compat_additive_only": true,
    "suite_green": null,
    "truth_clean": true,
    "demotion_criterion_named": true
  },
  "decision": "KEEP",
  "reason": "second-next objective met by measured sim + additive-only ABI witness; horizon proven from issue label"
}
```

`horizon_proven=false` forces `horizon` to `gen/now` semantics regardless of the
claimed label (the fence). A `null` witness field means "not required by this
horizon's objective"; a required field that is `false`/absent is a REVERT.

## Promotion / demotion / assumption (for this artifact)

- **Promotion evidence** (what moves this memo toward `gen/now`): implement a
  `GenerationProfile`/`ClassifyGeneration` seam in
  [`internal/shipgate`](../internal/shipgate/shipgate.go) +
  [`internal/rsiloop`](../internal/rsiloop/rsiloop.go), **default-off** behind an
  explicit gate, with a contract test that reproduces the four-candidate readout
  above exactly (its rows are the fixture) and proves C3 flips to KEEP while C4
  stays REVERT. That green test **plus one captured RSI run** over a real
  `gen/second-next` candidate is the promotion witness. Until then this stays a
  `gen/second-next` design applied by hand or by a planning agent.
- **Demotion / retirement evidence**: retire this design if the RSI loop never
  actually adjudicates higher-horizon candidates — if every real keep-bit
  candidate is `gen/now`, a horizon index is dead weight and should be removed,
  not defended. Demote (park) it if the compat/simulation witness cannot be made
  non-forgeable cheaply, in which case the second-next objective must stay
  fail-closed to `ClassFull` (i.e., the feature is inert) rather than admit a
  weaker keep. Fold-and-delete if a future `fak` verb subsumes the objective
  table into its help text.
- **Invalidating assumption**: the design assumes a `gen/second-next` candidate's
  **objective can be witnessed non-forgeably at keep-time** — that a passing
  simulation and an additive-only ABI/schema check are measurable from artifacts
  the candidate did not author, cheaply, in the isolated worktree. If "preserves
  optionality" can only ever be asserted by the candidate (no measurable
  simulation or compat oracle exists), then a horizon-aware keep is **forgeable**
  and must not be admitted into the keep-bit at all — the honest outcome is to
  keep the strict single-objective gate and route second-next value through
  planning artifacts (#1652/#1644) instead. A second, sharper assumption: the
  per-horizon objective table is a **hand-set prior**, not fit to outcomes; if
  realized keeps under the second-next objective do not correlate with landed
  architectural value, recalibrate the table against realized promotions, do not
  defend it.

## Continue here (no epic reread)

A future agent needs no epic reread. To advance #1657's follow-on:

1. Add `Horizon` + `ClassifyGeneration(c) Horizon` to
   [`internal/rsiloop`](../internal/rsiloop/rsiloop.go), resolving the horizon
   from the candidate's issue label / `Generation:` commit sidecar — **never** a
   candidate-set field (mirror `Harness.Classify`'s non-forgeable seam).
2. Add a horizon-indexed `GenerationProfile` map + objective selector to
   [`internal/shipgate`](../internal/shipgate/shipgate.go) and fold it into
   `Evaluate` **behind a default-off gate**; unproven horizon ⇒ `gen/now` ⇒
   today's `ClassFull` rule (the fence).
3. Gate it with a contract test reproducing the four-candidate readout above; the
   arithmetic/verdicts are the fixture. Assert C3 flips to KEEP and C4 stays
   REVERT.
4. Emit `fak-generation-keepbit/1` into `Measurement.Score` first (inert
   telemetry `Evaluate` ignores), so the objective readout is observable one
   release before any gate consumes it.
5. Keep the **compatibility edge with its demotion criterion**: the witness-
   strength axis MUST remain the keep-bit projection of the evidence classes in
   [`docs/generation.md`](../docs/generation.md) and the readiness gates
   (#1644). If those definitions change, rebind this axis in the same pass — the
   demotion criterion for the whole gate is "the objective/witness no longer maps
   to the canonical generation evidence definitions," which retires it rather than
   letting it drift into a second source of truth.

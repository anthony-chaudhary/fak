---
title: "Generation-Specific Witness Ladders"
description: "The minimum-evidence ladders that let each generation stream (now / next / second-next / future) carry a different proof bar for the same four claim types — planning, code, benchmark, and operator — while every claim still names a witness. A now bug fix and a second-next architecture bet ride different rungs, not different truth standards."
---

# Generation-Specific Witness Ladders

**Issue:** #1659.
**Parent:** #1625.
**Stream:** `gen/second-next`.
**Milestone:** Generation G2 - Second Next Gen.
**Status:** design memo / compatibility policy — a witness-ladder taxonomy a
later `gen/next` or `gen/now` stream can enforce with a lint or `fak hygiene`
gate, not yet a runtime gate. Architectural option: expose only through this
doc; never a default gate.

This memo is the handoff a future agent can use without rereading the whole
generation epic. It answers one question that #1625 raised but did not resolve:
future research and now bug fixes need *different proof bars*, but both need
witnesses — so what is the **minimum evidence** that promotes a claim within
each generation, per claim type? The canonical stream taxonomy and the flat
promotion/demotion evidence list live in [`docs/generation.md`](generation.md);
this memo is the *ladder* refinement of that list: it turns "code work needs a
focused test" into a per-generation rung with an explicit floor.

## Why a ladder and not one bar

The generation contract already says *what kind* of evidence each surface owes
(a code claim owes a test; an operator claim owes a before/after readout). It
does not say *how much*. That gap is the contention #1625 exists to kill:

- Hold `gen/future` to the same measured-benchmark bar as `gen/now` and no
  research ever clears intake — the horizon dies of proof cost.
- Hold `gen/now` to the same simulation-is-enough bar as `gen/future` and a bug
  fix ships on a plausible story with no repro — the trunk rots.

A ladder fixes both: the *witness type* is constant across generations (a claim
always names its evidence), but the *rung* — the minimum strength of that
evidence — rises as work climbs from `future` toward `now`. Same truth
standard; different proof cost, matched to horizon.

The load-bearing invariant: **no rung is ever "no witness."** The lowest rung is
still a checkable artifact (a named assumption, a simulation input, a committed
note). A generation label lowers the *cost* of the witness, never removes it.

## The four claim types

Every generation issue makes at least one of these claims. The ladder is defined
per claim type because their cheapest honest witness differs:

- **Planning claim** — "this is the right shape / horizon / dependency edge."
- **Code / CLI claim** — "this behavior now exists and is correct."
- **Benchmark claim** — "this is faster / cheaper / higher-scoring by N."
- **Operator / loop claim** — "the loop has less ambiguity, contention, or
  stale assumption than before."

## The ladders

Read a column as: *to make this claim at this generation, the minimum witness
is X.* A claim may always over-witness (ship a `gen/future` idea with a real
test); it may never under-witness (ship a `gen/now` benchmark claim on a
simulation).

### Planning claim ladder

| Generation | Minimum witness (the floor) |
|---|---|
| `gen/future` | A committed memo naming the decision it could influence **and** at least one invalidating assumption. No integration surface required. |
| `gen/second-next` | The `gen/future` floor **plus** a named cross-generation dependency edge or compatibility policy, **plus** a stated demotion/retirement criterion. |
| `gen/next` | The `gen/second-next` floor **plus** a named integration surface (the exact file/verb/schema it will touch) and a dogfood or gate plan. |
| `gen/now` | A committed note, issue update, project-field change, or saved view a later agent can act on today, bound to a milestone, with the witness path already runnable. |

### Code / CLI claim ladder

| Generation | Minimum witness (the floor) |
|---|---|
| `gen/future` | A prototype or spec **behind an explicit gate** (default-off, or doc-only), with the assumption it would test named. Inert-by-design is fine. |
| `gen/second-next` | A compatibility test or simulation showing the new behavior does not break an existing reader/caller across generations — proof-of-safety, not yet proof-of-value. |
| `gen/next` | A focused test that fails before, passes after, covering the generation-specific claim, plus one captured live run — behind a default-off gate until dogfood clears. |
| `gen/now` | A focused test (fail→pass) landed in the **same commit**, plus a witnessed commit (`(fak <leaf>)` stamp, `dos verify`). Default exposure allowed. |

### Benchmark claim ladder

| Generation | Minimum witness (the floor) |
|---|---|
| `gen/future` | A *modeled* estimate with its inputs stated and labeled **MODELED / projected** — never "measured." The simulator (`docs/generation-future-proof-of-value-simulator.md`) output counts here. |
| `gen/second-next` | A simulation or micro-measurement on a fixture, with the counterfactual named and the extrapolation-to-real-workload assumption stated. Still labeled projected. |
| `gen/next` | A captured measurement on a real (if small) workload, reproducible from a named command, labeled **MEASURED**, with the baseline it beats named. |
| `gen/now` | A captured, reproducible measurement with baseline, N, and command, provenance-labeled `WITNESSED` (fak authored) vs `OBSERVED` (relayed), passing the provenance guard. Regression-gated if it claims a default. |

The benchmark ladder is why this issue routed to the `bench` lane: it is the
column where the "measured vs modeled" provenance discipline
([`tools/check_provenance_labels.py`](../tools/check_provenance_labels.py)) does
the enforcing, and where `internal/bench` fan-run output is the `gen/now`-rung
artifact. A benchmark claim that labels a MODELED number "measured" is not a
low rung — it is a **broken** witness and fails CI at every generation.

### Operator / loop claim ladder

| Generation | Minimum witness (the floor) |
|---|---|
| `gen/future` | A described mechanism and the ambiguity/contention it would reduce, with the metric that would show it named — no live readout required. |
| `gen/second-next` | A simulated or single-instance before/after (one dispatch trace, one lease decision) showing the mechanism changes the readout in the expected direction. |
| `gen/next` | A real before/after readout from a dogfood run showing less ambiguity, contention, or stale assumption, behind an operator-only gate. |
| `gen/now` | A captured before/after operator readout from the live loop, default-on, showing the specific ambiguity/contention/stale-assumption it retired. |

## Orthogonality (the generation invariants this artifact must restate)

The ladders are metadata and a proof-cost policy — not a branch, a priority, or
a runtime switch.

- **Orthogonal to priority.** A rung is not a value judgment. A `gen/future`
  planning claim on the lowest rung can be the highest-priority thing to study;
  a `gen/now` claim on the highest rung can be trivial cleanup. The ladder sets
  the *minimum proof cost* for the horizon, not the *importance* of the work.
  `gen/future` is a horizon label, never "lower priority" — matching #1659's
  non-goal.
- **Orthogonal to shared trunk.** Every rung lands on `main`, by explicit path,
  under the same DCO and ship-stamp rules. The ladder explicitly forbids the
  per-generation branch (#1659 non-goal): a lower rung buys a cheaper witness,
  never a side worktree or a stale-trunk exemption. All four generations climb
  the same trunk.
- **Orthogonal to runtime feature gates.** A rung decides *how much evidence
  promotes a claim*; a feature gate decides *whether the code is reachable at
  runtime*. `gen/future` and `gen/next` code rungs explicitly require a
  default-off gate as part of the witness, but the gate is the exposure control
  and the rung is the evidence control — neither substitutes for the other. A
  `gen/now` code claim can be default-on precisely because its rung (fail→pass
  test + witnessed commit) is the strongest, not because the label unlocks
  exposure.
- **Orthogonal to completion percentage.** A ladder rung is not a progress bar.
  An ongoing optimization program reports a frontier at whatever rung its claims
  currently clear; it does not fake a percentage by climbing a rung it has not
  witnessed.

## Promotion evidence (future → second-next → next → now)

This memo promotes when a later stream can *enforce* the ladder, not just state
it:

- **future → second-next:** a worked example per claim type — one real issue at
  each rung — showing the ladder classifies actual fak work, not just toy cases.
  Naming those four issues is the promotion witness.
- **second-next → next:** a lint or `fak hygiene` gate that reads an issue's
  generation label + claim type and *refuses* a commit whose attached witness is
  below the floor rung (e.g. a `gen/now` benchmark claim with a MODELED label).
  The benchmark column can reuse the existing provenance guard as its first
  wired rung.
- **next → now:** that gate runs default-on in `make ci`, and the milestone /
  release report reads each generation lane's rung coverage as a debt input, so
  an under-witnessed claim is caught at commit time rather than in review.

## Demotion / retirement evidence

- **Demote** if a rung proves miscalibrated in practice — e.g. the
  `gen/second-next` code rung ("compatibility test, not proof-of-value") lets a
  bet accumulate that never clears the `gen/next` value rung, so the ladder is
  hiding over-carry rather than pacing it. That pushes the ladder back to
  `gen/future` for a recut of the second-next rung.
- **Retire** the memo if the flat evidence list in
  [`docs/generation.md`](generation.md) plus a single wired gate fully subsume
  the ladders; then this doc collapses into a pointer from the hub with no
  standalone taxonomy left to carry.
- **Retire** if generations stop carrying different proof costs (all work
  converges on one bar), which would make per-generation rungs vacuous.

## Invalidating assumptions (kill criteria)

State them so a later agent can check them cheaply:

1. **Four claim types are enough.** This memo assumes planning / code /
   benchmark / operator partition every generation claim. If a fifth honest
   claim class appears whose cheapest witness fits none of these columns (a
   security-posture claim, a cost-of-ownership claim), the ladder must gain a
   column, not stretch an existing one by analogy. **This is the assumption most
   likely to fail** — the benchmark column is already doing double duty for
   "cost" claims that may deserve their own ladder.
2. **The floor is honestly cheaper, not just cheaper.** The whole design rests
   on a lower rung being a *real* witness (a checkable assumption, a labeled
   model) rather than a dressed-up "no witness." If in practice the `gen/future`
   rungs get filled with unfalsifiable assumptions that no recheck ever fires,
   the ladder has laundered "no proof" into "low rung" and must add a
   falsifiability check to each lowest rung.
3. **The rung floors are enforceable from cheap signals.** Promotion assumes a
   gate can read generation label + claim type + attached witness class at
   commit time. Today only the benchmark column has a wired enforcer (the
   provenance guard); the other three floors are asserted, not checked. Until the
   `second-next → next` gate exists, a commit can sit below its floor and nothing
   refuses it.

## Handoff (continue from here without the epic)

A future agent picking this up should: (a) pick one real open issue per
generation stream and tag which rung its witness currently clears — that is the
`future → second-next` promotion witness and it also stress-tests assumption 1;
(b) wire the `second-next → next` gate benchmark-column-first, since the
provenance guard already exists and only needs the generation-label + claim-type
read bolted on; (c) if a claim type is found that fits no column, cut a new
column rather than overloading the benchmark one. The hub
([`docs/generation.md`](generation.md)) stays the front door; this memo is the
minimum-evidence refinement of its Evidence section.

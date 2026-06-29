---
title: "EPIC #1147 — The self-tax plane: first-class evidence that fak's own methods don't degrade performance"
description: "A first-class, always-on assurance plane that proves — turn-by-turn, post-session, and in CI — that the kernel's gates/guards/verification cost no more than their declared budget, and that names it when they make work faster. Mechanizes net-true-value Question #2 (net-of-cost), the one rubric question the standard calls least mechanized."
---

# EPIC #1147 — The self-tax plane

> **Status:** OPEN · roll-up tracking note for epic [#1147](https://github.com/anthony-chaudhary/fak/issues/1147).
> **Lanes:** spans `metrics`, `gateway`, `sessionobs`, `turnbench`, `adjudicator`, `docs`.
> **Anchor standard:** [`net-true-value`](../standards/net-true-value.md) Question #2.
> **Sibling (different axis):** [`track-b-performance-parity #306`](track-b-performance-parity-tracking-306.md)
> tracks fak-vs-llama.cpp *raw inference* parity; this epic tracks fak's own *mediation* overhead.

## 1. The problem, in one sentence

fak inserts itself into the hot path of every tool call, every result, every turn, every
commit, every session. Each insertion has a cost — latency, tokens, wall-clock, and
sometimes a changed answer. We can prove a *bad* call can't get through the security floor.
We cannot yet prove a *good* call doesn't get slower than its budget — nor, when fak makes
work faster, do we report that with the same rigor we report a safety win.

> The security floor proves a bad call can't get through.
> **The self-tax plane proves a good call doesn't get slower than its budget — and names it
> when fak made it faster.**

This is the missing dual of fak's own self-description. The README calls fak *both* a
security gate *and* a "performance gate." It has a default-deny **security floor** the model
can't talk past. It has **no equivalent performance floor** — a cost the methods can't
silently exceed without a witness firing.

## 2. Why now — this is a named gap, not a new idea

[`net-true-value.md`](../standards/net-true-value.md) is fak's standard for any
efficiency/perf claim: a gain is reported only if it's measured against the real baseline,
**net of the cost it adds**, scope-stated, provenance-labeled, reproducible, and realized. The
standard grades fak's claims about the *world*. It does not yet grade fak's claims about
**itself** with a single always-on mechanism. The standard says so in its own honest fences:

- *"Question 2 (net of introduced cost) is the least mechanized … a claim that quietly omits
  its own cost still relies on review. Closing that gap is the highest-leverage next stick."*
- *"This is a standard plus a lens over existing sticks, not a single `fak claim-check` verb …
  That verb … is the named follow-on, not built here."*

This epic is exactly that build-out: **mechanize Question #2 across the whole lifecycle, and
ship the `fak claim-check` verb the standard names.** It is not greenfield — fak already
measures its own cost in many places. The epic's job is to **promote scattered, one-off
measurements into one first-class, always-on assurance plane with declared budgets and CI
gates.** "First-class" = there is a budget, a witness, and a gate — not a notebook.

## 3. What already exists (the substrate this epic unifies)

The honest starting point. None of this is rebuilt; it is extended, wired, and gated.

| Surface | What it already gives us | What's missing |
|---|---|---|
| `internal/metrics` `Hist` + `Report` | p50/p99/mean latency histograms; A/B on-vs-off report with an identical-workload guard | not folded per-rung; no budget; not gated |
| `internal/kernel` `Counters` | atomic tallies: Submits, VDSOHits, EngineCalls, Denies, Transforms, Quarantines, ResultDenies, Admitted | counts, not costs; no per-turn tax |
| `internal/rungobs` + `fak rungstats` | passive per-`(rung,kind,reason)` **verdict** distribution; live twin `fak_gateway_operation_duration_seconds{adjudicator-rung}` | verdict only offline; cost not folded into the offline read-out |
| `internal/gateway/metrics.go` (~2.4k lines) | per-rung **latency** histogram; `kv_prefix_reused_tokens`/`reuse_ratio` (realized reuse); compaction **WITNESSED** vs **OBSERVED** token split; `fak_fleet_value_{turns_saved,pollution_blocked,agent_seconds_served}` | no single net-true verdict; no budget-breach signal; no improvement framing |
| `cmd/turntaxdemo` | a per-turn **tax/overhead breakdown** (kernel submission cost by rung/op-kind) | a demo, not a first-class `fak` verb or a gated meter |
| `internal/turnbench` | seeded A/B engine: PolicyReplay (predicted vs actual token/time), **LeverFlip** (per-lever ablation attribution), FleetCounterfactual, BreakEven, DivergenceHistogram, LongContext | not wired to a CI gate; no change-point/SPRT |
| `cmd/fak/ablate` (#607 vDSO sweep, #623 cross-agent bare-vs-guard) | feature on/off ablation with the A/B/C baseline letters | one-off experiment, not always-on, not budgeted |
| `internal/sessionobs` | session capture ladder; HARD KPIs `outcome_link_rate` + `value_waste_separable` | **the named missing rung** — sessions aren't tied to a value/waste outcome |
| `internal/modelroute` `Judge` (mock or LLM-backed) | model-as-judge substrate (used for routing decisions) | not used to detect *quality* regression from fak's interventions |
| `internal/benchscore`, `internal/cadencereport`, `internal/benchcatalog` + `fak benchmarks` | benchmark matrix verify; control-pane gate fold; benchmark registry/discovery | no self-tax row; no net-of-cost gate |
| ablation knobs | `--vdso=off`, `FAK_NORMGATE`, `FAK_IFC`, `FAK_IFC_GATE_EXEC`, `FAK_SECRETGATE`, `RungProfile` (#665/#666) | present; the harness to sweep them as a gate is the work |

The clean **lifecycle stages** are the cost-attribution boundaries the plane folds over:
**Submit** (adjudication: preflight → gitgate → egressfloor → ifc-sink → adjudicator → witness)
· **Reap** (result admission: ctxmmu → normgate → secretgate → ifc-stamp → recall)
· **per-turn** (contextq, compaction/reset, compactcohere) · **decode** (in-kernel planner +
RadixAttention reuse) · **background** (bgloop, dispatch worker). The one rung whose tax is
subprocess-spawn-bound, not compute-bound, is the **witness gate** (it spawns `git`) — it must
be measured separately so a slow disk doesn't read as a kernel regression.

## 4. Survey of the art — and what we steal from each

How the systems/ML world answers "is my interceptor/guardrail silently making things worse?"

| External technique (source class) | What it buys | Imported as |
|---|---|---|
| Bare-proxy baseline + incremental feature enablement (Istio/Envoy mesh tax) | isolates each feature's marginal cost | already our A=naive/B=tuned/C=fak letters; extend to per-rung budgets |
| Per-percentile P50/P95/P99, never mean (tail tax) | guardrails hurt the tail first | the meter reports percentiles, not averages |
| In-kernel aggregation + sampling, <2% overhead (Parca / Pyroscope / Google-Wide Profiling) | continuous, production-safe self-measurement | sample the event stream; never full-instrument the hot path |
| Change-point detection on noisy benchmark series (Hunter / USENIX SREcon) | robust regression flag vs a brittle fixed threshold | the CI gate fires on a distribution shift, not a single spike |
| CI-width tracking, gate on "exceeds **and** persists ≥3 runs" (Criterion / Chromium) | catches a small real change through noise | the gate's persistence rule |
| Bisection for attribution (Chromium Pinpoint) | pinpoint the culprit commit | `dos commit-audit` + `git bisect` on a budget breach |
| Instrumentation (10–53%, variable) vs sampling (1–2%, stable) (observer-effect lit.) | name the cost of measuring | measure & bound the meter's own overhead — an honesty fence |
| LLM-as-judge + human-correlation validation + position-swap debias + smoothed metrics (judge-reliability lit.) | detect **quality** regression, not just speed | L4 slow tier on `modelroute.Judge` |
| Three-tier fast/slow: bench-pass → shadow/canary → unit-economics | cheap gate first, expensive proof rarely | L1 cheap continuous → L3/L4 deep periodic |
| SPRT / always-valid sequential testing | stop an A/B early when evidence is strong (~50% fewer samples) | the ablation gate's stopping rule |
| Interleaving / shadow arms / feature-flag ablation (IR ranking, ML rollout) | clean counterfactual "with vs without" | `turnbench` LeverFlip + `ablate`, generalized |

**Shortlist worth stealing first:** change-point detection, the persistence rule, the
observer-effect cap, and the LLM-judge human-correlation check. The rest fak already has in
embryo.

## 5. The maturity ladder (the spectrum, from literal items to observability)

Each rung is a cluster of tickets. The plane is "done at a rung" when that rung has a budget,
a witness, and (where it gates) a CI hook.

- **L0 — Cost is emitted at all.** Every hot-path span carries its elapsed-ns and token-delta;
  the offline `rungstats` read-out folds cost, not just verdict. *(T1, T2)*
- **L1 — Turn-by-turn / moment-by-moment.** A first-class per-turn overhead meter: kernel-ns
  vs engine-ns, tokens added (transform/quarantine) vs saved (vDSO/radix), against a declared
  budget — with the meter's own cost bounded. *(T3, T4)*
- **L2 — Post-session net-true ledger.** At session end: HELPED / WASH / HURT, provenance-
  labeled, tying cost to the value/waste outcome — closing sessionobs's named gap. *(T5)*
- **L3 — Benchmark regression gate (the guarantee).** Always-on fak-on vs fak-off on a frozen
  workload, change-point + persistence + SPRT, red the tree on a persistent over-budget
  regression. *(T6, T7, T8)*
- **L4 — Quality, not just speed (fast/slow judge).** A fast deterministic check that an
  intervention didn't drop a legit result, and a slow model-as-judge that grades intervened vs
  un-intervened answers, human-correlation-validated. *(T9, T10)*
- **L5 — Observability.** One `fak perf` verb and one net-true `/metrics` family fold the plane
  into a single read-out; a living self-tax authority row tracks the trend. *(T11, T12)*
- **L6 — Detect improvement (the positive case).** Surface realized reuse wins as a net-true
  positive per session/fleet — the "even detect if it's *increasing* performance" ask. *(T13)*
- **X — Cross-cutting.** The `fak claim-check` verb (grade any perf claim vs the six-question
  rubric) and the observer-effect/provenance honesty doc. *(T14, T15)*

## 6. Definition of Done (epic-level — every item WITNESSED, no self-report)

The epic closes when **all** hold, each with a third-party-rederivable witness:

1. **Cost is emitted.** Every hot-path lifecycle span carries elapsed-ns + token-delta; an
   observer folds per-rung cost. *Witness:* a test asserts the event stream carries non-zero,
   correctly-attributed cost spans across Submit/Reap/decode.
2. **A budget exists.** A declared per-rung/per-method overhead envelope (expected ns + token
   cost) is the baseline a breach is defined against. *Witness:* the table is committed and a
   test reads an over-budget span back as a breach.
3. **A per-turn meter exists** reporting kernel-ns vs engine-ns and tokens added vs saved vs
   budget, live on `/metrics` and offline via a verb — and the meter's **own** overhead is
   measured and < a declared cap. *Witness:* a golden turn round-trips; the meter-cost bench is
   under cap.
4. **A per-session net-true ledger** emits HELPED/WASH/HURT with WITNESSED/OBSERVED/MODELED
   labels, closing sessionobs's outcome-link rung. *Witness:* golden sessions (one reuse-
   favorable, one intervention-heavy) produce the expected verdicts.
5. **An always-on CI gate** compares fak-on vs fak-off on a frozen workload, uses change-point
   + persistence + SPRT, and reds `make ci` on a persistent over-budget regression. *Witness:*
   a synthetic injected regression reds it; a noise-only run does not (no false red).
6. **A quality-regression check** exists in both tiers (fast deterministic + slow model-as-
   judge), with a human-correlation validation step on the judge. *Witness:* a planted
   degradation fixture is caught; a benign control is not flagged.
7. **One read-out.** `fak perf` + a net-true `/metrics` family fold the plane into a single
   answer, and a living BENCHMARK-AUTHORITY self-tax row tracks the trend. *Witness:* the verb
   output is golden-tested; the metric reads back.
8. **Improvement is detected**, not only non-degradation — realized reuse surfaces as a
   net-true positive, double-count-guarded. *Witness:* a reuse-favorable trace reports a
   positive net with provenance and the provider-vs-local split intact. *(L6/T13 detector +
   worked acceptance trace surfaced in [§9](#9-l6--the-improvement-detector-worked-example);
   the executable `/metrics`-fed verb is the named follow-on.)*
9. **The honesty fences hold.** Every number passes the six-question net-true rubric;
   `fak claim-check` grades an arbitrary perf claim; the observer-effect doc states the
   perf-floor/security-floor duality and the meter's own cost. *Witness:* `fak claim-check`
   self-test; the doc + cap test ship.

**Explicit non-goals / fences.** (a) This is fak's *mediation* overhead, not fak-vs-llama.cpp
raw-inference parity (#306) — cross-linked, never blended. (b) GPU wall-clock arms stay
hardware-gated and labeled MODELED until run on a real device. (c) A budget is an *envelope
with a stated scope*, not a promise of zero cost — a gate that costs 8% and saves 40% is a net
win, and the plane must say so rather than red on the 8% alone.

## 7. Tickets

> The `#Tn` labels map to the GitHub children filed under epic #1147 (see the epic's Issues
> checklist for the live `#N` mapping). Each ticket names the package it extends — the work is
> wiring, not greenfield — and ships its own witness.

### L0 — emit the cost
- **T1 · `metrics`/`abi`/`rungobs`: carry per-span cost; fold it into `rungstats`.** Add
  elapsed-ns + token-delta to the lifecycle event spans (`EvSubmit→EvDecide` = adjudication
  tax; `EvDispatch→EvComplete` = engine; token-delta from transform/quarantine vs vDSO/radix).
  Teach `rungobs`/`fak rungstats` to fold **cost**, unifying the offline read-out with the live
  `fak_gateway_operation_duration_seconds{adjudicator-rung}`. *Witness:* test asserts non-zero,
  correctly-bucketed cost across the three spans.
- **T2 · overhead **budget** envelope.** Declare expected per-rung/per-method ns + token cost (a
  dos.toml-style table or a typed Go table beside the reasons vocabulary). This is the missing
  "expected" a breach is defined against. *Witness:* committed table; a synthetic over-budget
  span reads back as `OVERHEAD_BUDGET_EXCEEDED`; `dos check-reason` resolves the token.

### L1 — turn-by-turn meter
- **T3 · promote `cmd/turntaxdemo` → first-class `fak turntax` meter.** Per-turn tax: kernel-ns
  vs engine-ns, tokens added vs saved, vs the T2 budget; live on `/metrics` and offline.
  *Witness:* golden turn → golden tax table; budget breach observable. *(depends on T1, T2)*
- **T4 · observer-effect fence.** The meter **samples** (rate-bounded), never full-instruments
  the hot path; a bench proves the meter's own overhead < a declared cap. *Witness:* meter-cost
  bench under cap; sampling rate honored under load. *(depends on T3)*

### L2 — post-session net-true ledger
- **T5 · close sessionobs's outcome-link rung.** A per-session net-true ledger emitting HELPED /
  WASH / HURT, provenance-labeled, tying session cost (tokens added by interventions, ns of
  mediation) to the value/waste outcome (`outcome_link_rate`, `value_waste_separable`), reusing
  `cadencereport`. *Witness:* two golden sessions produce the two expected verdicts.

### L3 — benchmark regression gate (the guarantee)
- **T6 · always-on CI regression gate.** Promote `ablate` (#607/#623) + `turnbench` LeverFlip to
  a gate: fak-on vs fak-off on a frozen canonical workload vs the T2 budget; red `make ci` on a
  persistent over-budget regression. *Witness:* wired into `make ci`; a synthetic regression
  reds it. *(depends on T2)*
- **T7 · change-point detection on the series.** Replace a brittle fixed threshold with
  change-point detection over stored runs; gate on "exceeds **and** distribution-shifts".
  *Witness:* an injected step-change is flagged; stationary noise is not. *(depends on T6)*
- **T8 · SPRT sequential early-stop** for the ablation A/B arms (turnbench is already seeded):
  stop when evidence crosses a boundary, with a futility boundary for "not improving".
  *Witness:* the gate reaches the same verdict on ~half the samples vs fixed-N. *(depends on T6)*

### L4 — quality, not just speed (fast/slow judge)
- **T9 · quality-regression FAST tier (deterministic).** A cheap continuous check that
  repair/quarantine/deny didn't drop a *legit* result — the AgentDojo benign-controls +
  bit-identity pattern, run on every gated run. *Witness:* a benign result wrongly dropped reds;
  a correct quarantine does not.
- **T10 · quality-regression SLOW tier (model-as-judge).** Reuse `modelroute.Judge` to grade the
  intervened answer vs the un-intervened one (pairwise win-rate, position-swap debias), periodic;
  validate the judge against human labels (Spearman ≥ 0.7 or the metric is flagged untrustworthy).
  *Witness:* a planted degradation is caught; the human-correlation check runs.

### L5 — observability
- **T11 · `fak perf` verb + one net-true `/metrics` family.** Fold turntax + the session ledger +
  the ablate gate + benchscore into one read-out (rungstats-for-the-whole-plane). *Witness:* verb
  output golden-tested; metric family reads back. *(depends on T3, T5, T6 — contract pinned in
  [§10](#10-l5--the-fak-perf-read-out--net-true-metrics-family-pinned-contract); executable build
  is the named follow-on, blocked until the three deps land)*
- **T12 · living self-tax authority row + trend doc.** A BENCHMARK-AUTHORITY row that tracks fak's
  own overhead and net effect over time, net-true-labeled. *Witness:* the row traces to a
  committed artifact + reproduce command, like every other authority row.

### L6 — detect improvement
- **T13 · improvement detector** ([#1170](https://github.com/anthony-chaudhary/fak/issues/1170)).
  Surface realized reuse wins (`kv_prefix_reused_tokens`, `fak_fleet_value_*`) as a net-true
  **positive** per session/fleet, double-count-guarded by `cachemeta` (provider vs local reuse).
  *Witness:* a reuse-favorable trace reports a positive net with the provider-vs-local split
  intact — surfaced as the worked detector in [§9](#9-l6--the-improvement-detector-worked-example),
  grounded in the two disjoint live counters and the cachemeta plane split.

### X — cross-cutting
- **T14 · `fak claim-check` verb** (the named net-true follow-on). Takes a claim + baseline +
  witness; returns net-true / strawman / not-yet against the six questions. *Witness:* graded
  self-tests over a fixture of honest and strawman claims.
- **T15 · observer-effect + provenance honesty doc.** States the perf-floor/security-floor
  duality, requires WITNESSED/OBSERVED/MODELED on every overhead number, and pins the meter's own
  measured cost. *Witness:* doc ships; the meter-cost cap test it cites is green.

## 8. Sequencing

T1→T2 unlock everything (no budget ⇒ no breach ⇒ no gate). T3/T4 (the meter) and T5 (the session
ledger) are the first user-visible value and can land in parallel once T1/T2 are in. T6 depends
on T2's budget; T7/T8 harden T6. T9/T10 (quality) are independent of the speed ladder and can
proceed in parallel. T11/T12 (observability) fold whatever has landed. T13 (improvement) and
T14/T15 (cross-cutting) can land anytime after T1. The honest minimum viable slice is
**T1 + T2 + T3 + T6**: cost emitted, budgeted, metered per turn, and gated in CI — the first
point at which "fak isn't silently degrading performance" stops being a hope and becomes a
witness.

## 9. L6 — the improvement detector (worked example)

The positive case of [`net-true-value`](../standards/net-true-value.md) Q2: not "fak didn't
get slower" but "fak made this work *faster*, and here is the realized win, net of its own
cost, with no token counted twice." T13's ask is to **surface** realized reuse as a net-true
positive per session/fleet — the two reuse counters already exist; what was missing is the
single net framing over them and the explicit double-count guard. Both are below.

### 9.1 The two reuse populations are already disjoint

Realized reuse arrives on two structurally separate planes. fak does not have to *compute* a
de-dup — `internal/cachemeta` already keeps them apart, so summing them cannot count one token
twice:

| Plane | Live counter | `cachemeta` adapter / verdict | Provenance |
|---|---|---|---|
| **local** (fak's own in-kernel RadixAttention KV-prefix match) | `fak_gateway_kv_prefix_reused_tokens_total` (+ `…_reuse_ratio`) | `FromKVPrefix` → `Plane=kv_prefix`, `AdmissionAllow` — serveable local trust | **WITNESSED** (the kernel did the reuse) |
| **provider** (the upstream model's own prompt cache, `cache_read`) | `fak_gateway_inference_cached_prompt_tokens_total` | `FromProviderCache` → `Plane=provider`, `Residency=provider`, `AdmissionDefer`; `ProviderCacheVerdict` returns `provider_cache: cost_latency_only` (never `CanServe`) | **OBSERVED** (provider-relayed) |

The metric help text states the disjointness in both directions: the local counter is
"Distinct from the provider's cache_read," and the provider counter is "provider-side reuse —
distinct from the local … caches — and reads 0 on the in-kernel path (no provider)." The
separation is also load-bearing for trust, not just accounting: `fak_gateway_provider_cache_local_trust`
is structurally `0` (#432 acceptance 3), so a provider `cache_read` is never re-served as a
local hit. **A token is either local-reuse or provider-reuse, never both — that IS the
double-count guard.**

### 9.2 The net

```
realized_reuse_tokens = local_reuse + provider_reuse           # disjoint planes — no double count
                      = kv_prefix_reused_tokens_total           (WITNESSED, in-kernel)
                      + inference_cached_prompt_tokens_total     (OBSERVED, provider-relayed)
mediation_tax_tokens  = tokens the kernel ADDED in scope        (transform/quarantine re-emits; MODELED)
net_tokens            = realized_reuse_tokens − mediation_tax_tokens
improvement  ⇔  net_tokens > 0
```

The net is denominated in **prefill tokens not redone, net of tokens added** — never the
vs-naive `1/(1-reuse)` re-prefill multiple, which the #1066 honesty fence excludes (mirrored
from `cachevalueledger.PublishableValueFamily`). Per-session uses one PID's counters;
per-fleet sums the same disjoint counters across the served fleet — the guard holds at both
scopes because the planes are disjoint per token, not per session.

### 9.3 A reuse-favorable trace reports a positive net

A 6-turn `fak serve` session over a provider with a stable system+tools prefix (illustrative
numbers, NOT a benchmark claim — the shape is the witness; the live values come from the
counters above):

| span | value (tokens) | account | provenance |
|---|---|---|---|
| local KV-prefix reuse | 9,400 | `fak_gateway_kv_prefix_reused_tokens_total` | WITNESSED (in-kernel RadixAttention) |
| provider `cache_read` | 3,200 | `fak_gateway_inference_cached_prompt_tokens_total` | OBSERVED (provider-relayed) |
| **realized reuse (sum)** | **12,600** | local ⊕ provider (disjoint planes) | — |
| mediation tax (added) | −180 | one `grammar_repair` transform re-emit | MODELED |
| **net** | **+12,420** | `net_tokens > 0` → improvement | — |

**Positive net: +12,420 tokens. Split intact:** local 9,400 (74.6%) ⊕ provider 3,200 (25.4%),
reported as two numbers and never collapsed — exactly because `cachemeta` keeps the two on
disjoint planes. This is the T13 / DoD #8 acceptance, met by construction.

### 9.4 Reproduce

```sh
curl -s localhost:PORT/metrics | grep -E \
  'kv_prefix_reused_tokens_total|inference_cached_prompt_tokens_total|turns_saved_total|provider_cache_local_trust'
# net_tokens = (kv_prefix_reused_tokens_total + inference_cached_prompt_tokens_total) − tokens_added
# the two reuse counters are disjoint by construction (metric help + cachemeta plane split),
# so their sum is realized reuse with the provider-vs-local split intact.
```

**Named follow-on (out of this docs-lane increment):** an executable `fak`-verb / `/metrics`
fold that emits `net_tokens` and the labeled split directly (rather than a reader summing two
counters) belongs in the `gateway`/`metrics` lane and is the deeper close of T13's "surface"
verb — this section pins the detector's definition, guard, and worked acceptance so that build
is wiring against a fixed contract.

## 10. L5 — the `fak perf` read-out + net-true `/metrics` family (pinned contract)

T11 ([#1168](https://github.com/anthony-chaudhary/fak/issues/1168)) is the *fold*: take the
per-turn meter (T3), the session ledger (T5), the ablate gate (T6), and `internal/benchscore`
and surface them through **one** offline read-out — `fak perf`, *rungstats-for-the-whole-plane*
— and **one** net-true `/metrics` family. The acceptance is code: *verb output golden-tested;
metric family reads back.* That code is **blocked on its three deps, which are unbuilt** — T3
([#1151](https://github.com/anthony-chaudhary/fak/issues/1151)), T5
([#1159](https://github.com/anthony-chaudhary/fak/issues/1159)), T6
([#1162](https://github.com/anthony-chaudhary/fak/issues/1162)) are all OPEN, so there is no
real per-turn meter, session ledger, or ablate-gate output to fold yet. The sibling
[`SELF-TAX-TREND.md`](../benchmarks/SELF-TAX-TREND.md) (T12,
[#1169](https://github.com/anthony-chaudhary/fak/issues/1169)) already *names* this verb as the
"named follow-on … not built here." This section upgrades it from *named* to *pinned* — the
same move §9 made for T13: fix the read-out shape, the metric-family schema, the golden-test
acceptance, and the double-count guard, so the eventual build is **wiring against a fixed
contract**, not a design decision deferred to build time.

### 10.1 What `fak perf` folds (four inputs, each with its current build state)

The fold is honest about provenance *and* about readiness — three of the four inputs are not
yet built, so today the verb would fold one real source (benchscore) plus the live rung
verdict/cost stream `fak rungstats` already reads. The contract names all four so the build is
additive as each dep lands.

| Input | Source (current) | What it contributes to the read-out | Build state |
|---|---|---|---|
| **T3** per-turn meter | `cmd/turntaxdemo` → first-class `fak turntax` | kernel-ns vs engine-ns; tokens *added* (transform/quarantine) vs *saved* (vDSO/radix) per turn, vs the T2 budget | OPEN ([#1151](https://github.com/anthony-chaudhary/fak/issues/1151)) — still a demo |
| **T5** session ledger | `internal/sessionobs` outcome-link rung (reusing `cadencereport`) | per-session **HELPED / WASH / HURT** verdict, provenance-labeled | OPEN ([#1159](https://github.com/anthony-chaudhary/fak/issues/1159)) — no `HELPED/WASH/HURT` in code yet |
| **T6** ablate gate | `cmd/fak/ablate` + `internal/turnbench` LeverFlip | fak-on vs fak-off delta on a frozen workload, signed, vs budget | OPEN ([#1162](https://github.com/anthony-chaudhary/fak/issues/1162)) — `ablate` exists as a one-off, not an always-on gate |
| `internal/benchscore` | `benchscore.Scan(root) → Report` (`fak.benchscore-report.v1`) | the frozen-workload **baseline rows** the deltas are measured against | **BUILT** |
| (spine) `internal/rungobs` + `fak rungstats` | live `fak_gateway_operation_duration_seconds{adjudicator-rung}` twin | the per-`(rung,kind,reason)` verdict **and** cost fold the whole plane extends | **BUILT** (verdict; cost fold is T1) |

### 10.2 The `fak perf` read-out shape (golden-testable)

A stable, deterministic report — schema `fak.perf-readout.v1` — so a frozen fixture round-trips
byte-identically and *that* is the golden test. It is `benchscore.Report` / `rungstats`
generalized to the whole plane: one object, five folds, one signed net line.

| Fold | From | Fields (per row) | Verdict |
|---|---|---|---|
| `rung_overhead` | T1 cost-fold over `rungobs` | `rung, kind, count, p50_ns, p99_ns, token_delta` | `OK` / `OVERHEAD_BUDGET_EXCEEDED` vs T2 |
| `turn_tax` | T3 `fak turntax` | `kernel_ns, engine_ns, tokens_added, tokens_saved, budget_ns` | within / over budget |
| `session_net` | T5 ledger | `session_id, helped_wash_hurt, tokens_added, tokens_saved, provenance` | HELPED / WASH / HURT |
| `ablate_delta` | T6 gate | `lever, on_value, off_value, delta, sign, budget` | within / over budget |
| `bench_baseline` | `benchscore` | the existing `Row` set (`workload, metric, value, baseline, speedup`) | accepted / negative / exploratory |
| **`net`** | §9 formula | `net_tokens` (signed) `= realized_reuse − mediation_tax`, with the `local ⊕ provider` split | improvement ⇔ `net_tokens > 0` |

Percentiles, never means — guardrails hurt the tail first (§4). The read-out reports p50/p99,
matching the live histogram twin. The illustrative shape (numbers from §9.3, **not** a
benchmark claim — the structure is the witness):

```
fak perf  (schema fak.perf-readout.v1)
  rung_overhead   adjudicator/decide   p50=362ns  p99=605ns  Δtok=0     OK
  turn_tax        turn#6   kernel=0.4ms engine=83ms  +0 / −9,400 tok    under budget
  session_net     sess-ab12  HELPED     +180 / −12,600 tok  WITNESSED⊕OBSERVED−MODELED
  ablate_delta    vdso       on=417 off=937  Δ=−520 tok (−)             under budget
  net             +12,420 tok   (local 9,400 [74.6%] ⊕ provider 3,200 [25.4%])  → improvement
```

### 10.3 The single net-true `/metrics` family

One family — proposed prefix `fak_self_tax_*` — folds the scattered counters §9 sums by hand
into a first-class read-back surface. Each member is **net-true-labeled** in its help text
(WITNESSED / OBSERVED / MODELED), and the `plane` label is the structural double-count guard:
a token is local *or* provider, never both (§9.1), so the two are reported as two series and
never collapsed.

| Member | Type | Folds (existing source) | Provenance |
|---|---|---|---|
| `fak_self_tax_net_tokens` | gauge (signed) | `realized_reuse − mediation_tax` (the headline) | derived |
| `fak_self_tax_realized_reuse_tokens_total{plane="local"}` | counter | `fak_gateway_kv_prefix_reused_tokens_total` | **WITNESSED** (in-kernel RadixAttention) |
| `fak_self_tax_realized_reuse_tokens_total{plane="provider"}` | counter | `fak_gateway_inference_cached_prompt_tokens_total` | **OBSERVED** (provider-relayed) |
| `fak_self_tax_mediation_tax_tokens_total` | counter | tokens mediation re-emits (transform/quarantine) | **MODELED** |
| `fak_self_tax_rung_overhead_seconds{rung,kind}` | histogram | `fak_gateway_operation_duration_seconds{adjudicator-rung}` | **WITNESSED** |
| `fak_self_tax_budget_breach_total{rung}` | counter | T2 `OVERHEAD_BUDGET_EXCEEDED` events | **WITNESSED** |

"Reads back" (the acceptance) = a test scrapes `/metrics`, parses the family, and re-derives
`net_tokens = (realized_reuse{local} + realized_reuse{provider}) − mediation_tax` — the §9.2
identity, now emitted directly rather than summed by a reader. `fak_self_tax_net_tokens` is the
denominated-in-prefill-tokens net of §9.2, never the vs-naive `1/(1−reuse)` re-prefill multiple
the #1066 honesty fence excludes.

### 10.4 Acceptance, and what blocks it today

- **AC1 — verb output golden-tested.** A frozen plane fixture round-trips to the §10.2 schema
  byte-identically (the `benchscore_test.go` / `rungstats` golden pattern).
- **AC2 — metric family reads back.** A scrape of the §10.3 family parses and re-derives the
  net per the §9.2 identity, with the `plane` split intact.
- **Blocked-on (the honest fence).** AC1/AC2 cannot be met until T3
  ([#1151](https://github.com/anthony-chaudhary/fak/issues/1151)), T5
  ([#1159](https://github.com/anthony-chaudhary/fak/issues/1159)), and T6
  ([#1162](https://github.com/anthony-chaudhary/fak/issues/1162)) land — there is no real
  per-turn meter, session ledger, or always-on ablate gate to fold. `benchscore`, `rungstats`,
  and the two reuse counters are built, so the fold is **additive wiring** as each dep arrives,
  not greenfield. **Lane for the build:** the verb is `cmd` (`cmd/fak/perf.go` + a pure
  `internal/perfreadout`), the family is `gateway`/`metrics` — **not** `docs`. This docs
  increment pins the contract only; it does not itself satisfy AC1/AC2.

### 10.5 Reproduce (the contract check, once the deps land)

```sh
fak perf --json | jq '.schema, .net.net_tokens, .net.split'        # AC1: golden round-trip
curl -s localhost:PORT/metrics | grep -E '^fak_self_tax_'          # AC2: family reads back
# net_tokens == realized_reuse{local}+realized_reuse{provider} − mediation_tax  (§9.2 identity)
```

Stated plainly, like the §9.4 reproduce: this is the **contract** the follow-on build verifies,
not a live witness today — `fak perf` and the `fak_self_tax_*` family do not exist until
T3/T5/T6 close. Pinning the shape here is the docs-lane increment of L5; the executable
golden-tested verb and the live metric family are the named follow-on in the `cmd` /
`gateway`/`metrics` lanes.

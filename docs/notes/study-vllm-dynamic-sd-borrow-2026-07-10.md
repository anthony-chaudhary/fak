---
title: "Borrow scout: vLLM Dynamic Speculative Decoding (#32374) → fak (2026-07-10)"
description: "Read the MERGED Dynamic-SD change in vLLM (PR #32374, squash 4ef4492e, 2026-06-14). A 4-axis solo pass over fak's LOAD-CONTROL plane concluded DIVERGENT/PRESENT-by-other-means with 0 leaves; two adversarial re-runs (find→refute+confirm→judge, 75 agents over 18 axes A-R) OVERTURNED that to 4 fileable leaves — all hand-verified and FILED: J #4269 (dispatch preflight CapTerms.Map() drops contraction_cap on the tick it binds, internal/dispatchtick/preflight.go:490-500); L #4270 (abi reserved-range disjointness is promised in registry.go docstring but Register* only catches exact-integer dups, internal/abi/registry.go); D #4271 (over-ceiling temperature/top_p forwarded raw where max_tokens is saturated, internal/agent/stream.go — proposal); O #4272 (anthropic/gemini adapters silently drop threaded structured-decode, internal/agent/adapters.go — proposal). 16 axes correctly non-fileable, dedups gh-verified (#4036/#4038 CLOSED; P already declined on #3624 with 'do not re-run'). See the ADDENDUM."
---

# Borrow scout: vLLM Dynamic Speculative Decoding (#32374) → fak (2026-07-10)

Study of vLLM's **Dynamic Speculative Decoding** — PR **#32374** "[V1][Spec Decode] Add Dynamic SD",
merged as squash `4ef4492e9b7a5a7ba295da783d456d45db5eb9d6` (2026-06-14), read read-only in scratch, never
the fak tree. Borrow mode is **INSPIRE** (reimplement an idea in Go), never vendoring — vLLM is Apache-2.0,
so the anchors below are provenance, not a license to copy.

This closes the same loop as the two sibling passes filed the same day
([lmnr](study-lmnr-borrow-scout-2026-07-10.md), [kvcached](BORROW-KVCACHED-STUDY-2026-07-10.md)), on a
different source. Unlike those two it files **no leaf** — and the bulk of this note is *why that is the
honest verdict*, so a future scout does not re-derive it.

> **⚠️ SUPERSEDED IN PART — see the ADDENDUM below.** The 4-axis solo pass recorded here concluded
> "0 leaves." Two adversarial re-runs (find → refute + confirm → judge; 46 + 29 agents) overturned that:
> **4 fileable leaves survived refutation**, all on axes the 4-axis pass never opened — a provider-wire
> sampling seam (**D**), a dispatch-preflight observability seam (**J**), a frozen-ABI reserved-range
> partition invariant (**L**), and an adapter structured-decode capability seam (**O**). None is in the
> load-control plane this pass studied. The original body below is kept verbatim as the historical record;
> the ADDENDUM is the current verdict.

## ADDENDUM (2026-07-10, adversarial re-run) — verdict changed: 4 leaves

The solo pass above distilled the merge into **4** axes (T1–T4), all in fak's **load-control plane**, and
filed 0. An adversarial re-run widened the lens to **11 granular axes (A–K)** and ran each through
independent **witness → refuter + confirmer → judge** stages, then a synthesis + a completeness critic that
re-read the actual diff. That critic flagged **7 further axes (L–R)** the 11 missed, witnessed in a **round 2**
(same find → refute + confirm → judge, 29 agents). Combined result: **4 leaves survive refutation** (all four
re-verified by hand against source) — D and J from round 1, **L and O from round 2** — with **16 axes correctly
non-fileable**. Two round-2 non-survivors are notable honesty checks: **P** (guard managed-cache ON-branch
over-claim) was **already declined one pass ago on #3624 with an explicit "do not re-run,"** and the workflow
correctly refused to re-file it; **Q** points at a dormant metric bucket owned by tracked epic #4102/#4106.

### Leaves to file (survived adversarial verify + hand re-verification)

- **D — Saturate over-ceiling `temperature`/`top_p` in the outbound provider builder, like `max_tokens`
  is, instead of forwarding them raw to a provider 400.** Verified: `internal/agent/stream.go:385-386`
  saturates `max_tokens` to `p.MaxTokensCap`, but `:388-391`→`:420` ships `temperature` and `:424` ships
  `top_p` raw to `adapter.MarshalRequest` (`temperature` is omitted *only* on the `forceResponsesStream`
  path, `:421`). An over-ceiling `temperature` (>2.0 OpenAI-compat) or `top_p` produces the exact
  provider-400 class `MaxTokensCap` exists to prevent. **Honest framing:** there is no `p.TemperatureCap`
  to saturate *to*, and clamping `temperature` mutates requested sampling semantics — so the leaf is
  *"clamp `top_p` (safe, ≤1.0) + document/omit `temperature`, OR document the deliberate asymmetry at
  `chat.go:784`,"* not blind saturation. Not a dup of **#4069** (discrete effort-tier enum — different
  operand/seam) or **#4036/#4038/#3368** (worker-concurrency, different subsystem); open-issue search for
  temperature/top_p/sampling clamp returns no match. Domain: dead-center headless gateway. *(task tracked)*
- **J — Emit `contraction_cap` value in dispatch preflight `CapTerms.Map()` so the chosen drain target is
  observable per decision, not just its reason.** Verified: `CapTerms.Map()`
  (`internal/dispatchtick/preflight.go:490-500`) emits `configured/lease/host/seat/worker_floor/
  effective_cap` + `limiting` but **drops `contraction_cap`** — while the field (`:198-202`) is set
  per-decision (`:395-396`) and can become `Limiting="contraction"` (`:406-410`). So on a
  contraction-bound tick the operator JSON (`out := res.Map()`, `cmd/fak/dispatch_tick_preflight.go:70`)
  shows *which* term bound but not *what value*. `contraction_cap` has **exactly one** repo-wide site (the
  struct tag). **#4038** (which introduced the field) is **CLOSED** — it created the omission and cannot
  own it; no open dup. Fix = one `Map()` line + a binding test mirroring `preflight_contraction_test.go`.
  Domain: headless dispatch admission controller — fak's own adaptive concurrency knob. *(task tracked)*
- **L — Assert the abi reserved-range disjoint-partition invariant the registry docstring already
  promises.** Verified: `internal/abi/registry.go:39-44` declares *"the static disjointness contract …
  Two leaves with disjoint blocks cannot collide; Register\* enforces it at init() time,"* but
  `RegisterOp` (`:500-508`) and `RegisterVerdictKind` (`:512-523`) panic **only on an exact-integer
  duplicate** — never on block-to-block overlap, nor on a `Code()`/kind landing outside or straddling its
  declared `Range`. `inRange` (`:73`) is dead in production (only `pkg/abi/abi_test.go` references it). The
  blocks are disjoint *today*, so this is a **promised-but-unverified invariant**: a future additive block
  (e.g. `Range{80,112}` straddling `OpsSpec{64,96}`+`OpsAsync{96,128}`) stays silent until two subsystems
  pick the same integer. Leaf = a `TestReservedRangesPartitionDisjoint` (sort each family's blocks by `Lo`,
  refuse `Lo[i] < Hi[i-1]` + gap check) + optionally gate `Register*` to require the claimed number lie
  inside a declared block. Lane clear of the DSD dispatch-cap family (none touch `internal/abi`); not
  #4069. **Low-risk (adds a test/assertion).** Domain: frozen-kernel ABI invariant. *(task tracked)*
- **O — Surface the silent structured-decode drop (fail-fast OR map-to-native) for providers without a
  native field, instead of dropping `response_format`/`logit_bias`/`guided_decode` at the adapter seam.**
  Verified: `internal/agent/stream.go:416-432` threads `ResponseFormat`/`LogitBias`/`ExtraBody` into
  `MarshalRequest` with no per-provider capability check; `anthropicAdapter.MarshalRequest`
  (`adapters.go:774-812`) builds its body from Model/MaxTokens/Temperature/TopP/TopK/Stop/Messages/Tools and
  **never reads** those fields (gemini `:1028-1069` same, native `responseSchema` un-wired). A caller
  expecting schema-constrained JSON silently gets free-form. **Honest framing — design-discussion, not a
  bug:** fak does this *deliberately* — #560 ships the carrier and `TestPerRequestStructuredDecodeForwarding`
  (`adapters_test.go:1734`) **test-pins** silent-omit for non-native providers. So the leaf proposes a typed
  per-consumer capability assert (fail-fast, or map-to-native e.g. Anthropic tool-use), an **increment on
  #560's seam** that would revisit its pinned behavior — not a dup, not a clean bug fix. Domain: core
  model-call adapter surface. *(task tracked)*

**Two tiers.** **J and L are clean, low-risk, clearly-correct** (an observability one-liner + a frozen-ABI
invariant test). **D and O are real but design-discussion** — each touches deliberate behavior (D: no
`TemperatureCap` exists and clamping mutates sampling semantics; O: #560 test-pins the silent omit), so each
should be filed as a proposal, not asserted as a bug. All four were **filed** (operator-approved, 2026-07-10):
**J → [#4269](https://github.com/anthony-chaudhary/fak/issues/4269)**,
**L → [#4270](https://github.com/anthony-chaudhary/fak/issues/4270)**,
**D → [#4271](https://github.com/anthony-chaudhary/fak/issues/4271)** (proposal),
**O → [#4272](https://github.com/anthony-chaudhary/fak/issues/4272)** (proposal).

### Scorecard — 11 axes (A–K)

| Axis | Verdict | Fileable | One-line |
|---|---|---|---|
| A dense-total-lookup | DIVERGENT | no | No integer-range-keyed dense hot-path table; need met total-by-other-means (`effortcost.Of` anchor + scalar setpoints). Already recorded as T1. |
| B totality-floor-refuse | DIVERGENT | no | `RouterConfig` has zero json/toml tags, never unmarshalled; floorIndex clamp + unbounded top = total-by-construction. No operator-YAML surface to refuse. |
| C carry-forward-gapfill | DIVERGENT | no | `effortcost.Of` totalizes via one CONSTANT anchor (opposite of position-dependent hold); no sparse band schedule to densify. #4069 owns nearest. |
| **D clamp-saturate-ceiling** | **PARTIAL** | **YES** | `max_tokens` saturated; sibling `temperature`/`top_p` forwarded raw → provider-400 asymmetry. **Leaf.** |
| E graded-depth-by-load | PARTIAL | no | fak grades load on the POPULATION plane (#4036/#4038/#3368); per-unit depth knobs binary/load-blind. Deferred twice under #809 w/ unmet cost-sensitivity precondition. |
| F fallback-static-when-signal-absent | PRESENT | no | `EvaluatePreflight` folds each live signal only under a present-and-positive guard → static `MaxWorkers` on absence. Owned by #3368. |
| G knob-as-callparam | PRESENT | no | `governor.go:120 Admit(loop, policy, now)` takes the whole Policy bundle per-call; load-blind `MaxConcurrent` read tracked by #4199 / epic #1333. |
| H async-carry-prevvalue | DIVERGENT | no | `speculate.go:393-395` carries produce-time epoch into a per-epoch provisional buffer (txn identity, not depth knob); native scheduler is synchronous lockstep. Owned by #809. |
| I compile-once-index-hot | PRESENT | no | validate-once-fail-loud + O(1)/linear decision path is pervasive; the one leak (`routing.go:358` 3-tier per-call sort) is immaterial; dense-array-by-batch-size inapplicable to sparse PromptTokens key. |
| **J adaptive-knob-observability** | **PARTIAL** | **YES** | `CapTerms.Map()` drops `contraction_cap` on the very tick it binds. **Leaf.** |
| K config-rangelist-plus-predicate | PRESENT | no | presence-derived enable predicate (zero-value-disables + map-membership override, no enable-bool) ships in `governor.go`, no-drift-tested; `speculate.go` range-list carrier gap is config SHAPE, deferred under #809. |

Every dedup claim was verified via `gh`: **#4036/#4038 CLOSED**; **#3368/#1333/#4069/#3020/#809/#4199 OPEN**.
Neither survivor is a dup of #4069 (discrete effort-tier enum), the #4036/#4038/#3368/#1333 worker-concurrency
plane, or #3020 (SD eval) — D operates on provider-wire sampling params, J on the preflight observability Map.

### Completeness critic — 7 further axes (L–R), round 2 in progress

Reading the real diff, the critic found 7 transferable sub-techniques the 11-axis spine does not name, of
which **M, N, O, P** are the highest-transfer for a Go kernel:

- **L** — canonicalize unordered operator interval config, THEN enforce a disjoint-partition invariant (`utils.py:59-68`).
- **M** — derive the effective dimension by MEASURING the produced artifact, not trusting the requested knob (`gpu_model_runner.py:4711`). *The DOS "verify-from-artifact, not self-report" discipline expressed numerically — strongest miss.*
- **N** — zero-knob returns typed-empty but STILL runs the mandatory state-sync side effect (`llm_base_proposer.py:526-533`).
- **O** — each consumer fail-fast asserts a threaded knob vs its own static capability; `<=` vs `==` encodes true dynamic support (`ngram_proposer.py:145` vs `medusa.py:49`).
- **P** — boot-time auto-downgrade of an incompatible optimization to the safe path, warn-once / per-backend deny-list (`vllm.py:767-782`).
- **Q** — per-bucket DENOMINATOR instrumentation so a success-rate-per-bucket (knob-tuning feedback) is derivable (`metrics.py:31,38,49`).
- **R** — clamp operator-declared range KEY-DOMAIN to runtime capacity (distinct from D's VALUE clamp) (`utils.py:111,121`).

**Round-2 verdict (29 agents, find → refute + confirm → judge):** 2 of 7 fileable.

| Axis | Verdict | Fileable | Disposition |
|---|---|---|---|
| **L canonicalize-then-partition** | **ABSENT** | **YES** | Leaf above — abi reserved-range disjointness promised in docstring, never enforced by `Register*`. |
| M realized-dim-from-artifact | PRESENT | no | Ships at `fleetaccounts/wave.go` (`granted := len(lanes)` sizes the wave from the produced slice, never `req.Count`) + warmsplice `RestoredPositions`/`SerializeSpan(0,kv.Len())`. Residue = a one-line field-propagation fix (`warmsplice.go:203`) with no production consumer — not a borrow. |
| N zero-knob-preserves-sideeffect | DIVERGENT | no | Mandatory progress never skipped on zero-knob, but via an **inverted/split** decomposition (warmsplice syncs only on the warm path; cold re-prefill lives in the driving loop). K=0 guardrail already homed on **#3078** (family #3197/#2236). |
| **O per-consumer-capability-assert** | **DIVERGENT** | **YES** | Leaf above — adapters silently drop threaded structured-decode; a fail-fast/map-to-native increment on shipped **#560** (which test-pins the omit). |
| P incompatible-mode-auto-downgrade | PARTIAL | no | Present on the AUTO branch of `resolveGuardManagedCache`, absent on the pinned ON branch (`guard_managed_cache.go:68-69`) — **but this exact P4 seam was confirmed a real bug and declined as a #3624 enrichment one pass ago, with an explicit "do not re-run."** Owned by OPEN #3892/#3624/#2193 (epic #3569). |
| Q per-bucket-rate-instrumentation | PARTIAL | no | Denominator-beside-numerator idiom ships for reactive/global levers (`gateway/metrics_render.go` `oom_retry_total`); the one adaptive knob's `patternStat{hits,trials}` (`speculate.go:109-112`) never reaches a metrics path **and is dormant** (`WithSpeculator` has zero live call sites). Belongs under tracked epic **#4102/#4106**. |
| R domain-clamp-to-runtime-capacity | DIVERGENT | no | Parent need (declared max reconciled to true capacity, refuse loud) ships a different way — functional value min-clamp (`preflight.go:264-283`) + `KVPreemptor.Admit` "exceeds-capacity" (`preemption.go:230-241`); fak materializes no dense per-key table for R's index-write-safety form. Owned by **#3368/#4038**.

All round-2 dedups were verified against real in-repo issue references; two honest caveats flagged (#2193 and
#4106 are not locally grep-able, but their load-bearing siblings/parents #3624/#3569/#3892 and #4102 confirm
the dispositions).

---

_Original 4-axis solo pass follows verbatim (historical record):_

## Method & honesty boundary

- **Read the merged diff, not the feature pitch.** The 23-file squash was read at its anchors (below), plus
  its 218-line test (`tests/v1/spec_decode/test_dynamic_sd.py`) which pins the exact semantics — carry-forward
  gap-fill, first-range-must-start-at-1, clamp-to-runtime-max, and `propose()` taking depth as a per-call
  parameter. Semantics are cited from the test, not inferred from names.
- **Witnessed each distilled technique against fak on-axis** via `fak_feature_query` + raw grep/read, with a
  false-ABSENT guard (grep fak's own prior borrow of the same family) and a false-PRESENT guard (confirm the
  cited fak seam covers the *axis*, not just the capability name). Seams read: `internal/loopmgr/governor.go`,
  `internal/dispatchtick/preflight_usagecap.go` + `preflight_setpoint.go` + `preflight_contraction.go`,
  `internal/modelroute/effortcost.go`, `internal/fleetaccounts/apextier.go`, `internal/spec` (via the shipped
  "single-pass batched + tree-attention verify" claim), and the dispatch **wave** path in `cmd/fak/dispatch_tick.go`.
- **No leaf is filed on a vibe.** The bar this note holds itself to is the kvcached study's own: *"filing a
  test for a non-existent seam is noise"* (its parked axis 12). The one live lead here proposes a **new control
  coupling**, not the completion of an existing graded mechanism, and sits adjacent to already-tracked issues —
  so it is recorded as a deferred lead, not filed.

## What the merge actually does (three separable techniques)

vLLM's static SD proposes a **fixed** `k` draft tokens every step. Dynamic SD makes `k` a function of the
current **batch size**, because speculative work has cost ∝ batch × depth: past a critical concurrency the
verify/redo cost dominates and a fixed depth becomes net-negative. Three separable, repo-agnostic techniques:

- **T1 — validated sparse-range → dense-total O(1) lookup.** The operator authors sparse inclusive
  `(lo, hi, k)` bands (`num_speculative_tokens_per_batch_size`, `config/speculative.py:161`). A builder
  (`spec_decode/dynamic/utils.py:77`) *compiles* them once into a dense array indexed by batch size, so the
  hot path is a single array index (`core/sched/scheduler.py:1008` — `self.dynamic_sd_lookup[batch_size]`).
  The robustness core is **totality**: the first band *must* start at 1 (`utils.py:70` — "so every runtime
  batch size is defined") else it refuses at load time; **carry-forward** gap-fill (`utils.py:106`) covers
  interior gaps; a **clamp-to-runtime-max** (`utils.py:113/123/143`, `min(k, vllm_max)`) covers the tail. The
  lookup is total by construction — no runtime key is ever undefined.
- **T2 — grade optimistic depth DOWN as aggregate load rises.** The control *law*: reduce per-request draft
  depth as batch size climbs, raise it as batch drains, and fall back to the static `k` when the batch is
  empty (`scheduler.py:1008` guards `len(num_scheduled_tokens) > 0`). Graded degradation, not a binary cutoff.
- **T3 — depth as a per-invocation caller parameter, not a worker field.** `ngram_proposer.propose()`
  (`spec_decode/ngram_proposer.py`) now takes `num_speculative_tokens` per call (asserted `<= self.k`, the
  ceiling) instead of reading a fixed instance `self.k`, so the scheduler varies depth per step without
  rebuilding the proposer. Plus the async-scheduler carry hazard (`async_scheduler.py`): when depth changes
  between produce/consume steps, the *prior* step's spec-count must be tracked for correct indexing.

## Axis-by-axis witness against fak

| # | Axis (vLLM technique) | Verdict | fak seam | Disposition |
|---|---|---|---|---|
| 1 | T1: validated sparse-range → **dense total** O(1) lookup (first-covers-floor totality + carry-forward + clamp) | **DIVERGENT** (total by other means) | `modelroute/effortcost.go:61` `EffortMultiplier.Of` (label→value map, total via unknown→1.0 anchor); `dispatchtick/preflight_setpoint.go` (scalar setpoint). No **integer-range-keyed** hot-path table exists to retrofit. | recorded |
| 2 | T2: grade optimistic **depth** down as aggregate load rises (graded, not binary) | **PARTIAL → deferred** | `loopmgr/governor.go:135` `Admit` grades nothing (binary `MaxConcurrent` refuse); `preflight_setpoint.go`/`preflight_contraction.go` grade **population** (wave size), not per-worker depth; `effortcost.go` grades **cost/effort**, not by load. | deferred lead (below) |
| 3 | T3: depth as per-invocation **caller param** (+ async knob-changed-between-steps carry) | **PRESENT / DIVERGENT** | `loopmgr/governor.go:120` `Admit(loop, policy, now)` already takes the knob per call; the async-slice carry hazard is inference-pipeline-specific — no fak analog. | recorded |
| 4 | *Literal* analog: spec-decode **K by batch size** | **ABSENT but dormant** | `internal/spec` `SpeculativeGreedy`/`SpeculativeTree` verify a **fixed** kk-token draft; the path is `FAK_POLYMODEL`-gated, CPU-synthetic, off-mainline — **not** a batched serving engine, so "batch size" is a dormant variable. #3020 (DeepSeek MTP/SD eval) names `accepted_token_ratio` but no by-batch-size K. | recorded |

## Tally

4 axes: **1 DIVERGENT**, **1 PARTIAL-deferred**, **1 PRESENT/DIVERGENT**, **1 ABSENT-but-dormant**.
**0 leaves filed.** 1 strong deferred lead recorded.

Why no leaf: fak's load-control plane deliberately occupies this same space with **scalar min-folds +
graded-population setpoints + a default-anchor effort ladder**, and every crisp borrow either duplicates a
tracked issue or proposes a speculative new coupling:

- **T1 has no home.** fak has no piecewise integer-**range** config on a hot path. Its keyed controls are
  label→value maps already total by a default anchor (`effortcost.Of` → 1.0), or scalar setpoints/min-folds —
  both total by construction, differently. Retrofitting vLLM's band-compiler would mean *inventing* a config
  surface fak does not have. Tradeoff stated: operator-authored bands buy explicit per-band control at the cost
  of a validate/compile step; fak's default-anchor maps buy simplicity at the cost of not expressing
  "different value per load band" — a cost fak has not needed to pay.
- **T2's population half is PRESENT and tracked.** fak grades *how many workers* by load — `preflight_setpoint`
  (**#4036**, externally-written setpoint, grow-now/shrink-on-drain), `preflight_contraction` (**#4038**, floor
  the shrinking dimension), **#3368** (predictive floor + reactive clamp), and the governor's own note that a
  dispatch loop "can be given a **derived ceiling** instead of a fixed cap" (**#1333**). The graded-degradation
  *family* is likewise already being borrowed — **#4069** saturates an unsupported effort tier to the nearest
  supported one instead of refusing. So the general law is not missing.

## Strong deferred lead (recorded, not filed)

**Couple a per-worker _optimistic-depth_ knob to measured aggregate host load — the "grade per-unit depth, not
just population" flavor fak's setpoint family (#4036/#4038/#3368) does not cover.** The concrete knob that
recently landed is **warm-KV staging / managed-cache posture** across relaunch (`feat(session): stage and
restore warm KV across relaunch`, #4133; `fix(resume): front resumed child with managed-cache posture`, #3779).
Today that posture is a load-**blind** binary (auto/on/off); vLLM's worldview says its *depth* (how much warm KV
to stage per worker) should degrade gracefully as fleet concurrency climbs toward the host-RAM-critical point
that [SAFE-CONCURRENCY-HEADROOM-2026-07-01](SAFE-CONCURRENCY-HEADROOM-2026-07-01.md) documents.

Not filed because it **proposes a new coupling** rather than completing an existing graded mechanism, and is
adjacent to the tracked derived-ceiling family (#1333/#4036/#3368) — filing now risks either a dup or noise
(cf. kvcached study's parked axis 12: "filing a test for a non-existent seam is noise"). The honest next step
is to **witness the staging knob's actual cost-sensitivity to fleet load** before filing; if the coupling is
real, its natural home is the #4036/#4038 dispatch-capacity work, extended from population to per-worker depth.

## Dedup notes

- **#4036 / #4038 / #3368 / #1333 / #1653** already own graded/derived **concurrency (population)** control —
  T2's population half. Cross-referenced, not duped.
- **#4069** (saturate effort tier instead of refuse) already owns the **graded-degrade-not-refuse** family.
- **#3020** (DeepSeek V4 MTP / speculative-decoding eval) owns fak's SD-quality axis (`accepted_token_ratio`,
  `quality_parity`, `CompareSpeedup` refusing an unwitnessed speedup) — the natural home if a by-batch-size K
  ever becomes live, but that requires a batched serving path fak's `internal/spec` is not.

_Not committed: branch is `main` and the working tree carried extensive unrelated uncommitted changes at study
time. This note is on disk as an untracked artifact, matching the two sibling 2026-07-10 borrow notes._

---
title: "Generation Map For The Hermes-Inspired Observability Track"
description: "Classifies #2864 and #2930 (observer/middleware hook contracts) as gen/next, records that their enforcement contract already shipped unwired under the closed #2904 — the same mechanism filed under three separate epics — and names the residuals plus the Likely-files trap that mis-routes this track."
---

# Generation Map For The Hermes-Inspired Observability Track

This note records the generation classification for
[#2864](https://github.com/anthony-chaudhary/fak/issues/2864) (observer +
middleware hook contracts that adjudicate and witness every hook, fail-closed vs
Hermes fail-open), filed under epic
[#2834](https://github.com/anthony-chaudhary/fak/issues/2834) (Track H —
observability contracts).

It is the Track H companion to
[`GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md`](GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md),
written because that map is cache/context-scoped and has no row this track
belongs in.

Snapshot source on 2026-07-19:

- `gh issue list --label hermes-inspired --state all`
- `gh issue view 2834` / `2864` / `2863` / `2904` / `2871`
- `go test ./internal/metrics/ -run 'Throwing|Panicking|FailClosed|FailOpen|Chain' -count=1`
- a repo-wide sweep for callers of the seam (`metrics\.(Apply|Chain|Middleware|Observer|FailMode)`)

Generation remains orthogonal to priority, shared trunk, and runtime feature
gates, as [`docs/generation.md`](../generation.md) states. A `gen/*` label only
says which horizon owns the assumption and what evidence moves it.

## Headline: #2864's Contract Already Shipped, Under A Different Epic — But Unwired

The decisive finding, and the reason this note exists. **#2864 substantially
duplicates the closed [#2904](https://github.com/anthony-chaudhary/fak/issues/2904)** —
"feat(observability): adjudicated middleware/observer contract — Hermes' hooks are
fail-open" — which landed under the *parallel* epic
[#2871](https://github.com/anthony-chaudhary/fak/issues/2871), not under #2834.

Two epics decomposed the same Hermes source material independently, so the same
mechanism was filed twice under two parents. #2904 built it; #2864 is still open
describing it as unbuilt.

The shipped seam is `internal/metrics`:

| #2864 scope item | State | Where |
|---|---|---|
| Observer + middleware hook contracts (mirroring `hermes.observer.v1` / `hermes.middleware.v1`) | **shipped** | `internal/metrics/middleware_contract.go` — `Observer` and `Middleware` interfaces |
| Per-hook fail-open/fail-closed declaration | **shipped** | `FailMode`, whose **zero value is `FailClosed`**, so a middleware that forgets to declare its mode denies-on-error by construction |
| A crashing guard-middleware denies rather than skips | **shipped** | `Apply` + `safeHandle` recover a panic into an error; an errored fail-closed link returns `VerdictDeny`/`ReasonDefaultDeny` |
| Fail-closed survives composition (`next_call` nesting) | **shipped, beyond #2864's ask** | `middleware_chain.go` — `Chain` folds by `abi.FoldRank`, so a throwing enforcer denies the whole chain order-independently |
| **Witnessed hook-invocation record (hook id, verdict, latency)** | **NOT shipped** | residual 1 — no such record in either file |
| **Enforcement actually on the `llm_request`/`tool_request` path** | **NOT shipped** | residual 2 — the seam has **zero callers**; see below |

### Residual 2 is the load-bearing one: the seam is inert

A repo-wide sweep for `metrics.Apply` / `metrics.Chain` / `metrics.Middleware`
returns **no matches**. The only reference to the seam outside its own package is
`cmd/extseamsdemo/main.go`, and that is a *catalog row* describing the seam's
trust and failure posture — not an invocation.

So the guarantee is real but **library-local**. Nothing in `internal/gateway`,
`internal/adjudicator`, or `internal/kernel` routes a call through `Apply` or
`Chain`. #2864's done condition — "a middleware declared as a guard that throws
**during `llm_request`/`tool_request`** results in the call being denied" — is
therefore *not* satisfied in the running system. It is satisfied only as a
proven property of a package nothing calls.

This distinction is easy to miss and is the single most useful thing in this
note: reading `middleware_contract.go` alone makes #2864 look finished.

### The acceptance gate #2864 calls unknown is already answered

#2864's contract overlay says verbatim `unknown -- needs operator input: what is
the exact test name/path for the crashing-guard-middleware-denies test?` That
question has an answer in the tree today:

- `internal/metrics/middleware_contract_test.go` — `TestThrowingEnforcerDenies`,
  `TestPanickingEnforcerDenies`, `TestFailClosedIsZeroValue`, `TestFailOpenObserverDefers`
- `internal/metrics/middleware_chain_test.go` — `TestChainFailClosedEnforcerDeniesWholeChain`,
  `TestChainDenyIsOrderIndependent`

Verified green on 2026-07-19:

```text
go test ./internal/metrics/ -run 'Throwing|Panicking|FailClosed|FailOpen|Chain' -count=1
ok  github.com/anthony-chaudhary/fak/internal/metrics  0.005s   (11 tests / 3 subtests PASS)
```

The issue's named witness exists and passes. No operator input is needed to
unblock the gate; it was answered by #2904 and never propagated back to #2864.
What those tests do *not* prove is residual 2 — they drive `Apply`/`Chain`
directly, so they would stay green even though no production path calls either.

## Active Classification

This is a grooming map, not an automatic label mutation. Labels were applied only
where the evidence was actually read — #2864 here. The track's other unlabeled
children stay unlabeled pending their own evidence pass.

**#2864 classified `gen/next`, milestone `Generation G1 - Next Gen`.** It carried
no `generation` label, no stream label, and no milestone, so it was invisible to
every `generation-*` dispatch view.

Why `gen/next`: the stream is "near-term foundation that should be runnable by
agents soon, but still needs a gate, dogfood run, schema, or **default-exposure
proof**." That is residual 2 exactly. The foundation is built and green; what is
missing is exposure — a live call site plus a dogfood run showing a throwing
guard denies a real `llm_request`/`tool_request`, not a fixture.

Why not `gen/now`: `gen/now` requires a clear *current-product* witness. An
uncalled seam does not improve the current product — a guard that nothing invokes
denies nothing in production. Labeling it `gen/now` because the package is green
would be the current-work-laundering anti-pattern, crediting a library guarantee
as a shipped protection.

Why not `gen/second-next`: no simulation, compatibility policy, or
cross-generation dependency management is implicated. The additive-only
frozen-ABI promise in
[`generation-abi-compatibility-policy.md`](../generation-abi-compatibility-policy.md)
already covers adding a record and a call site.

Sibling precedent agrees: the classified `hermes-inspired` set skews `gen/next`
(#2872-#2875, #2877, #2884-#2886, #2888-#2889, #2895, #2902), including the
nearest-shaped sibling #2863.

## The Mis-Routed Likely-Files Trap

Worth reading before picking up any issue in this track; it costs a lane lease
and a wasted dispatch to rediscover.

#2864's contract overlay lists its `Likely files` as `docs/observability/README.md`
and `docs/middleware/README.md`. **Both are Hermes' doc paths, not fak's**,
mechanically copied from the issue's own description of the Hermes mechanism:

- `docs/middleware/README.md` does not exist in this repo at all.
- `docs/observability/README.md` does exist, but it is the *operator route for a
  running `fak serve` gateway* — healthz, metrics, `/debug/vars`, trace-id
  correlation. It has no plugin surface in it.

Because the file-tree router derives the lane from those paths, #2864 routes to
the **`docs` lane** while the code it concerns lives in `internal/metrics` (a
code lane). A docs-lane worker cannot land either residual. Expect the same trap
on sibling overlays: the generator copied Hermes' documentation paths into
`Likely files` wherever the issue body cited them.

## Promotion Evidence

What retires the residuals and closes #2864:

- **A live call site.** `Chain` invoked on the real tool-request path so a
  guard-declared middleware's failure denies an actual call. This is the
  default-exposure proof that moves the issue toward `gen/now`, and it is the
  work #2864 still genuinely owns after #2904.
- **A witnessed invocation record** threaded through `Apply` carrying **hook
  identity, verdict, and measured latency per invocation** — not aggregate
  counters, which the issue explicitly rules out in its confusion-risks section.
- **A test that the record survives the panic path.** A recovered panic must
  still emit a record, or the witness is exactly as silently-disableable as the
  swallowed exception the issue exists to prevent.
- **A dogfood run** on a real `fak serve` session, since the existing unit tests
  cannot distinguish "wired" from "inert".

## Demotion, Parking, Or Retirement Evidence

- **Re-scope rather than retire.** Everything #2864 asks for *except* the
  invocation record and the wiring is shipped and green under #2904. The honest
  move is to narrow #2864's body to those two residuals and cross-link #2904,
  rather than leave it open implying the whole contract is unbuilt. Closing it
  outright would be wrong — residual 2 is a real, unshipped safety gap.
- **#2865 absorbs it.** #2865 (fail-closed audit enumerating every fak
  guard/hook) covers the general fail-closed adjudication and witness path. If it
  lands the per-hook witness record and the wiring first, #2864 retires into it.
- **The seam is deliberately inert.** If an operator confirms
  `internal/metrics`'s middleware seam is intended as an *offered extension
  point* for third-party compiled middleware rather than a path fak itself routes
  through, then residual 2 is not a gap and #2864 collapses to residual 1 alone —
  a much smaller issue, arguably `gen/now`.
- **Per-invocation latency proves too costly.** The record sits on the
  per-tool-call security path. If measuring it trips the
  `GATE_LATENCY_REGRESSION` budget, the record demotes to sampled or aggregate,
  retiring #2864's done condition as written and needing a re-scoped issue rather
  than a quiet weakening.

## Invalidating Assumption

The classification rests on residual 2 — that the seam is genuinely uncalled. That
was checked by a repo-wide grep for `metrics.(Apply|Chain|Middleware|Observer|
FailMode)` returning no matches, plus a targeted scan of `internal/gateway`,
`internal/adjudicator`, and `internal/kernel`. **A grep cannot see an indirect
call**, so the named disproof — an adapter satisfying the interface under another
name — was run as a signature sweep across `internal/` and `cmd/`:

```text
grep -rn "abi.ToolCall) (abi.Verdict, error)" --include=*.go internal/ cmd/
internal/metrics/middleware_contract.go:65      (the interface declaration)
internal/metrics/middleware_contract_test.go:20 (a test stub)
```

Two hits, both inside the seam's own package: the declaration and a test double.
No production type implements `Middleware`. The disproof was attempted and
failed to disprove, which is why residual 2 is stated as a finding rather than a
suspicion.

The assumption that remains untested: a caller could satisfy the contract through
a *structurally different* signature — an adapter that wraps `Apply` behind its
own verdict type rather than implementing `Middleware` directly. Nothing in the
sweep suggests one exists, but a signature grep cannot rule it out.

A second invalidator: this note assumes #2904's seam is the one #2864 means. Both
cite the same Hermes contracts and the same fail-open critique, so the
identification is strong — but if Track H intends a *separate*, plugin-facing ABI
distinct from the in-process trusted-compiled seam that `extseamsdemo` catalogues
as `trust: trusted-compiled`, then the duplicate finding weakens to "adjacent
prior art" and #2864 is unbuilt foundation rather than a re-scope.

## #2930: A Third Filing Of The Same Mechanism, Under A Third Epic

[#2930](https://github.com/anthony-chaudhary/fak/issues/2930) ("observer +
middleware plugin contracts — fail-open telemetry taps and adjudicating
middleware around every llm/tool call") descends from epic
[#2908](https://github.com/anthony-chaudhary/fak/issues/2908), a *third* parent
distinct from both #2834 (which owns #2864) and #2871 (which owned #2904).

So the count in this note's headline is not two but **three**: #2904 built the
mechanism, #2864 describes it as unbuilt, and #2930 describes it as unbuilt
again — each under a different epic that decomposed the same Hermes source
material without seeing the others. Re-verified on 2026-07-19 against #2930's
three acceptance clauses:

| #2930 acceptance clause | State | Where |
|---|---|---|
| A versioned observer + middleware interface over the existing seam | **interfaces shipped, the version is NOT** | `internal/metrics/middleware_contract.go` declares `Observer` and `Middleware`, but the only `v1` tokens in the file name *Hermes'* contracts (`hermes.observer.v1`, `hermes.middleware.v1`) in comments. fak's own seam carries no version token |
| A fail-open guarantee for observers | **shipped** | `Observer` carries no `FailMode` (fail-open by definition); `Apply`'s `FailOpen` branch returns `VerdictDefer`, and `safeHandle` recovers a panic so a broken observer cannot block a call |
| A journaled decision for middleware | **NOT shipped** | no invocation record in `middleware_contract.go` or `middleware_chain.go` — identical to #2864's residual 1 |
| (implied by "around every llm/tool call") enforcement on the real path | **NOT shipped** | residual 2, re-confirmed: `metrics.(Apply\|Chain\|Middleware\|Observer\|FailMode)` returns no matches repo-wide |

### #2930 is not a pure duplicate — it owns one residual #2864 does not

The **versioning** ask is #2930's alone. #2864 asks for adjudication and a
witness record; it never asks the seam to declare a version. #2930's acceptance
opens with "a *versioned* observer + middleware interface," and the tree has no
such token. That is a real, small, unshipped deliverable that closing #2930 as a
duplicate of #2904 would silently drop.

### The "without touching core" premise is false against today's seam

Worth catching before anyone implements #2930 as written. The issue's gap
statement asks for an ABI "a third party can build against **without touching
core**." The shipped seam cannot satisfy that, by design rather than omission:
`cmd/extseamsdemo/main.go` catalogues it as `Attachment: in-process`,
`Trust: trusted-compiled`, `UseWhen: "the code must surround every call and
ships in the trusted binary"` — and the demo's own validator *refuses* any
in-process seam not marked `trusted-compiled`. A third-party middleware today
must compile into the trusted binary, which is exactly "touching core."

So #2930 conceals a fork the issue never names: either it means the existing
in-process trusted-compiled seam (and "without touching core" is loose wording
for "without editing the adjudicator"), or it means a genuine out-of-process
plugin ABI (a much larger architectural bet). The classification below assumes
the first; the second would re-stream it.

### Active Classification

**#2930 classified `gen/next`, milestone `Generation G1 - Next Gen`** — matching
#2864, its nearest-shaped sibling. It carried `enhancement`, `observability`,
`substrate`, and `class:infra` but no `generation` label, no stream label, no
`hermes-inspired`, and no milestone, so it was invisible to every `generation-*`
dispatch view.

Why `gen/next`: the stream is "near-term foundation that should be runnable by
agents soon, but still needs a gate, dogfood run, schema, or default-exposure
proof." #2930's residuals are one of each — the journal record and the version
token are *schema*, and the inert seam is missing *default-exposure proof*.

Why not `gen/now`: `gen/now` needs a current-product witness, and a seam with
zero callers taps nothing and adjudicates nothing in production. Crediting the
green package as shipped protection is the current-work-laundering anti-pattern.

Why not `gen/second-next`: under the in-process reading, adding a record and a
version token is covered by the additive-only frozen-ABI promise in
[`generation-abi-compatibility-policy.md`](../generation-abi-compatibility-policy.md);
no simulation or compatibility policy is implicated.

### Promotion Evidence

- A **live call site** routing a real `llm_request`/`tool_request` through
  `Chain`, plus a dogfood run on a real `fak serve` session — the unit tests
  cannot distinguish "wired" from "inert."
- A **journaled decision record** per middleware invocation (hook identity,
  verdict, latency), surviving the recovered-panic path, or the witness is as
  silently-disableable as the swallowed exception it exists to prevent.
- An **explicit version token** on the seam, satisfying #2930's unique clause.

### Demotion, Parking, Or Retirement Evidence

- **Re-scope, do not close as duplicate.** The fail-open observer half is
  genuinely shipped, but the journal, the version token, and the wiring are not.
  Narrow #2930 to those three and cross-link #2904 and #2864.
- **#2864 absorbs it.** #2864 and #2930 now differ only by the version token. If
  #2864 lands the record and the wiring first, #2930 retires into it plus a
  one-line version constant — the cheapest resolution available, and the reason
  these two should be worked as one unit rather than dispatched separately.
- **The seam is deliberately inert.** If an operator confirms the seam is an
  *offered* extension point rather than a path fak routes through, residual 2 is
  not a gap and #2930 collapses to the record and the version token alone.

### Invalidating Assumption

The classification assumes #2930's "without touching core" means the existing
in-process trusted-compiled seam. If it instead means an out-of-process plugin
ABI — a third party shipping a binary fak loads without recompiling — then this
is not `gen/next` foundation work at all: it needs a compatibility policy, a
process boundary, and a trust model the `trusted-compiled` catalogue explicitly
refuses today, which is `gen/second-next` by definition. **This is the one
question an operator should answer before #2930 is dispatched to implementation**,
and it is not answerable from the issue body, which asserts the constraint
without naming a mechanism.

A second invalidator, inherited from the #2864 analysis above: residual 2 rests
on a signature grep, which cannot see an adapter that wraps `Apply` behind its
own verdict type.

## Intake Repair Still Owed

`hermes-inspired` was missing from #2864 and #2930 and is missing batch-wide —
#2865, #2866, and #2867 also lack it despite descending from the same
decomposition. Repaired here for #2864 and #2930 only. The rest want a sweep,
which would also have surfaced the #2864/#2904/#2930 triplication at filing
time: all three issues carry the label semantics but only #2904 carried the
label.

That sweep is the actual systemic fix. Three epics (#2834, #2871, #2908) each
decomposed the same Hermes material into the same leaf, and nothing at filing
time cross-checked them, so the fleet now carries two open issues describing
already-shipped code as unbuilt. A dedupe pass keyed on `hermes-inspired` would
have caught all three at intake.

---
title: "fak model routing: per-aspect models and ensembles"
description: "How fak routes one request by aspect, from tool calls to reasoning steps, with deterministic policy manifests and configurable model ensembles."
---

# Model routing — first-class at every level (`fak route`)

fak model routing is a way to route a single request at any aspect — the whole request, one tool call, a sub-query, a planner state, or a reasoning step — each to a different model, with first-class ensembles folded by a configurable reduction (first, vote, best_of, all_reduce, or concat), all expressed as one deterministic, verifiable policy manifest. Most LLM routers answer only "which single model serves this whole request?"; fak makes the routing decision first-class at every level instead. The routing decision spine and the ensemble reduce are shipped and witnessed by go test (internal/modelroute, fak route), along with an offline routing benchmark (fak routebench) that compares per-aspect and ensemble policies against a single-model baseline with no model in the loop. Live multi-model dispatch that executes a decision on real engines is still a stub tracked as a GitHub issue series; any "10x" is a categorical capability framing and a target to be measured, never a measured result.

> **Status.** The routing **decision** spine and the ensemble **reduce** are
> [SHIPPED] (`internal/modelroute`, `fak route`, witnessed by `go test`). The
> **offline routing benchmark** (`fak routebench` — per-aspect + ensemble vs
> single-model on cost/quality/latency, no model in the loop) is [SHIPPED]. The
> **live multi-model dispatch** that executes a decision on real engines is
> [STUB] — tracked as a GitHub issue series. See [`CLAIMS.md`](../CLAIMS.md).

## The one-paragraph version

Most LLM "routers" answer one question: *which single model should serve this
whole request?* fak makes model routing first-class at **every level**. The unit
of routing is an **aspect** — the whole request, **one tool call**, a sub-query, a
planner state, a reasoning step — so a single request can send its `refund_payment`
tool call to a two-model guard ensemble, its `search_kb` call to a small model, and
its hard reasoning step to a large model, **each decided by the same policy**. And
an **ensemble** — a *set* of models on one item, folded by a **reduction**
(`first` / `vote` / `best_of` / `all_reduce` / `concat`) — is a first-class plan,
not a bolt-on.

## Why this is different from the SOTA

Surveyed 2025–2026 routers and gateways. Every one routes the **whole request** to
**one model**; the only shipped model ensemble is a single fixed recipe.

| Product | Routes at | Ensemble | The gap fak fills |
|---|---|---|---|
| RouteLLM (LMSYS) | request | none | binary strong/weak pick of one model per request |
| Martian | request | none | one best model per request; proprietary learned mapping |
| NotDiamond | request | none | per-prompt single-model selection |
| Unify.ai | request | none | trained predictor → one model+provider per prompt |
| OpenRouter | request | **fallback** + Fusion | Fusion is a *fixed* parallel-synthesize recipe, not a configurable per-aspect reduction |
| Portkey | request | fallback | deeply composable config, but each request still resolves to **one** model; keys are whole-request only |
| LiteLLM Router | request | fallback | load-balance/failover among deployments of one model |
| Aurelio Semantic-Router | request | none | routes to an *intent/route*, not to a model |
| vLLM / SGLang router | replica | none | balances **replicas of the same model** for KV locality — not model selection (a different layer) |

**The honest claim** (no measured multiple): *to our knowledge, fak is the only
design that routes at any aspect of a single request — each to a different model —
with first-class ensembles and configurable reductions, expressed as one
deterministic, verifiable policy.* This is a **categorical** capability gap, not a
benchmarked speed/quality win. Any "10×" is a **target to be measured**, never an
inferred or borrowed number. "Deterministic" is scoped to the routing **decision**
and the reduce **fold** — model **outputs** from non-bit-exact engines are not
reproducible, and we never claim they are.

The axes on which per-aspect + ensemble routing can become 10× over time:

1. **Granularity** — sub-request routing is a new capability no surveyed product exposes.
2. **First-class ensembles with configurable reductions** — declarable, not a fixed recipe.
3. **One policy** instead of hand-assembling a router + a gateway + an ensemble tool + an intent layer.
4. **Determinism + verifiability** of the routing decision (auditable, content-addressable).
5. **Routing inside the agent loop** — the tool call is already an in-process syscall, so per-aspect routing rides an existing cut point at near-zero added latency.

## The shape

```
Subject  ──Route──▶  Decision { Plan }
  aspect            Plan = one Member  (a PICK → abi.ToolCall.Engine)
  tool                   | many Members + a Reduction  (an ENSEMBLE)
  prompt_tokens
  latency           Votes ──Combine(reduction)──▶ Result { output, winner, tally }
  complexity              first | vote | best_of | all_reduce | concat
  labels{...}
```

- **Subject** — the classified aspect to route. Unset fields are wildcards.
  `Aspect` is an **open** set (route your own named stage); `Latency`,
  `Complexity`, and `Reduction` are **closed** vocabularies.
- **Plan** — `len(Members)==1` is a single pick; `>1` is an ensemble + a reduction.
  `Scout` names an optional cheap classify-first model.
- **Manifest** — an ordered `Rule` list (`Match → Plan`) + a fail-closed `Default`.
  A version-tagged JSON file, validated fail-loud (`fak route --dump` → edit →
  `--check` → `--manifest`), exactly like the capability-floor policy manifest.
- **Combine** — folds member outputs deterministically (member order preserved).

## How it works (the data flow)

A routed call moves through the kernel in five steps. Steps 1–2 are the shipped
pure spine; steps 3–5 are the wiring the epic tracks. The ordering is not
cosmetic — it is what keeps the default-deny floor intact (see the contract below).

```
   the host (gateway / agent loop)                 the kernel
   ───────────────────────────────                 ──────────
1. classify ──▶ Subject{aspect, tool, tokens, latency, complexity, labels}
2. Route(Subject) ──▶ Decision{ rule, Plan }            (pure, deterministic)
                          │
                          ├─ PICK  (1 member)
                          │     3. set ToolCall.Engine = Plan.Primary()  ◀── BEFORE submit
                          │     4. Kernel.Submit (adjudicate — residency PDP sees the engine) ─▶ Reap (dispatch)
                          │
                          └─ ENSEMBLE (N members)
                                3. for each member: a ToolCall with Engine = member.Model
                                4. N independent Submit (each adjudicated) ─▶ Reap (each dispatched)
                                5. gather outputs IN MEMBER ORDER ─▶ Combine(reduce) ─▶ Result
```

1. **Classify.** The host turns the thing it is about to do into a `Subject` — the
   aspect (a whole request, one tool call, a sub-query, a step), the tool name, an
   estimated prompt length, a latency/complexity hint, and any labels (domain,
   tenant, language).
2. **Route.** `Manifest.Route(Subject)` walks the rules top-to-bottom; the first
   `Match` that fires returns its `Plan`, else the fail-closed `Default`. This is
   pure and side-effect-free — the same subject always yields the same decision.
3. **Bind the engine (pre-submit).** For a single-model plan the host writes
   `Plan.Primary()` to `abi.ToolCall.Engine`. For an ensemble it builds **N** tool
   calls, one per member, each carrying its member model in `Engine`.
4. **Adjudicate, then dispatch.** Each call goes through `Kernel.Submit`, which
   folds the adjudicator chain (including the residency PDP that reads `Engine`)
   *before* dispatching. The kernel's `routeFor` then resolves `Engine` to a
   registered engine and runs the call.
5. **Reduce (ensemble only).** The host gathers the members' outputs **in member
   order** and folds them with `Combine(Plan.Reduce, votes)` into one `Result`.

Today the spine produces the `Decision` (steps 1–2) and the fold (`Combine`, step
5's math); steps 3–4 — writing `Engine` and executing — are the [STUB] wiring.

## Manifest reference (`fak-route/v1`)

A manifest is an ordered rule list plus a fail-closed default. `fak route --dump`
prints a starter; `--check` validates one (unknown fields are rejected).

**Top level**

| Field | Type | Meaning |
|---|---|---|
| `version` | string | schema tag; omit for current, a different MAJOR is refused |
| `default` | Plan | applied when no rule matches — **must** name ≥1 model (fail-closed) |
| `rules` | [Rule] | evaluated top-to-bottom; **first match wins** |

**Rule** = `{ name (unique), match, plan }`.

**Match** (every set field must hold; unset = wildcard)

| Field | Type | Meaning |
|---|---|---|
| `aspect` | string (open) | `request` / `tool_call` / `query` / `state` / `step` / your own stage |
| `tool` | string | exact name, or a single trailing `*` prefix (`git_*`), or `*` for any |
| `min_prompt_tokens` / `max_prompt_tokens` | int | token band; `max=0` is unbounded |
| `latency` | enum | `interactive` / `batch` (closed) |
| `min_complexity` | enum | floor: `low` < `medium` < `high` (closed) |
| `labels` | map | every pair must equal the subject's label |

**Plan** = `{ members, reduce, scout, reason }`

| Field | Type | Meaning |
|---|---|---|
| `members` | [Member] | 1 = a PICK; >1 = an ENSEMBLE |
| `reduce` | enum | required for an ensemble: `first` / `vote` / `best_of` / `all_reduce` / `concat` |
| `scout` | string | optional cheap model that classifies the subject first |
| `reason` | string | free-text note surfaced in the decision trace |

**Member** = `{ model, weight (vote/aggregate weight, default 1), role (primary / drafter / verifier / judge / …) }`.

**Reductions:** `first` (fastest-wins / fallback), `vote` (weighted majority, deterministic tie-break), `best_of` (highest `Vote.Score` from a judge), `all_reduce` (weighted numeric **mean** of scalar outputs — *not* a tensor all-reduce), `concat` (gather, member order).

## The matching primitive (`Match.Matches` — the envelope-matching spine)

`Match.Matches(Subject)` (`internal/modelroute/modelroute.go`) is the single
tag-matching primitive every routing rule reduces to. It has the *shape* of MPI's
point-to-point envelope match, without the point-to-point delivery:

- **A set field is a required tag; an unset field is a wildcard.** `Match` tests the
  `Subject` field-by-field under logical AND — every field the rule sets must hold, and
  a field the rule leaves empty matches anything. An empty `aspect`, `tool`, `latency`,
  or `min_complexity`, or an unbounded token band (`max_prompt_tokens=0`), each play the
  `MPI_ANY_SOURCE` / `MPI_ANY_TAG` wildcard role for their own dimension — "match any
  value of this field." This is the only wildcard discipline in the routing spine:
  anchor here, do not reinvent a parallel matcher.
- **`Labels` are key/value tag pairs.** Every pair the rule sets must equal the
  subject's label for the same key (`s.Labels[k] == v`); a key the rule omits is a
  wildcard for that key. Labels are the OPEN tag channel — domain, tenant, language,
  taint — that a deployment matches on without a code change.
- **`tool` adds one wildcard form.** It matches an exact name, a single trailing `*`
  prefix (`git_*` matches `git_push`), or bare `*` for any tool. The remaining fields
  are exact, or — for the token band and `min_complexity` — banded / floored.
- **Rules are first-match-wins.** `Manifest.Route` walks `Rules` top-to-bottom and
  returns the first `Match` that fires, else the fail-closed `Default`: the deterministic
  first-match receive of the envelope analogue. Put the most specific rules first.

**Honesty caveat — it selects an engine, not a receiver.** `Match.Matches` selects a
*Plan* (and therefore the engine or engines) for a *Subject*; it does not select a
receiver for a message. fak borrows the envelope-matching **structure** — set field =
required tag, unset = wildcard, first-match-wins — from MPI's `MPI_ANY_SOURCE` /
`MPI_ANY_TAG` receive. It does **not** borrow point-to-point delivery, source ranks, or
rendezvous: there is no message queue and no source-rank ordering behind a match. The
match decides *which model runs*; the wiring contract above is what actually runs it.

## The 60-second proof (no key, no model, no GPU)

```bash
# per-tool-call routing — a write-shaped tool call goes to a two-model guard ensemble
go run ./cmd/fak route --aspect tool_call --tool write_file

# a real manifest: route different aspects of one request to different models
go run ./cmd/fak route --manifest examples/model-routing.example.json --aspect tool_call --tool search_kb        # -> small
go run ./cmd/fak route --manifest examples/model-routing.example.json --aspect step --complexity high            # -> large

# the ensemble half, end to end: fold stand-in member outputs through the plan's reduction
go run ./cmd/fak route --manifest examples/model-routing.example.json \
  --aspect tool_call --tool refund_payment --simulate "approve,deny,approve"   # -> vote: approve (2 vs 1)

# author / validate the policy
go run ./cmd/fak route --dump                                   # the built-in starter manifest
go run ./cmd/fak route --check examples/model-routing.example.json
```

## The cost lens (usage saved vs the SOTA frontier)

Routing earns its keep by *not* sending every aspect to one big model — so on every
decision `fak route` prints a rough estimate of what the chosen plan costs against
the SOTA baseline: one frontier model for everything (the naive default a
request-level router reduces *from*).

```bash
go run ./cmd/fak route --latency interactive --prompt-tokens 100
# usage (rough public list prices, overridable; not a bill): ~92% cheaper than
# always-frontier -- plan ~$1.25 vs $15 /Mtok-out (saves ~$13.75/Mtok-out)

go run ./cmd/fak route --aspect tool_call --tool write_file
# usage ...: +100% vs one frontier call -- 2-model ensemble ~$30 vs $15 /Mtok-out
# (a deliberate reliability spend) [unpriced, charged at frontier: guard-a, guard-b]

go run ./cmd/fak route --check examples/model-routing.example.json   # a cost tag per rule
```

The math is deliberately rough and **honest by construction**:

- Anchored to the repo's published price convention (Opus-class **$3 in / $15 out
  per Mtok** — see `experiments/parity`, `cmd/fanbench`) and fully overridable:
  `--prices small=0.25/1.25,large=3/15` and `--frontier MODEL` reprice the lens, so
  the number is a transparent function of stated inputs, never a hidden claim.
- An **ensemble costs more** than one frontier call, so its "savings" is negative —
  reported as a deliberate reliability **premium**, never dressed up as a saving.
- An **unpriced** model is charged at the conservative frontier rate *and* disclosed
  — fak never invents a cheap number to flatter the route.
- It is a **price-rate estimate** for choosing a policy, **not** a measured
  speed/quality multiple (the same distinction this page draws above).

## The wiring contract (load-bearing — read before wiring dispatch)

The decision spine is pure; executing a decision on real engines is the [STUB]
half. The wiring **must** honor three rules so it cannot regress fak's default-deny
floor:

1. **Route before adjudicate.** Write the chosen model to `abi.ToolCall.Engine`
   **before** `Kernel.Submit`, never as a dispatch-time override. The residency PDP
   (`internal/engine`) reads `c.Engine` *inside* the adjudication fold to deny a
   tenant/sensitive payload bound for a **remote** engine. If routing set the model
   only at dispatch, that gate would have adjudicated an empty route and the
   sensitive payload would reach a remote model **fail-open**.
2. **An ensemble expands to N independently-adjudicated calls.** Executing a Plan
   with more than one member is N separate `Kernel.Submit` calls, each carrying its
   member model in `Engine`, each crossing the syscall boundary on its own.
3. **Member order is preserved into the fold.** The dispatcher gathers member
   outputs into the `Combine` `[]Vote` in `Plan.Members` order (not engine
   completion order), or the order-sensitive reductions stop being deterministic.

## Connecting routed models to providers (LiteLLM, routers, your accounts)

The manifest above picks *abstract* model ids ("small", "large", "guard-a"). Where each
of those actually runs — a LiteLLM proxy fronting 100+ providers, an OpenRouter or Portkey
gateway, a direct provider wire, or a local engine — is a binding the dispatch layer
resolves, and it is the same OpenAI wire pointed at a different `base_url` in nearly every
case (the field's lingua franca). So fak does not reimplement a provider: it owns the
*decision* (per aspect, with ensembles) and the *floor*, and lets an aggregator be the
connectivity for each chosen member. The dedicated guides:

- **[fak + LiteLLM](integrations/litellm.md)** — the three topologies (fak in front of a
  LiteLLM proxy, fak as a governed node behind it, and fak's per-aspect routing
  dispatching *through* it) and what each means.
- **[Routers & gateways](integrations/routers.md)** — OpenRouter, Portkey, LiteLLM
  Router, Unify, Martian: fak as a complement to request-level routers.

**Residency is fail-closed across every backend.** The engine-residency PDP
(`internal/engine`) treats any route it cannot prove is on-box — a provider wire, a
LiteLLM/OpenRouter aggregator, or your own gateway — as **remote**, and denies a
tenant-scoped / sensitivity-tagged payload bound for it before dispatch (an `inkernel` /
`local` / `on-device` route is exempt). Connecting your routing to a third-party
proxy therefore cannot silently open an exfiltration path — an unknown backend is assumed
remote, not trusted.

## Routing presets (examples/routing-presets/)

For adopters who want a starter that matches a single goal, the multi-rule
mega-example above is split into **named, single-purpose presets** — the routing
analogue of how `examples/presets/` ships ready-made capability floors. Copy the
one that matches your intent, then `fak route --check` it. Each is a valid
`fak-route/v1` manifest (a different schema + loader from the `fak-policy/v1`
pack in `examples/presets/`, so it lives in its own directory); a round-trip test
in `internal/modelroute` guards every preset against rot.

| Preset | Goal | Shape |
|---|---|---|
| [`cost-saver.json`](../examples/routing-presets/cost-saver.json) | spend less | interactive/short + read-shaped tool calls → small; only `min_complexity: high` → large; default → small |
| [`guard-writes.json`](../examples/routing-presets/guard-writes.json) | never ship a write unchecked | every `write_*` / `delete_*` tool call → a two-model `vote` ensemble; else a single default |
| [`best-of-quality.json`](../examples/routing-presets/best-of-quality.json) | best answer on hard work | hard aspects → a drafters + judge `best_of` ensemble; medium → medium; cheap → small |
| [`scout-then-route.json`](../examples/routing-presets/scout-then-route.json) | classify before you route | a cheap `scout` labels complexity first, then high → large / low → small |

```bash
go run ./cmd/fak route --check examples/routing-presets/cost-saver.json
go run ./cmd/fak route --manifest examples/routing-presets/guard-writes.json --aspect tool_call --tool write_file
```

A `fak route --preset NAME` resolver (copy-by-name without spelling the path) is
an optional follow-up; the presets are plain manifests today, so `--manifest
<path>` already loads any of them.

## The offline routing benchmark (`fak routebench`)

The survey above frames per-aspect + ensemble routing as a *categorical*
capability gap and is explicit that any "10x" is "a target to be measured, never
an inferred or borrowed number". `fak routebench` is the measuring instrument. It
runs a **corpus** of recorded cases through **two** manifests — a routed policy
(per-aspect + ensemble) and a single-model baseline (the SOTA shape: one frontier
model for everything) — and prints the delta on three axes:

- **cost** — reuses the `fak route` cost lens (rough $/Mtok-out summed over
  members); per-aspect routing pays the frontier rate only on hard aspects, an
  ensemble pays it on every member (a deliberate premium).
- **latency** — a rough per-call latency summed over members (the latency
  analogue of the cost lens); an ensemble does *N* members' work, so its total
  compute is the sum (a parallel dispatch's wall-clock is bounded by the max,
  which this lens deliberately does not assume).
- **quality** — the fraction of cases whose folded output equals the expected
  answer; an ensemble can *win* here (a `vote`/`best_of` that folds to the right
  answer where a single model errs) and a downgrade can *lose* (a cheap model
  wrong where the frontier was right).

**Offline means offline.** Each case carries the stand-in OUTPUT every candidate
model produces for it (a recorded answer, never a live model call) — exactly as
`fak route --simulate` already does — so the benchmark reuses the two pure,
already-witnessed halves of the package (`Route` + `Combine`) over fixed votes.
It is **deterministic end to end**: no key, no GPU, no network. It measures what
the *policy* does to a *recorded workload*, not what a non-bit-exact engine would
do live (that is the [STUB] dispatch half). Every figure is a **rough lens**, never
a bill or a measured SLA.

```bash
# the built-in 8-case demo corpus + DefaultManifest vs a one-frontier-model baseline
fak routebench

# your own corpus + manifests (the demo corpus + the two baseline manifests ship as fixtures)
fak routebench --corpus examples/routing-bench/demo-corpus.json \
               --routed examples/routing-bench/routed.json \
               --single examples/routing-bench/single-model.json

fak routebench --dump-corpus > my-corpus.json   # the starter corpus to edit
fak routebench --json                            # machine-readable comparison
```

The built-in demo corpus is an **honest trade, not a rigged win**: per-aspect
routing is cheaper and faster on the easy aspects (they hit the small/mid tier),
the two-model `vote` ensemble is a deliberate *premium* that *rescues* one case the
single model gets wrong, and a downgrade to the default *loses* one case the
single model got right — so on the demo the quality deltas offset (cost ~20%
cheaper, total compute ~10% less, quality tied). The corpus is a recorded fixture
to make the benchmark runnable now, **not** a claim about real traffic. A
round-trip test in `internal/modelroute` guards every committed fixture against rot
and re-asserts the documented numbers.

## Roadmap (the GitHub issue series)

The decision spine is the foundation; the offline benchmark (`fak routebench`) is
shipped; the rest is wiring, each a tracked issue:

- Wire a single-model route into the kernel/gateway: set `ToolCall.Engine` from
  `Decision.Plan.Primary()` **pre-submit** (honoring the residency ordering).
- Execute an ensemble Plan in the gateway: N adjudicated submits + `Combine`.
- Per-tool-call routing inside the agent loop (`agent.execViaKernel`).
- Scout-model live classification (a cheap model fills `Subject.Complexity`/labels).
- Telemetry → learned routing (LIVE cost/latency/quality feedback feeding the
  policy, RouteLLM-style but per-aspect — the offline benchmark measures a recorded
  corpus; this is its live, self-improving counterpart).
- Manifest hot-reload + `fak serve --route-manifest`.
- Free-text ensemble reductions (a judge/verifier model for `best_of` beyond scalar scores).
- Routing observability (per-aspect decisions in `/metrics` + the decision journal).
- Speculative/draft roles bridged to `internal/polymodel` (drafter/verifier members).
- Industry-scorecard row positioning vs the surveyed routers.

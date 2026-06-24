# Model routing — first-class at every level (`fak route`)

> **Status.** The routing **decision** spine and the ensemble **reduce** are
> [SHIPPED] (`internal/modelroute`, `fak route`, witnessed by `go test`). The
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

## Roadmap (the GitHub issue series)

The decision spine is the foundation; the rest is wiring, each a tracked issue:

- Wire a single-model route into the kernel/gateway: set `ToolCall.Engine` from
  `Decision.Plan.Primary()` **pre-submit** (honoring the residency ordering).
- Execute an ensemble Plan in the gateway: N adjudicated submits + `Combine`.
- Per-tool-call routing inside the agent loop (`agent.execViaKernel`).
- Scout-model live classification (a cheap model fills `Subject.Complexity`/labels).
- Telemetry → learned routing (cost/latency/quality feedback, RouteLLM-style but per-aspect).
- Manifest hot-reload + `fak serve --route-manifest`.
- Free-text ensemble reductions (a judge/verifier model for `best_of` beyond scalar scores).
- Routing observability (per-aspect decisions in `/metrics` + the decision journal).
- Speculative/draft roles bridged to `internal/polymodel` (drafter/verifier members).
- Industry-scorecard row positioning vs the surveyed routers.

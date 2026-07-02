---
title: "fak + LiteLLM: govern the gateway, route per aspect, keep one wire"
description: "How fak and LiteLLM compose — three concrete topologies (fak in front of a LiteLLM proxy, fak as a governed model behind LiteLLM, and fak's per-aspect routing dispatching through LiteLLM), what each one means for governance and residency, and why supporting LiteLLM is one integration, not a hundred."
---

# fak + LiteLLM

[LiteLLM](https://docs.litellm.ai/) gives one OpenAI-compatible endpoint in front of 100+
providers, with load-balancing, failover, budgets, and key management. `fak` is the agent
kernel: a default-deny capability floor that adjudicates every tool call, plus first-class
**per-aspect model routing** and **ensembles**. They solve different problems, so they
compose cleanly — the only question is *where fak sits relative to the proxy*.

> **TL;DR.** LiteLLM is connectivity; fak is governance + routing. They speak the same
> wire (OpenAI Chat Completions), so wiring them together is a one-line base-URL change in
> whichever direction you need. Put fak **in front of** LiteLLM to govern everything it
> routes; put fak **behind** LiteLLM as a governed model node; or let fak's **own
> per-aspect routing** dispatch each aspect through LiteLLM so you never reimplement a
> provider. A payload that leaves the box is treated as **remote** by the residency floor
> in every case.

## The one insight: it's one integration, not a hundred

The OpenAI Chat Completions wire is the field's lingua franca. LiteLLM, OpenRouter,
Together, Groq, Fireworks, vLLM, SGLang, llm-d, llama.cpp, Ollama, and most clouds all expose
it. So "support LiteLLM" is not a bespoke adapter — it is **the OpenAI wire pointed at a
different `base_url`**. fak already speaks that wire on both sides (as a server to clients,
as a client to upstreams), which is why every topology below is a base-URL change, not
code.

What fak adds *above* that wire is exactly what an aggregator does **not** do:

- **A disinterested capability floor.** LiteLLM routes and meters tokens; it does not
  adjudicate the *tool calls* a model proposes. fak denies by structure, repairs malformed
  calls, and quarantines poisoned tool results — and because fak does not author your
  model, it referees with no conflict of interest.
- **Routing at every aspect, not just the request.** LiteLLM's router picks one
  deployment per request. fak routes an **aspect** — the whole request, *one tool call*, a
  sub-query, a reasoning step — and runs **ensembles** with configurable reductions. See
  [model routing](../model-routing.md).

## Three topologies (and what each one means)

### 1. fak in front of LiteLLM — govern everything the proxy routes

```text
your agent ──▶ fak serve ──▶ LiteLLM proxy ──▶ {OpenAI, Anthropic, Bedrock, …}
            (capability floor,  (connectivity,
             quarantine, audit)  failover, budgets)
```

LiteLLM speaks the OpenAI wire, so point `fak serve` at it like any upstream:

```bash
fak serve --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://127.0.0.1:4000/v1 \   # your LiteLLM proxy
  --model gpt-4o \                          # a model id your LiteLLM config serves
  --api-key-env LITELLM_KEY \               # the proxy's master/virtual key
  --policy floor.json                       # omit for the fail-closed default floor
```

Then point your agent at fak (`OPENAI_BASE_URL=http://127.0.0.1:8080/v1`). **What it
means:** you keep every LiteLLM feature — provider fan-out, retries, spend caps — and add
a capability floor and audit trail *in front of all of it*. The tool calls your agent
proposes cross the floor before LiteLLM ever routes them, and a poisoned tool result is
quarantined out of context before it reaches the model. This is the most common ask and is
fully shipped.

### 2. fak behind LiteLLM — a governed model node in your routing fabric

```text
your agent ──▶ LiteLLM router ──┬──▶ fak serve ──▶ model     (the governed lane)
                                └──▶ provider direct          (everything else)
```

Register `fak serve` as one OpenAI-compatible deployment in LiteLLM's `model_list`:

```yaml
# litellm config.yaml
model_list:
  - model_name: governed-gpt-4o
    litellm_params:
      model: openai/gpt-4o            # LiteLLM's "openai/" custom-provider prefix
      api_base: http://127.0.0.1:8080/v1   # fak serve
      api_key: os.environ/FAK_TOKEN        # if you set --require-key-env
```

**What it means:** fak becomes "the governed model" inside the routing fabric you already
run. You can send the high-risk agent, the sensitive tenant, or the untrusted workload
through the `governed-*` deployment and leave the rest direct — selective governance with
no re-architecting. Shipped, because fak is just an OpenAI-compatible endpoint here.

### 3. fak's per-aspect routing dispatching *through* LiteLLM — the differentiator

```text
your agent ──▶ fak  ── route per aspect / ensemble ──▶ per member ──▶ LiteLLM ──▶ provider
                 │                                                     (connectivity)
                 └─ owns: the decision, the floor, determinism, residency
```

This is the case the design is built for: *you use fak's kernel and ensemble, and want
LiteLLM to connect each chosen model to the actual provider.* fak **decides** which model
— or which ensemble, folded by a reduction — serves each aspect of a request (the
[categorical capability](../model-routing.md#why-this-is-different-from-the-sota) no
request-level router has). To **execute** that decision, each member dispatches to a
backend — and since LiteLLM speaks the OpenAI wire, a member's backend is simply the
OpenAI wire pointed at your LiteLLM proxy. fak never reimplements a provider; LiteLLM is
the connectivity fabric for the members fak chose.

The division of labor is the point:

| Concern | Owner |
|---|---|
| *Which* model / ensemble serves each aspect, and how outputs fold | **fak** (`internal/modelroute`, `fak route`) |
| The capability floor, quarantine, audit on each member call | **fak** (the kernel) |
| Determinism of the decision + the reduce; engine **residency** | **fak** |
| Connecting each chosen model to a concrete provider | **LiteLLM** |

**Status (honest).** The routing **decision** and the ensemble **reduce** are shipped and
pure (`fak route`, witnessed by `go test`). The **live multi-backend dispatch** that runs
each member on its bound backend and folds the results is the tracked `[STUB]` in
[`CLAIMS.md`](../../CLAIMS.md) — see the wiring contract in
[model routing](../model-routing.md#the-wiring-contract-load-bearing).
Authoring the routing policy and previewing the decision works today; binding each routed
model id to a LiteLLM-backed account and executing the ensemble is the model-routing
roadmap.

## Residency: a payload that leaves the box is remote — including through LiteLLM

The load-bearing safety property when you connect your routing to *any* aggregator: fak's
residency floor (`internal/engine`) is **fail-closed**. It treats an engine route it
cannot prove is on-box — a direct provider wire, a LiteLLM/OpenRouter/aggregator proxy, or
your own gateway — as **remote**, and denies a tenant-scoped or sensitivity-tagged payload
bound for it before dispatch. So routing a member through a LiteLLM proxy does not quietly
open an exfiltration path: a sensitive aspect routed off-box is refused, an on-box engine
(`inkernel`, a `local`/`on-device` route) is exempt. The route must be written to the call
**before** adjudication (route-before-adjudicate), which is exactly how the routing wiring
is specified.

This is why "first-class LiteLLM support" is safe by construction rather than a hole: the
floor classifies an unknown backend as remote, not as trusted.

## Prove the wire with no LiteLLM install (60 seconds)

You can confirm topology #1's gate is real before standing up a proxy — `fak serve` with
no `--base-url` runs a deterministic offline mock, so the floor is exercisable with no
model and no key:

```bash
python3 examples/wire-proof/verify.py   # -> PASS, exit 0
```

Then swap the mock for your LiteLLM proxy by adding `--base-url http://127.0.0.1:4000/v1`;
nothing else about your agent changes.

## Cross-references

- [Routers & gateways](routers.md) — OpenRouter, Portkey, LiteLLM Router, Unify, and the categorical-complement positioning vs request-level routers.
- [Model routing — first-class at every level](../model-routing.md) — the per-aspect + ensemble spine, the manifest, the cost lens, and the wiring contract.
- [Interoperability stance](interoperability.md) — bring your own agent, model, and protocol; the one opinion fak keeps.
- [Compatibility matrix](compatibility-matrix.md) — the sourced row for LiteLLM-backed harnesses (OpenHands, Aider, CrewAI, DSPy, Google ADK, Strands) and every other surveyed tool.
- [Integration index](README.md) — the universal "repoint one base URL" recipe.
- [Claims ledger](../../CLAIMS.md) — what is shipped vs stub, claim by claim (the live multi-backend dispatch is honestly tagged `[STUB]`).

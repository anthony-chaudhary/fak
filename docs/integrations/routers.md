---
title: "fak + routers and gateways (OpenRouter, Portkey, LiteLLM Router, Unify)"
description: "How fak relates to LLM routers and gateways — it is a complement, not a competitor. Routers pick one model per request and connect to providers; fak governs the tool-call boundary and routes at every aspect with ensembles. The three topologies and the honest categorical positioning."
---

# Routers & gateways

LLM **routers** and **gateways** — [OpenRouter](https://openrouter.ai/docs/quickstart),
[Portkey](https://portkey.ai/docs), [LiteLLM](litellm.md) (proxy *and* Router),
[Unify](https://unify.ai/), [Martian](https://withmartian.com/),
[NotDiamond](https://www.notdiamond.ai/), the [Vercel AI Gateway](https://vercel.com/docs/ai-gateway) —
answer one question: *given a request, which single model/provider should serve it, and
how do I reach it reliably?* They optimize **connectivity** (one wire to many providers),
**reliability** (failover, load-balance), and **selection** (cost/quality routing per
request).

`fak` answers a different question: *should this tool call run at all, and which model
serves each **aspect** of the request?* It is the capability floor plus
[per-aspect + ensemble routing](../model-routing.md). So fak is a **complement** to a
router, not a replacement — and the two compose over the shared OpenAI wire.

> **TL;DR.** A router connects and picks one model per request. fak governs the tool-call
> boundary and routes at every aspect (request, tool call, sub-query, reasoning step) with
> ensembles. Use both: the router for connectivity and failover, fak for the floor and the
> sub-request routing the router cannot express. Wiring is a base-URL change in either
> direction.

## Complement, not competitor

fak's routing is deliberately a different granularity from a request-level router. The
honest survey (full table and sourcing in [model routing](../model-routing.md#why-this-is-different-from-the-sota)):

| Product | Routes at | Ensemble | fak's relationship |
|---|---|---|---|
| OpenRouter | request | fallback + Fusion (fixed recipe) | complement: govern it (front), or be a node behind it; fak adds per-aspect + configurable reductions |
| Portkey | request | fallback | complement: composable gateway config; fak adds the tool-call floor + sub-request routing |
| LiteLLM Router | deployment | load-balance/failover of one model | complement: connectivity/HA; fak routes *which model*, per aspect — see [litellm.md](litellm.md) |
| Unify / Martian / NotDiamond | request | none | complement: learned per-request pick; fak routes sub-request aspects + runs ensembles |
| Vercel AI Gateway | request | none | complement: one key, many providers; fak governs + routes above it |
| **AgentGateway** (Linux Foundation) | **connectivity** (LLM, MCP, A2A) | **guardrails** (regex, moderation, webhooks) | **connectivity peer**: AgentGateway is the head-on connectivity competitor (MCP+A2A+LLM data plane). fak does not out-connect it; it adds the in-kernel capability floor + bit-exact KV cache they leave open. They focus on multi-protocol transport and rich observability; fak focuses on adjudication at the tool-call boundary and per-aspect ensemble routing. Compose fak behind AgentGateway for governed LLM/MCP/A2A connectivity, or front AgentGateway for multi-backend HA behind fak's floor. |

The claim fak makes is **categorical, not a benchmark**: to our knowledge it is the only
design that routes at *any aspect of a single request*, each to a different model, with
first-class ensembles and configurable reductions, under one deterministic, verifiable
policy. "Deterministic" is scoped to the routing *decision* and the *fold*, never to
non-bit-exact model outputs. Any speed/quality multiple is a target to measure, never an
inferred number.

## The three topologies (same as any gateway)

Every router here speaks the OpenAI wire (OpenRouter, Together-style aggregators, the
Vercel gateway) or is reachable as an upstream, so the wiring mirrors
[LiteLLM's](litellm.md):

1. **fak in front of the router** — `fak serve --base-url <router>/v1` governs everything
   the router routes. Example for OpenRouter:

   ```bash
   fak serve --addr 127.0.0.1:8080 --provider openai \
     --base-url https://openrouter.ai/api/v1 \
     --api-key-env OPENROUTER_API_KEY --model anthropic/claude-3.5-sonnet \
     --policy floor.json
   ```

2. **fak behind the router** — register `fak serve` as one OpenAI-compatible model in the
   router's deployment list (the router sends the governed lane through fak, the rest
   direct). The selective-governance pattern.

3. **fak's per-aspect routing dispatching through the router** — fak owns the decision and
   the floor; the router/aggregator is the connectivity for each chosen member. See the
   division-of-labor table and honest `[STUB]` status for the live multi-backend dispatch
   in [litellm.md, topology #3](litellm.md#3-faks-per-aspect-routing-dispatching-through-litellm--the-differentiator).

## Residency holds for every router

As with LiteLLM, fak's residency floor is **fail-closed**: a member or upstream routed to
*any* remote router/aggregator (or your own gateway) is treated as remote, so a
tenant-scoped or sensitivity-tagged payload bound off-box is denied before dispatch. An
on-box engine (`inkernel`, a `local`/`on-device` route) is exempt. Connecting your routing
to a third-party router does not silently widen the data-egress surface.

## Cross-references

- [fak + LiteLLM](litellm.md) — the flagship router/proxy integration, with the three topologies in full.
- [Model routing — first-class at every level](../model-routing.md) — the per-aspect + ensemble spine and the surveyed-router comparison this page summarizes.
- [Clouds & hosted providers](../supported/clouds.md) — OpenRouter, Together, Groq, Fireworks, Bedrock, Vertex, Azure over the OpenAI-compatible wire.
- [Compatibility matrix](compatibility-matrix.md) — OpenRouter, Together, Groq, Fireworks and 40 more, each with its wire and the exact repoint key.
- [Interoperability stance](interoperability.md) — bring your own agent, model, and protocol.
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — shipped vs stub, claim by claim.

---
title: "fak vs an API gateway / LLM router (OpenRouter, Portkey, LiteLLM, Kong AI Gateway)"
description: "The honest side-by-side for the router/gateway question 'don't I already have this?'. A router picks which model serves a request; fak governs which effects a tool call is allowed to have. They sit at different layers and compose, with fak in front of the gateway."
slug: fak-vs-routers
keywords:
  - fak vs OpenRouter
  - fak vs LiteLLM
  - fak vs Portkey
  - fak vs Kong AI Gateway
  - LLM router comparison
  - agent tool-call governance
  - default-deny capability gate
date: 2026-07-03
---

# fak vs an API gateway or LLM router

If you already run OpenRouter, Portkey, LiteLLM, or a Kong AI Gateway, the fair first
question is "don't I already have this?" Short answer: no, and you probably want both.
A router answers *which model should serve this request, and how do I reach it reliably?*
fak answers a different question one layer down: *should this tool call run at all, and
which model serves each aspect of the request?* The two do not overlap, so the useful
move is to keep the router and put fak in front of it.

This page is the focused router/gateway version of the whole-category
[capability matrix](matrix.md). The matrix scores fak against three product categories at
once; this page stays on the one category a router user cares about and shows the compose
topology. The deeper categorical argument, with the full surveyed-router table and
sourcing, lives in [routers & gateways](../../integrations/routers.md); this page links to
it rather than restating it.

## Which model vs which effect

A router operates on the *request*. Given an incoming request it selects a model or
provider, then connects to it, fails over if that provider is down, and often load-balances
across several. That is real, load-bearing work: one wire to many providers, retries,
per-request cost and quality routing. It is also the whole job. A router does not look
inside the turn at the individual tool calls the model wants to make.

fak operates on the *tool call*. Every tool call the model proposes crosses a default-deny
[capability floor](../../explainers/policy-in-the-kernel.md) before it runs: a tool that is
off the allow-list cannot be called no matter what the model was told, a destructive
argument (`rm -rf`, a write into `.git/`) is refused with a reason from a closed
vocabulary, and a poisoned tool *result* is held out of the model's context by structure
rather than by a classifier that has to catch the attack. The lock is a capability check,
not text recognition, so it fails closed regardless of which model the router picked.

Those are orthogonal jobs. The router chooses the effector; fak decides which effects that
effector is permitted to have. Neither one can do the other's job, which is exactly why
they compose instead of competing.

## The products are the same category, packaged differently

"Router / gateway" covers products that ship in genuinely different shapes, and it is worth
being accurate about that rather than flattening them:

- **OpenRouter** is a hosted service. You point at `openrouter.ai/api/v1` and it fans out
  to many providers behind one key. There is no binary to run.
- **LiteLLM** is an open-source Python proxy (and an in-process Router library). You run it
  yourself as a service, or import the Router in your own Python.
- **Portkey** is a gateway offered both as a hosted service and as a self-hostable
  gateway; the config is composable (fallbacks, retries, load-balancing).
- **Kong AI Gateway** is the AI plug-in surface on the Kong gateway, which ships as a
  compiled gateway binary/service you operate.

fak is a single static Go binary you drop in with one base-URL change (`fak serve
--base-url <router>/v1`, or `fak guard -- claude`). So on "is it one binary" the router
category is honestly mixed: Kong is a gateway binary, LiteLLM is a Python proxy, OpenRouter
is hosted. That mixed packaging is why the whole-category [matrix](matrix.md) scores the
gateway column *Partial* on the single-binary row rather than yes or no. None of this is a
knock on the routers; it just means "gateway" is a shape, not a single deliverable.

Some of these gateways also run guardrail plugins (content moderation, regex, PII redaction
on the request or response body). That is real request-path safety, and it is worth using.
It is not the same thing as adjudicating an individual tool call as a default-deny
capability lock: a guardrail plugin screens the text flowing through the wire, while fak
gates *whether the effect happens* on a path the model does not control. Use both; they
catch different failures.

## How they compose

Because every router here speaks the OpenAI wire (or is reachable as an upstream), the
wiring is a base-URL change in either direction. The three topologies are the same as for
any gateway, covered in full in [routers & gateways](../../integrations/routers.md#the-three-topologies-same-as-any-gateway);
the common one is the first.

**fak in front of the router.** fak owns the tool-call floor for everything, and the router
does the connectivity and model selection behind it:

```
agent / harness ──► fak serve ──► router (OpenRouter / Portkey / LiteLLM / Kong) ──► providers
                    │
                    └─ default-deny capability floor on every tool call
```

```bash
fak serve --addr 127.0.0.1:8080 --provider openai \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY --model anthropic/claude-3.5-sonnet \
  --policy floor.json
```

Your agent talks to fak; fak governs each tool call, then hands the request to the router,
which picks the model and reaches the provider. You keep the router's failover, key
management, and per-request routing, and you gain a capability floor the router does not
express.

The other two topologies are fak *behind* the router (register `fak serve` as one
OpenAI-compatible model so only the governed lane flows through it) and fak's own per-aspect
routing dispatching *through* the router. Both are described, with their honest status, in
the routers doc.

## The residency floor still holds

fak's residency floor is fail-closed and does not care that a router sits downstream. A
member or upstream routed to any remote router or aggregator is treated as remote, so a
tenant-scoped or sensitivity-tagged payload bound off-box is denied before dispatch. An
on-box engine (a local or in-kernel route) is exempt. Putting a router behind fak does not
silently widen your data-egress surface. Details in
[routers & gateways](../../integrations/routers.md#residency-holds-for-every-router).

## Where fak is weaker, honestly

This is a complement page, not a takedown, so the gaps belong on it:

- **fak does not out-connect a gateway.** A mature router's breadth of providers, its
  failover, load-balancing, key vaulting, and per-request cost routing are its whole job and
  it does them well. fak is not trying to replace that layer, and its own multi-backend
  dispatch (topology #3) is still marked stub in the [LiteLLM
  doc](../../integrations/litellm.md).
- **The KV eviction win is local-only.** fak's addressable, bit-exact mid-run KV eviction
  is the in-kernel local-model path. When the model lives upstream behind a hosted router,
  there is no local KV prefix to evict, so that evictor is a no-op by design (a quarantined
  result is still paged out before the model reads it).
- **The novelty is assembly, not invention.** A 29-claim prior-art audit scored 0/29 novel.
  Every primitive fak uses is established; the contribution is putting them in one in-process
  gate at the tool-call boundary. See [`CLAIMS.md`](../../../CLAIMS.md).

Any performance figure lives in the deeper docs and is witnessed there (the tuned ~4.1×,
not the naive multiple); this page makes no new benchmark claim and no market-adoption
claim.

## Cross-references

- [Routers & gateways](../../integrations/routers.md) — the full categorical positioning,
  the surveyed-router table, and all three topologies in depth. The source this page
  summarizes.
- [Capability matrix across the category](matrix.md) — the one-glance table scoring fak
  against gateways, guardrails libraries, and inference servers at once.
- [fak + LiteLLM](../../integrations/litellm.md) — the flagship router/proxy integration,
  with the three topologies in full and the honest stub status for multi-backend dispatch.
- [Compatibility matrix](../../integrations/compatibility-matrix.md) — OpenRouter, Together,
  Groq, Fireworks and 40 more, each with its wire and the exact repoint key.
- [Objections & one-line answers](../objections.md) — item 5 is the thread-ready version of
  "another gateway? I already have a router."
- [Claims ledger](../../../CLAIMS.md) — shipped vs stub, claim by claim.

## Verify

```
test -f docs/adoption/compare/vs-routers.md          # this artifact exists
fak score seo                                        # new doc does not red the SEO scorecard
```

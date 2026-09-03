---
title: "Capability matrix: fak vs guardrails libraries, API gateways & inference servers"
description: "A one-glance, sourced capability table across the whole category — fak against guardrails/output-validation libraries, LLM API gateways and routers, and vLLM/SGLang-class inference servers — scored yes/partial/no on the six things fak claims, with an honest note on every cell and fak's own gaps named."
slug: fak-capability-matrix
keywords:
  - fak vs vLLM
  - fak vs guardrails
  - agent kernel comparison
  - default-deny capability gate
  - LLM gateway comparison
  - prompt injection quarantine
  - capability matrix
date: 2026-07-03
---

# Capability matrix: fak across the whole category

This is the one-page table to screenshot before a build-vs-buy discussion. It scores
fak against the three neighbouring product categories people already run, on the six
capabilities fak actually claims. It is deliberately compact. Where an existing fak doc
already argues a cell in depth, this page links there rather than restating it.

The categories are different jobs, so most cells are honestly "no" for a reason, not a
knock. A guardrails library validates text. A gateway routes requests. An inference
server serves tokens. fak governs the tool-call boundary and fronts any of them. The
useful question is not "which is best" but "which layer owns this capability" — and where
fak's own coverage is partial, the note says so.

## The category columns

- **Guardrails / output-validation libraries** — NeMo Guardrails, Guardrails AI, Llama
  Guard and similar. They screen model input and output against configured rails or
  validators.
- **LLM API gateways & routers** — LiteLLM, OpenRouter, Portkey, Kong AI Gateway and
  similar. They sit on the request path and route, key-manage, rate-limit, and (some) run
  guardrail plugins.
- **Inference servers** — vLLM, SGLang, llama.cpp, TensorRT-LLM and similar. They turn a
  model plus a prompt into tokens, fast.

## The matrix

| Capability | fak | Guardrails libs | API gateways / routers | Inference servers |
|---|---|---|---|---|
| Default-deny capability gate on the **tool-call** path (fails closed regardless of the model) | Yes [1] | Partial [5] | No [8] | No [11] |
| Prompt-injection **result quarantine** — poisoned tool output held out of context by structure | Yes [2] | Partial [6] | No [9] | No [11] |
| **Addressable, bit-exact mid-run KV eviction** (`max\|Δ\| = 0`) | Yes [3] | No [7] | No [10] | Partial [12] |
| **Commit-level verify** — a false "done" refused from git evidence, not the agent's word | Yes [4] | No [7] | No [10] | No [11] |
| **Structured refusal** from a closed reason vocabulary | Yes [4] | Partial [6] | Partial [9] | No [11] |
| **Single static binary**, drop-in via one base URL | Yes [1] | No [7] | Partial [8] | Partial [13] |

Legend: **Yes** the capability is a first-class, documented feature; **Partial** something
adjacent exists but with a material gap named in the note; **No** it is outside that
category's job.

## Notes and sources

Cells about fak cite the in-repo artifact that proves them. Cells about a category cite
that category's documented purpose; where a specific product's behaviour could not be
confirmed from primary docs, the note says `unverified` instead of guessing.

1. fak is a single static Go binary that drops in with one base-URL change and default-denies
   every tool call unless the policy allows it. The lock is a capability check, not a text
   classifier, so a tool that is off the allow-list cannot be called no matter what the model
   was told. See [`fak-vs-alternatives-comparison.md`](../../fak-vs-alternatives-comparison.md)
   and the [compatibility matrix](../../integrations/compatibility-matrix.md) (41 of 47
   surveyed harnesses drop in with one base-URL change).
2. Suspicious tool *results* are held out of context by structure (`QUARANTINE` verdict),
   not by a classifier that must catch the attack. See
   [`CLAIMS.md`](../../../CLAIMS.md) (result-admit / KV-quarantine bridge entries) and the
   [objections card](../objections.md) items 1 and 4.
3. The addressable KV cache evicts one span mid-run and leaves the cache bit-for-bit
   identical, verified at `max|Δ| = 0` by a green test rather than asserted. Honest fence:
   this is the in-kernel local-model (`--gguf`) path; on a proxy or subscription seat the
   model lives upstream, so there is no local KV to evict and the evictor is a no-op by
    design (documented in the [Claude integration guide](../../integrations/claude.md),
   "Current limits on the subscription seat").
4. Commit-level verify and structured refusal are the DOS trust substrate: a claimed "done"
   is refused from git evidence, and refusals carry a reason from a closed vocabulary rather
   than free text. See [`CLAIMS.md`](../../../CLAIMS.md) and the
   [objections card](../objections.md).
5. Guardrails libraries can gate actions to a degree — for example NeMo Guardrails
   execution and dialog rails can constrain which actions a flow runs. The gap: that gate is
   flow- and LLM-mediated, so it is not a fail-closed capability lock that holds independently
   of what the model decides. Basis: the category's own docs describe rails as configured,
   model-driven flows ([NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails),
   [Guardrails AI](https://www.guardrailsai.com/docs)).
6. Output validators and rails can block or redact model output, and a rail violation is a
   structured event. The gap versus fak: the unit of control is text content, not a tool
   *result* held out of context by structure, and there is no shared closed refusal-reason
   vocabulary spanning a fleet. Basis: category docs (as in note 5).
7. Out of scope for a text-validation library: no KV-cache surface, no git-evidence verify,
   and these ship as Python libraries rather than a single static binary. Basis: category docs
   (as in note 5).
8. Gateways and routers add real request-path controls (keys, rate limits, some guardrail
   plugins) but they route *requests*; they do not adjudicate individual tool calls as a
   default-deny capability lock. On packaging, the category is mixed: Kong AI Gateway ships as
   a gateway binary, LiteLLM is a Python proxy, and OpenRouter is a hosted service — hence
   Partial on "single binary". Basis: [LiteLLM](https://docs.litellm.ai/),
   [OpenRouter](https://openrouter.ai/docs), [Kong AI Gateway](https://docs.konghq.com/gateway/latest/ai-gateway/).
   fak complements this layer rather than replacing it (see the
   [objections card](../objections.md) item 5).
9. Some gateways return structured policy errors and can run guardrail plugins, so
   result-side blocking and structured refusal are Partial. The gap: no tool-call capability
   lock and no git-evidence verify. Basis: category docs (as in note 8).
10. Request routers do not expose a per-span KV-cache eviction surface or a commit-level
    verify. Basis: category docs (as in note 8).
11. Adjudicating tool calls, quarantining results, verifying a claimed "done", and emitting a
    closed-vocabulary refusal are outside a token engine's job — an inference server serves
    tokens. This is the point of the "front it, don't replace it" framing in
    [`fak-vs-alternatives-comparison.md`](../../fak-vs-alternatives-comparison.md).
12. Inference servers do evict KV cache: vLLM Automatic Prefix Caching and SGLang
    RadixAttention keep and drop prefix blocks, typically LRU under memory pressure. The gap
    versus fak's cell: that is throughput-driven block eviction, not content-addressable
    mid-run eviction of a specific span with a proven `max|Δ| = 0`. Basis:
    [vLLM docs](https://docs.vllm.ai/en/stable/), [SGLang docs](https://docs.sglang.ai/), and
    the per-instance-vs-cross-worker analysis in
    [`fak-vs-alternatives-comparison.md`](../../fak-vs-alternatives-comparison.md).
13. Packaging is mixed: llama.cpp's `llama-server` is a single compiled binary, while vLLM
    and SGLang are Python packages with heavier dependency stacks — hence Partial. Basis:
    [vLLM docs](https://docs.vllm.ai/en/stable/), [SGLang docs](https://docs.sglang.ai/),
    [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Honest scope

No market-adoption claim is made here. The comparison is capability-by-capability, and the
categories are complements: fak is designed to sit in front of a gateway or an inference
server, not to replace it. fak's own gaps are on the table — the KV evictor is a no-op on
a proxy/subscription seat (note 3), and the novelty is assembly, not invention (a 29-claim
prior-art audit scored 0/29 novel; see [`CLAIMS.md`](../../../CLAIMS.md)). Cache and
performance figures live in the deeper docs and are witnessed numbers there (the tuned
~4.1×, not the naive multiple); this page makes no new benchmark claim.

## Where the depth lives

- [fak vs vLLM, SGLang & provider KV caching](../../fak-vs-alternatives-comparison.md) —
  the long-form infrastructure comparison with the cross-worker numbers.
- [Compatibility matrix](../../integrations/compatibility-matrix.md) — the 47-target
  "can it repoint a base URL" reference behind the drop-in claim.
- [Objections & one-line answers](../objections.md) — the thread-ready rebuttals for each
  cell above.
- [Model × backend coverage matrix](../../coverage-matrix.md) — the internal support grid
  (a different matrix: which model runs on which backend, not this category comparison).

## Verify

```
test -f docs/adoption/compare/matrix.md              # this artifact exists
fak score seo                                        # new doc does not red the SEO scorecard
```

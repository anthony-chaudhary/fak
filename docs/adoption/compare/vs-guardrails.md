---
title: "fak vs a guardrails library: an honest side-by-side"
description: "A fair, sourced comparison of fak against the guardrails class (Guardrails AI, NeMo Guardrails, Llama Guard). They recognize and classify content and fail open; fak is a default-deny capability gate on the tool-call path and fails closed. Where each fits, where they overlap, and why they are complements you run together."
slug: fak-vs-guardrails
keywords:
  - fak vs guardrails
  - Guardrails AI comparison
  - NeMo Guardrails comparison
  - Llama Guard comparison
  - default-deny capability gate
  - fail-open vs fail-closed
  - prompt injection quarantine
  - tool-call boundary
date: 2026-07-03
---

# fak vs a guardrails library

The first question anyone asks about fak is "how is this different from Guardrails AI
(or NeMo Guardrails, or Llama Guard)?" It is a fair question, because on the surface all
four sit near the model and all four are pitched as safety. The honest answer is that they
do different jobs and are stronger together than apart. A guardrails library reads content
and decides whether it looks acceptable. fak decides whether an action is allowed to run at
all. This page draws that line without a strawman: each tool's real strength is stated
first, and fak's own gaps are on the table.

If you want the wider category table (guardrails plus gateways plus inference servers, all
scored on the same six capabilities), that is the [capability matrix](matrix.md). This page
is the focused one-on-one with the guardrails class.

## The spine: fail-open recognizer vs fail-closed gate

A guardrails library is a **recognizer**. It inspects text — a prompt, a model response, a
retrieved passage — and classifies it against rules, validators, or a safety taxonomy. Its
protection is only as good as its ability to catch the bad thing. When an attacker phrases
the payload in a way the classifier does not recognize, the content passes. That is the
fail-open property, and it is inherent to any approach whose control unit is content: the
default outcome of a miss is that the text flows through.

fak is a **gate**. The control unit is the tool call, not the text. Every tool call is
default-denied unless the policy allows the capability, so a lever that is off the allow-list
cannot fire no matter what the model was convinced to emit. A miss does not open the door,
because there is no recognition step to miss. That is the fail-closed property. fak also
ships a result *detector*, and that detector is roughly 100% evadable by design — but it is
explicitly not the load-bearing part. The floor is the capability lock plus structural
containment (a suspicious tool result is held out of context by a `QUARANTINE` verdict, not
by catching the attack). See [`CLAIMS.md`](../../../CLAIMS.md) (the result-admit / quarantine
entries) and [policy in the kernel](../../explainers/policy-in-the-kernel.md).

Neither posture makes the other pointless. A recognizer catches bad *content* a gate does
not look at (a toxic completion, a leaked secret in a response, an out-of-schema payload). A
gate stops a bad *action* a recognizer cannot see (a destructive tool call the model was
tricked into proposing). The next sections give each guardrails tool its due and then place
fak against them.

## What each guardrails tool is for

**Guardrails AI.** A Python framework that wraps an LLM call in a "Guard" and runs
validators over the input and the structured output — type and schema checks, plus a hub of
validators for things like PII, toxicity, competitor mentions, and topic restriction. It can
re-ask or fix output that fails a validator. Its real strength is output *correctness and
shape*: if you need a model response to conform to a schema or to be scrubbed of a class of
content before your code consumes it, this is squarely the tool for that. Basis:
[Guardrails AI docs](https://www.guardrailsai.com/docs) and the
[validator hub](https://hub.guardrailsai.com/).

**NeMo Guardrails.** NVIDIA's open-source toolkit for adding programmable rails to a
conversational system, configured in the Colang modeling language. It supports several rail
types — input, dialog, retrieval, and output rails — so you can constrain what topics a bot
engages, shape the dialog flow, and screen responses. Its real strength is *conversational
control*: keeping an assistant on-topic and on-policy across a multi-turn dialog is what it
is built for, and the dialog-rail model is more expressive than a single content filter.
Basis: [NeMo Guardrails repo and docs](https://github.com/NVIDIA/NeMo-Guardrails).

**Llama Guard.** Meta's LLM-based input-output safeguard: a fine-tuned Llama model that
classifies a prompt or a response as safe or unsafe against a taxonomy of hazard categories.
Its real strength is *content safety classification* — it is a purpose-built, open-weights
moderation model you can run yourself instead of calling a hosted moderation API, and it
generalizes across phrasings better than a keyword filter. Basis: the
[Llama Guard model card and Purple Llama project](https://github.com/meta-llama/PurpleLlama).

These are genuine, useful tools. If the risk you are managing is "the model might say or
emit something harmful or malformed," a guardrails library is the right layer and fak does
not replace it.

## Where they overlap

There is one honest overlap. NeMo Guardrails' execution/dialog rails can constrain which
actions a flow is allowed to invoke, which sounds like fak's capability gate. And fak ships
a result detector that classifies tool output, which sounds like a guardrails validator. So
the surfaces touch at two points: action-constraint and content-classification. The
difference is which side owns the guarantee. NeMo's action constraint is expressed in a
model-mediated flow, so it holds to the extent the flow logic and the LLM behave; fak's is a
default-deny check in the call path that holds regardless of what the model decides. And
fak's content classifier is the deliberately non-load-bearing bonus, whereas a guardrails
library's classifier is its primary, and much more developed, product.

## Side-by-side

| Dimension | Guardrails class (Guardrails AI / NeMo / Llama Guard) | fak |
|---|---|---|
| Primary unit of control | Content — a prompt, response, or retrieved passage | The tool call (a capability on the call path) |
| Failure mode on a miss | Fail-open: unrecognized content passes | Fail-closed: an un-allowed tool call cannot fire |
| Content classification / PII / toxicity / schema | Strong — this is the core job | Weak; a bonus detector, ~100% evadable by design |
| Stops a destructive action the model was tricked into | No — it does not adjudicate tool calls | Yes — default-deny capability lock |
| Poisoned tool *result* held out of context by structure | No | Yes — `QUARANTINE` verdict, not a classifier catch |
| Conversational / dialog-flow control | Yes (NeMo dialog rails) | No — not fak's job |
| Packaging | Python libraries / a model you host | Single static Go binary, drop-in via one base URL |
| Structured refusal from a closed reason vocabulary | Partial (rail-violation events) | Yes (DOS refusal vocabulary) |

Rows about the guardrails class cite that category's documented purpose (the primary docs
linked above); where a specific product's behaviour could not be confirmed from primary
docs, the safe reading is `unverified` rather than a guess. This page makes no numeric or
benchmark claim about any competitor.

## These are complements — fak can front one

The useful architecture is not fak *or* a guardrails library. It is both, in series. You run
a guardrails library for what it is good at (validate and shape the *content* the model
produces or consumes) and put `fak serve` on the tool-call path for what it is good at
(default-deny the *actions*, quarantine poisoned results, verify a claimed "done" from git
evidence). fak is designed to sit in front of a model endpoint with one base-URL change, so
a guardrails validator on your response objects and fak on your tool boundary do not compete
for the same seat. A prompt-injection attack that slips past the content classifier still
hits fak's capability lock, and a malformed response that fak's action gate never inspects
still hits your output validator. See [tool call is a syscall](../../explainers/tool-call-is-a-syscall.md)
for the boundary fak governs and [`SECURITY.md`](../../../SECURITY.md) for the capability
floor.

## fak's own gaps, named

- fak is a poor content classifier and is not trying to be a good one. For PII redaction,
  toxicity scoring, or output-schema validation, a guardrails library is better and fak
  should defer to it. fak's own detector is a labeled, evadable bonus.
- fak governs the tool-call boundary. If your agent has no tool calls and the only risk is
  what the model *says*, fak's capability floor has little to grab and a guardrails library
  is the more relevant layer.
- The novelty here is assembly, not invention. A 29-claim prior-art audit scored 0/29 novel;
  every primitive is established (see [`CLAIMS.md`](../../../CLAIMS.md)). The contribution is
  putting the default-deny gate at the tool-call checkpoint, not a new algorithm.

## Honest scope

No market-adoption claim is made here, and no benchmark number is attached to any competitor.
Each guardrails tool's strength is stated from its own primary docs, and fak's fail-closed
claim traces to the capability floor in [`CLAIMS.md`](../../../CLAIMS.md) and
[`SECURITY.md`](../../../SECURITY.md). This is dimension **D — Positioning & comparison** of
the [concept-popularization epic](../../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).

## Where the depth lives

- [Capability matrix across the category](matrix.md) — the same comparison widened to
  gateways and inference servers, scored cell-by-cell.
- [Objections & one-line answers](../objections.md) — item 7 ("how is this different from a
  firewall or a guardrails library?") is the thread-ready version of this page.
- [fak vs vLLM, SGLang & provider KV caching](../../fak-vs-alternatives-comparison.md) — the
  infrastructure-layer comparison, for the caching and throughput angle.
- [Compatibility matrix](../../integrations/compatibility-matrix.md) — the drop-in reference
  behind "front one guardrails setup with one base-URL change".

## Verify

```
test -f docs/adoption/compare/vs-guardrails.md       # this artifact exists
fak score seo                                        # new doc does not red the SEO scorecard
```

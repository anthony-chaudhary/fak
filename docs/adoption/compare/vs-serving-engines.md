---
title: "fak vs vLLM / SGLang: a different boundary, not a race"
description: "The honest positioning against inference servers. vLLM and SGLang own raw throughput — tokens per second — and fak does not try to beat them there. fak owns the governance band those engines leave open: a default-deny capability gate on the tool-call path, structured refusals, and result quarantine, all as one static binary you put in front. The recommended move is to run both: keep vLLM/SGLang for throughput and front it with `fak serve`."
slug: fak-vs-serving-engines
keywords:
  - fak vs vLLM
  - fak vs SGLang
  - front vLLM with fak
  - inference server comparison
  - agent tool-call governance
  - default-deny capability gate
  - single static binary
date: 2026-07-05
---

# fak vs vLLM / SGLang

If you already run vLLM or SGLang, the fair first question is "isn't fak competing with my
serving engine?" Short answer: no, and you should keep the engine. A serving engine answers
*how do I turn a model and a prompt into tokens, as fast and as cheaply as possible?* fak
answers a different question that sits one layer up: *should this tool call be allowed to
run at all, and what happens to the result that comes back?* Those are different boundaries,
so the useful move is not to pick one — it is to run vLLM/SGLang for throughput and put
`fak serve` in front of it for governance.

This page is the focused inference-server version of the whole-category
[capability matrix](matrix.md), and the popularization cut of the deeper infrastructure
comparison in [fak vs vLLM, SGLang & provider KV caching](../../fak-vs-alternatives-comparison.md).
It is part of Dimension D of the
[concept-popularization epic](../../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).

## What vLLM and SGLang are for

An inference server serves tokens. Given a model and a prompt, it prefills, decodes, and
streams tokens back — and modern engines are extremely good at it. vLLM's continuous
batching and PagedAttention, SGLang's RadixAttention, paged KV caches, and multi-tenant
schedulers exist to push tokens/sec up and cost/token down under real concurrent load. That
is hard, valuable engineering, and it is the whole job of that layer.

fak does not out-serve them, and this page makes no claim that it does. fak ships an
in-kernel model path (`fak guard --gguf ...`), but the repo is explicit that it is a
**bit-exact correctness reference, not a tuned server**: no continuous batching, no paged
attention, no multi-tenant scheduler. For real serving you run a real serving engine. If
you need raw tokens/sec, vLLM or SGLang is the answer, full stop.

## What fak is for

fak governs the **tool call**, not the token stream. Every tool call the model proposes
crosses a default-deny [capability floor](../../explainers/policy-in-the-kernel.md) before
it runs:

- **Default-deny by capability, not by classifier.** A tool that is off the allow-list
  cannot be called no matter what the model was told; a destructive argument (`rm -rf`, a
  write into `.git/`) is refused. The lock is a capability check, so it fails closed
  regardless of which engine is behind it — it does not depend on recognizing an attack.
- **Structured refusals.** A refusal carries a reason from a closed vocabulary
  (`DEFAULT_DENY`, `POLICY_BLOCK`, `SECRET_EXFIL`, …) rather than free text, so a fleet can
  route on it.
- **Result quarantine.** A suspicious tool *result* is held out of the model's context by
  structure, so a poisoned fetched page never becomes the model's next set of instructions.
- **An audit ledger.** Each run emits a per-call verdict summary, e.g.
  `131 kernel decisions; 121 allowed / 5 denied / 2 repaired / 0 quarantined / 3 deferred`.

None of this is what a token engine does, and none of it is something a token engine was
built to do. This is the band the serving engine leaves open — the governance and
containment boundary at the point where the model asks to *act*, not where it asks to
generate.

## Why it is a different boundary, not a race

The two layers do not overlap, so a head-to-head "who is faster" framing is the wrong lens.
The engine owns the token path; fak owns the tool-call path. Neither can do the other's
job. A serving engine has no notion of "this tool call is off the allow-list" or "hold this
result out of context"; fak has no ambition to schedule batches across GPUs.

The one place the layers touch is cost, and fak is honest about it: putting an adjudicator
on the call path is not free. The measured guard tax is about **362 ns per decision**
(in-process, Apple M3 Pro — no network hop), and the gateway/adjudication overhead converges
toward roughly **3% at saturation**. That is the price of the boundary, and it is the shape
of the product rather than a bug to hide: fak buys you governance for a few percent of
throughput, it does not buy you throughput.

## Single binary vs a multi-process governance stack

What fak *does* collapse is the operational surface of governing that boundary. The usual
governed-serving stack around an engine is several cooperating processes — a reverse
proxy / gateway, a policy layer, a result-quarantine service, and an audit sidecar — each
its own deploy, port, and config. fak carries the same four responsibilities as in-process
stages of one static Go binary. You add flags, not components.

![Two columns with the same four responsibilities labeled identically on both sides — reverse proxy / gateway, policy / capability floor, result quarantine, audit journal. Left, the usual governed-serving stack runs them as four separate processes; right, one static fak binary holds the same four as in-process stages. A blue arrow reads "four processes → one": you add flags, not components](../diagrams/single-binary.svg)

That is the honest boundary of the drop-in claim. fak does not make your tokens faster, but
it replaces a multi-process governance stack with one process that does the same governing
job — and drops in with one base-URL change.

## Use both: front vLLM/SGLang with `fak serve`

Because vLLM and SGLang expose OpenAI-compatible endpoints, wiring fak in front of them is a
base-URL change. fak owns the tool-call floor for everything; the engine does what it is
good at behind it:

```
agent / harness ──► fak serve ──► vLLM / SGLang (OpenAI-compatible) ──► GPU
                    │
                    └─ default-deny capability floor on every tool call
```

```bash
# 1. Run your engine as usual (vLLM shown; SGLang is the same shape on its own port).
vllm serve <model> --port 8000            # OpenAI-compatible server at http://127.0.0.1:8000/v1

# 2. Put fak in front and point your agent at fak instead of the engine.
fak serve --addr 127.0.0.1:8080 --provider openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model <model> \
  --policy floor.json
```

Your agent talks to `fak serve`; fak adjudicates each tool call against the capability
floor, then forwards the request to vLLM/SGLang, which prefills and decodes at full speed.
You keep every bit of the engine's throughput work — batching, paged attention, prefix
caching — and you gain a default-deny capability floor, structured refusals, result
quarantine, and an audit ledger the engine does not express. Nothing about the engine
changes; you added a boundary in front of it.

## Where fak is weaker, honestly

This is a complement page, not a takedown, so the gaps belong on it:

- **fak does not out-serve an engine.** vLLM's and SGLang's throughput, batching, and
  scheduling are their whole job and they do it well. fak's in-kernel model path is a
  correctness reference, not a competitor to that layer, and this page quotes no tokens/sec
  figure against them.
- **The KV-eviction win is local-only.** fak's addressable, bit-exact mid-run KV eviction is
  the in-kernel local-model path. When the model lives behind vLLM/SGLang, the engine owns
  the KV cache, so that evictor is a no-op by design — a quarantined result is still paged
  out before the model reads it. (Both fak and the engines do prefix caching; the honest
  cross-worker delta is small and lives in the
  [long-form comparison](../../fak-vs-alternatives-comparison.md), not here.)
- **The novelty is assembly, not invention.** A 29-claim prior-art audit scored **0/29
  novel**. Every primitive fak uses — default-deny capability gates, KV reuse, audit
  journals — is established; the contribution is putting them in one in-process gate at the
  tool-call boundary. See [`CLAIMS.md`](../../../CLAIMS.md).

Any performance figure lives in the deeper docs and is witnessed there (the tuned ~4.1×
warm-cache reuse number, never a naive multiple). This page makes no new benchmark claim,
no tokens/sec claim against a serving engine, and no market-adoption claim.

## Cross-references

- [Capability matrix across the category](matrix.md) — the one-glance table scoring fak
  against inference servers, gateways, and guardrails libraries at once (inference-server
  column and notes 11–13).
- [fak vs vLLM, SGLang & provider KV caching](../../fak-vs-alternatives-comparison.md) — the
  long-form infrastructure comparison with the cross-worker KV numbers.
- [What fak is not](../../explainers/what-fak-is-not.md) — the "governance band, not
  throughput" framing and the four-processes-to-one operational-surface argument this page
  summarizes.
- [Objections & one-line answers](../objections.md) — item 3 is the thread-ready version of
  "why not just use vLLM or SGLang?"
- [Is fak just a firewall?](vs-firewall.md) and [fak vs an API gateway / LLM
  router](vs-routers.md) — the sibling boundary comparisons.
- [Claims ledger](../../../CLAIMS.md) — shipped vs stub, claim by claim.

## Verify

```
test -f docs/adoption/compare/vs-serving-engines.md   # this artifact exists
fak score seo                                         # new doc does not red the SEO scorecard
```

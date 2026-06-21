---
title: "How the KV Cache Changes as Agentic Context Grows"
description: "Appending tool output does not break the KV cache — that is the easy case. Hit rate erodes from latency-eviction and head-mutation, and it matters far more for agents than chat because of the input:output ratio."
slug: kv-cache-agentic-context
keywords:
  - KV cache
  - prefix caching
  - agentic context
  - tool use
  - prompt caching
  - cache hit rate
  - input output ratio
  - LLM inference
date: 2026-06-17
---

# How the KV Cache Changes as Agentic Context Grows

**Short answer:** Appending a tool result to the conversation does *not*, by itself, break the KV cache — prefix caching is built for append-only growth and handles it well. What actually erodes the hit rate in agent loops is (1) **eviction during tool latency** (your cached prefix is thrown out while a tool runs for seconds to minutes) and (2) **head-mutation** — any change ahead of the stable part of the context (summarization, an injected timestamp, a changing tool list, reordering, unstable serialization). A changed tool *result* does not silently serve a stale answer either: the new result tokens are literally different, so the prefix matches only up to that point and the suffix is recomputed. The reason it matters so much more for agents than for chat is the **input:output ratio** — agents re-send a huge transcript to read against a few generated tokens, so the same cache discounts most of the bill.

## What is a KV cache, and why is reuse always a prefix?

A transformer caches each token's attention **Key** and **Value** vectors so it never re-reads earlier tokens while decoding. Generation has two phases: **prefill** (process the whole prompt at once, build KV for every token — the expensive part for long contexts) and **decode** (emit one token at a time, each attending over all cached KV).

The load-bearing fact is that attention is **causal**: token *i*'s KV depends only on tokens *0..i*. So any two requests that share a token-identical prefix produce identical KV for that prefix — and the cache can be reused up to the **first token that differs**. From that token on, everything is invalidated and must be re-prefilled. This is exactly what "prefix caching" exploits (vLLM Automatic Prefix Caching, SGLang RadixAttention, and the prompt-caching APIs from major providers).

Key word: **prefix**. Reuse is only ever a contiguous run from token 0. A change at position *N* costs you everything at or after *N*.

## Does appending tool output break the KV cache?

No — that is the *easy* case, and the common mental model is wrong here. Walk an agent loop:

```
[system + tool defs][user]
   → assistant: think + tool_call_1
[tool_result_1]
   → assistant: think + tool_call_2
[tool_result_2]
   → assistant: final
```

Each model call re-sends the growing message list. If the loop is **strictly append-only**, call *k+1*'s prompt has call *k*'s prompt as an exact prefix. A correct prefix cache prefills only the *delta* (the previous assistant turn plus the new tool result) and reuses everything before it. That is the happy path, and it works.

"Breaks the cache" becomes the right verb only when reuse is lost, because of the **quadratic blow-up**. With *T* turns over a growing context, *no* prefix reuse means re-prefilling the entire history every call → total prefill cost proportional to **T²**. Perfect prefix reuse makes it proportional to **T** (only the delta each turn). A broken cache doesn't cost a constant — it turns a linear loop into a quadratic one in both latency and dollars.

## What actually erodes the cache hit rate in agent loops?

1. **Eviction during tool latency (the dominant practical cause).** KV memory is finite. While a tool runs — a shell command, a web fetch, a sub-agent, seconds to minutes — other traffic evicts your prefix under LRU, and you return to a cold cache. This is why hosted prompt caches have a short TTL (commonly ~5 minutes); a tool slower than the TTL guarantees a miss.

2. **Head-mutation — the structural killer.** Anything that changes the context *ahead* of the stable part invalidates everything after it:
   - **Summarization / compaction** of old turns rewrites the head.
   - **Dynamic injection** at the top — a current timestamp, retrieved memories, a changing tool list, per-turn reminders — makes the prefix differ every call → near-zero reuse.
   - **Reordering or dropping** old turns invalidates from the edit point on.
   - **Non-deterministic serialization** — JSON with unstable key order, or varying pretty-printing — silently changes tokens and breaks the match.

3. **Bad cache-breakpoint placement.** Caching only through the system prompt leaves the growing conversation uncached and re-prefilled every turn — the T² blow-up returns even though you "use caching." Move a breakpoint to the **end of the message history** each turn.

4. **Bloat from large or variable tool outputs.** A big file read or search dump must be prefilled once and then stored; it inflates memory (more eviction pressure) and lengthens every later prefill. It doesn't break the prefix, but it shortens how long the prefix survives.

A real, decisive illustration of how much head placement dominates: in one multi-tenant case, moving a single per-request UUID from the head to the **tail** of the prompt took the cache hit rate from **0.3% to 87%** — same content, same model, just stop mutating the head.

## Same tool call, changed file: does the loop reuse a stale answer?

A common worry: a tool call `read(f)` with *identical* arguments — will the loop reuse a stale result? In a normal append-only loop, **no, and that is the point.** The tool actually runs again, and if the world changed (the file went from A to B), the **appended result tokens are literally different**. The cached prefix matches only up to that result; from the changed token onward, the suffix **re-prefills**. You pay recompute — you do *not* silently serve the old answer. The divergence is visible, not hidden.

**When it *does* go silently wrong:** only if a *result cache* is keyed on the **call arguments alone**. Then an identical `read(f)` replays the old result A and the tool never re-runs, so the loop acts on stale data with no error or re-prefill to signal it.

**The fix:** key any result reuse on a **content version** — a hash, mtime, or etag of the underlying source — not just on the call arguments. Identical inputs do not imply an identical answer.

## Why does the KV cache matter far more for agents than for chat?

Prefix caching only ever discounts the **input (prefill)** side of a request. How much that is worth is set by the workload's **input:output ratio** — how many tokens you re-send to read versus how many you generate.

- **Chat** — short prompts and comparable-length replies, roughly **2:1**. Input is a small slice of the bill, so even a perfect cache deletes almost nothing.
- **Agentic** — the entire growing transcript is re-sent every turn against a handful of output tokens. Measured input:output ratios run around **239:1** machine-wide, with long-context research agents exceeding **1000:1**. Input *is* the bill — so the same cache deletes most of it.

The share of total spend a prefix cache can delete tracks the ratio directly:

| Workload | Input:output ratio | Share of spend a prefix cache can delete |
|---|---:|---:|
| Chat | ≈ 2:1 | ~0.3% |
| Research agent | ≈ 258:1 | ~38.9% |
| Always-on fleet | > 1000:1 | ~92.6% |

**The honest twist:** heavy agentic use barely dents the *percentage* hit rate. In one month of measured fleet traffic, cache-hit eroded only from **96.7% to 92.6%** as volume grew roughly 7×, with about **94%** of all ingested context still served from cache. The few points lost are exactly the head-mutations and evictions above. It matters anyway, because at 239:1 those few points sit on an enormous input base — the same hit rate is worth orders of magnitude more.

## How do you prove where the cache breaks?

The cheapest decisive method needs no GPU and no provider: **offline prefix-divergence analysis.** Log the exact prompt sent on each turn, tokenize it, and for each turn compute the longest common token prefix with the previous turn. An append-only loop shows reuse climbing toward 100%; a head mutation shows up as a sudden reuse cliff on the exact turn it happens.

```python
# Feed it JSONL: one {"turn": i, "tokens": [...]} per line.
import json, sys

def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i

prev = None
for line in sys.stdin:
    rec = json.loads(line); cur = rec["tokens"]
    if prev is None:
        print(f"turn {rec['turn']}: {len(cur)} tok (first, all cold)")
    else:
        m = lcp(prev, cur); reuse = m / len(cur) if cur else 0
        print(f"turn {rec['turn']}: {len(cur)} tok | reusable {m} ({reuse:0.1%}) | must-prefill {len(cur)-m}")
    prev = cur
```

For empirical confirmation on hosted models, log each provider's per-request cache accounting (cache-read vs cache-creation tokens) and plot the read-fraction per turn: a healthy append-only loop is mostly cache-read; a mutated one is mostly cache-creation. On self-hosted stacks, read the engine's prefix-cache-hit-rate counter directly, and watch **time-to-first-token versus context length** — flat TTFT as context grows means hits; rising TTFT means misses.

## How do you keep the hit rate high?

The highest-leverage fix is zero-infrastructure **cache-friendly prompt design**: keep the prefix stable and append-only. Put the static system prompt and tool definitions *first*; never place volatile content (timestamps, changing retrievals) ahead of stable content; use **deterministic serialization** (stable JSON key order); **append, don't edit**. To "remove" a tool, mask its logits rather than deleting it from the schema (deleting changes the prefix). To shrink state, externalize it to a file or scratchpad referenced by a stable handle instead of inflating the context.

Beyond prompt design: place explicit cache breakpoints at the end of the growing history (not just the system prompt) and match the cache TTL to your tool latency; use tree-structured prefix caches (RadixAttention) for parallel tool calls and fan-out from a shared prefix; and use KV offloading / hierarchical caches to survive slow tool calls without re-prefilling. Mid-context (non-prefix) reuse — recomputing only the small fraction of cross-attention that actually changes — is the active research edge for "a tool result in the middle invalidates everything after it."

## Frequently asked questions

**Does tool output break the KV cache?**
Not on its own. Append-only tool output is the easy case — a correct prefix cache reuses the whole prior context and prefills only the new turn. The cache erodes from latency-driven eviction and from head-mutation, not from appending.

**Why does a changed file cause a cache miss?**
Because the tool re-runs and returns a different result. The new result tokens differ from the cached stream, so the prefix matches only up to that point and the suffix is re-prefilled. The cost is visible recompute, not a silently stale answer.

**Can a cache silently serve a stale tool result?**
Only if a result cache is keyed on the call arguments alone, skipping re-execution. Key reuse on a content version (hash/mtime/etag) instead — identical inputs do not guarantee an identical answer.

**Why is KV caching worth more for agents than chatbots?**
Caching discounts input (prefill) tokens. Chat is roughly 2:1 input:output, so input is a small share of cost; agents run ~239:1 and higher because they re-send a large transcript each turn, so the same cache deletes most of the bill.

**Does heavy agentic use crater the hit rate?**
No — it erodes it a few points (for example, 96.7% to 92.6% as volume grew ~7×), with ~94% of context still cache-read. The percentage barely moves; the dollar impact is large only because it sits on a huge, input-heavy base.

---

*The mechanism (causal attention, prefix-only reuse, the T²-vs-T stakes) is standard transformer inference. The specific figures — the ~96.7%→92.6% hit-rate erosion, the 239:1 input:output ratio, the 0.3%→87% UUID-to-tail case, and the share-of-spend table — are observed measurements, not illustrative. Diagrams in the companion one-page PDF are schematic.*

**Related:** `agentic-serving-related-art.md` (private research companion) — the related-work map (where this mechanism sits vs the 2025–26 agentic-serving frontier, incl. the NVIDIA Dynamo mapping and the cross-agent correctness-gated-invalidation seam) · `FLEET-SWEEP-EXPLAINED.md` (private companion) — the cross-agent shared-cache measurement (the "result cache keyed on hash/mtime" point of §"Same tool call, changed file", measured at fleet scale).

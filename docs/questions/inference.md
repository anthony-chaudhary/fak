---
title: "Inference — the questions a newcomer actually asks"
description: "The question ledger for the inference axis: the questions a newcomer asks when they first try to follow a token through a model, each with a short answer and a pointer to the rung of the inference ladder that answers it in full."
slug: questions-inference
keywords:
  - inference questions
  - next-token prediction
  - why is the first token slow
  - prefill vs decode
  - KV cache explained
  - why is decode memory bound
  - quantization questions
  - speculative decoding
  - question loop
  - inference ladder
date: 2026-11-18
---

# Inference — the questions a newcomer actually asks

> **What this is.** The question ledger for the **inference** axis. Not a tutorial — a
> list of the questions a newcomer actually asks the first time they try to follow a
> token through a model, each with a short answer and a pointer to the rung of the
> [inference ladder](../explainers/inference/README.md) that answers it in full. The
> page *is* the questions; the answers are deliberately short, because the rung is
> where the real answer lives.

"Inference" here means one specific thing: **the loop that turns your prompt into the
next token, and everything done to make that loop fast without changing what it
computes.** A newcomer meets that sentence and immediately has questions — some
contrarian ("isn't this just autocomplete?"), some afraid ("why does the same prompt
give different answers?"), and some nobody has thought to ask ("why is decode
memory-bound and not compute-bound?"). This page collects them, labels them honestly,
and routes each one to the rung that closes it.

## How this page works

This page is the durable surface of the **question loop** — a super-loop-family member
whose whole job is to *ask* hard questions and *not* answer them. Two loops, in two
separate context windows, keep it honest:

- **The ASK loop** (role: *questioner*) writes the questions. Each tick it appends a
  handful of `open` rows to [`asked.jsonl`](asked.jsonl) and stops. It never proposes
  fixes, files tickets, or answers itself.
- **The NEXT-STEP loop** (role: *ticketer*) turns an unanswered question into work. Each
  tick it takes **one** `open` question, decides its fate, and — only if the question
  earns it — files **one** ticket.

**Why the two loops are separated.** A questioner that also holds the ticketing pen
slides into solution mode: it starts writing questions it already knows how to close, and
quietly stops asking the uncomfortable ones. The load-bearing rule is that **the
questioner must not be the one who decides the question is answered.** Reading the open
backlog is fine and necessary; *triaging* it is what corrupts the ASK role. A fresh
context per ASK tick keeps the provocations honest. The full schema, the four categories,
and the machine-checked ledger rules live in the [question-loop README](README.md).

The categories below are the ledger's own vocabulary — **CONTRARIAN**, **AFRAID**,
**UNASKED**, and **STEELMAN** — not a new taxonomy invented for this page.

## CONTRARIAN — questions that challenge what the repo claims is correct

### Isn't this just autocomplete?

Yes, in the literal sense, and the rung says so plainly. The model is a function from
token ids to a score per vocabulary entry, run in a loop; there is no plan being
consulted and no outline being followed. What makes it feel like more is that predicting
the next chunk well, across everything people write, requires absorbing grammar, facts,
and the shape of an argument as a side effect. The mechanism is small and completely
understandable; the behaviour that falls out of it is large, and nobody has a tidy
explanation for all of it.

→ full answer: [Level 1 — What is a token, and what is a model?](../explainers/inference/level-1-what-is-a-token.md)

### If it's just matrix multiplies, why does it need a GPU?

Because the arithmetic is not the hard part — the memory traffic is. A decode step reads
every weight to produce one token, and the arithmetic intensity works out to about 4
FLOP per byte for a 7B model at 4 bits, roughly 25 times below the ~100 FLOP/byte a
modern accelerator needs to be compute-bound. The arithmetic units finish long before the
memory bus does, and then they wait. A GPU is not there to do more multiplies; it is
there to move bytes fast. Prefill, which multiplies the same weights against many
positions, is compute-bound — so the same hardware is doing two different jobs.

→ full answer: [Level 4 — Prefill vs decode: why one model has two speeds](../explainers/inference/level-4-prefill-vs-decode.md)

### Isn't the KV cache just a memoization table?

No, and the difference is the whole design. A memoization table caches a pure function's
output; the KV cache caches a vector *at a position*, and RoPE bakes the position into
the stored key, so you cannot un-rotate it cleanly. That is why the cache keeps two
copies of K: `K` (post-RoPE) and `Kraw` (pre-RoPE), so a span can be repositioned by one
clean rotation, which is bit-exact to a fresh prefill, where composing two rotations
drifts ~1e-6 — enough to flip a greedy token. It is a first-class kernel-owned object
(`internal/model/kvcache.go:11`) with an absolute-position list and an eviction path, not
a lookup table.

→ full answer: [Level 4 — Prefill vs decode: why one model has two speeds](../explainers/inference/level-4-prefill-vs-decode.md)

### Isn't temperature just a hack?

It is a knob, but a principled one. Greedy decoding always takes the top token and is
completely predictable; sampling draws from the distribution in proportion to
probability, and temperature controls how adventurous that draw is — low concentrates on
the top scores, high spreads the probability out. The reason to want the dice is that
always taking the top token makes a model that is correct and dull, and sometimes stuck:
if the single best next token leads down a bad path, greedy has no way out. Temperature
is the parameter that decides how much of that escape hatch you want.

→ full answer: [Level 1 — What is a token, and what is a model?](../explainers/inference/level-1-what-is-a-token.md)

## AFRAID — the elephant in the room

### Why does the same prompt give different answers?

Because the default is not always greedy. Greedy decoding — take the highest score — is
deterministic: the same prompt always produces the same answer. Sampling rolls dice
weighted by the probabilities, so a token with a 40% chance comes up about 40% of the
time, and the same question produces different answers on different days. That is usually
what you want from something you are talking to, and usually not what you want from
something you are testing. Both modes are real and both are in the repo: the greedy pick
is `internal/model/kv.go:1652`, the dice are rolled in
`internal/agent/inkernel_sampling.go:14`.

→ full answer: [Level 1 — What is a token, and what is a model?](../explainers/inference/level-1-what-is-a-token.md)

### Why is the first token slow?

Because the first token is the *last* token of a whole-prompt pass, and the tokens after
it are each the *only* token of a one-token pass. Prefill ingests the whole prompt in one
wide pass — every position computed, the weights read once — so its cost grows with
prompt length; decode then makes one token at a time, and its cost is roughly flat per
token. A 2,000-token prompt takes roughly 400 ms to its first token and then roughly 15
ms per token after. Time to first token is dominated by prefill; inter-token latency is
dominated by model size.

→ full answer: [Level 4 — Prefill vs decode: why one model has two speeds](../explainers/inference/level-4-prefill-vs-decode.md)

### Does batching change the answer?

No. Batching changes *when* work is admitted, never *what* the work is — each token is
still computed by the same forward pass. What batching changes is arithmetic intensity:
serve 32 requests at once and the same weight read serves 32 tokens, so the ratio rises
from 4 to 128 FLOP/byte and crosses the balance point. The output distribution is
untouched. The cost is that per-request latency can rise, because a request admitted into
a busy batch shares bandwidth.

→ full answer: [Level 5 — What changes at scale](../explainers/inference/level-5-at-scale.md)

## UNASKED — questions nobody has thought to ask

### Why 128,000 vocabulary entries?

Because the vocabulary is the full list of chunks the model is allowed to produce, and
the tokenizer decides those chunks by counting: common words get their own token, rare or
long words get chopped into pieces that are themselves common. A modern model's list is
around 128,000 entries, so the output of one next-token decision is not one number — it
is about 128,000 numbers, one per possible next token. The LM head is the one place the
width changes, from `HiddenSize` to `VocabSize`. The model scores every one of those
chunks on every step and then throws almost all of them away.

→ full answer: [Level 1 — What is a token, and what is a model?](../explainers/inference/level-1-what-is-a-token.md)

### Why is decode memory-bound and not compute-bound?

Because the arithmetic intensity of a decode step is a division that cancels the model
size. Bytes read ≈ weights × bytes-per-weight; FLOPs ≈ 2 × weights × 1 token; divide and
the token count cancels, leaving `2 / bytes_per_weight` — about 4 FLOP/byte at 4 bits,
independent of how big the model is. A modern accelerator needs on the order of 100
FLOP/byte to be compute-bound, so decode sits roughly 25 times below the balance point.
The arithmetic units finish long before the memory bus does. Prefill is the opposite: the
same weights multiplied against 2,000 positions is 8,000 FLOP/byte, compute-bound by a
wide margin.

→ full answer: [Level 4 — Prefill vs decode: why one model has two speeds](../explainers/inference/level-4-prefill-vs-decode.md)

### Why does quantization not break it?

It does break it a little, and the rung is honest about that. Quantization stores each
weight in fewer bits, so a decode step reads fewer bytes and, if bandwidth-bound, runs
roughly proportionally faster — but rounding a weight to 4 bits loses information, the
outputs drift, and how far is an empirical question that is not uniform: some layers
tolerate aggressive quantization, others do not. The second cost people forget is the
dequantization step: the weights are stored packed, but the matmul needs real numbers, so
you unpack on the fly. The repo's own KV cache shows the shape of the trade — it
quantizes the hot path (attended K and V, read every token) and keeps full precision on
the cold path (`Kraw`, read on eviction), because a small error there compounds.

→ full answer: [Level 5 — What changes at scale](../explainers/inference/level-5-at-scale.md)

## STEELMAN — the strongest version of the objection

### Steelman: the model is thinking, not predicting — why reduce it to next-token prediction?

The steelman is real: to guess the next chunk of text well, across everything people
write, a model has to absorb an enormous amount about how language works — grammar,
facts, the shape of an argument, the way a story usually goes. A model that is genuinely
good at "what comes next" has, as a side effect, picked up a great deal of that, so when
it answers a question it is not retrieving an answer from a drawer; it is producing the
continuation a very well-read predictor would produce. The repo's counter is not that the
behaviour is unimpressive — it is that the mechanism is small and completely
understandable, and it is worth holding both halves at once. The machine is next-token
prediction; the behaviour that falls out of it is large, and nobody has a tidy
explanation for all of it.

→ full answer: [Level 1 — What is a token, and what is a model?](../explainers/inference/level-1-what-is-a-token.md)

### Steelman: a huge context window is strictly better — why not make it enormous?

The steelman is that more context means the model can see more, and there is no obvious
downside. The counter is memory: the KV cache grows linearly with sequence length, and
its size is `2 × layers × NumKVHeads × HeadDim × positions × bytes_per_element`. For a
32-layer, 8-KV-head, 128-head-dim model at 32k positions in 16-bit, that is 4.29 GB —
larger than the 3.5 GB of 4-bit weights for a 7B model. At 100k+ tokens it dominates
memory, and eviction, compression, and summarization all reduce it while risking the one
token the answer depended on. A huge window is not free; it is paid for in memory, and
the honest open edge is keeping that memory bounded without losing recall.

→ full answer: [Level 4 — Prefill vs decode: why one model has two speeds](../explainers/inference/level-4-prefill-vs-decode.md)

### Steelman: speculative decoding must change the output — how can guessing not change the answer?

The steelman is that a draft model proposing tokens is a second model in the loop, so the
output should be the draft's, not the target's. The counter is the verification step: the
draft proposes `k` tokens, the target runs one forward pass over the prompt plus all `k`
drafts, and the longest prefix where the draft matches the target is accepted; the rest
is rolled back. A correct speculative decoder produces *exactly* the same distribution as
the target model alone — the verification guarantees it. It is a latency win, not a
throughput win: it makes a single sequence finish faster, but under a full batch the
target is already saturated and there is no idle bandwidth for the draft to exploit.

→ full answer: [Level 5 — What changes at scale](../explainers/inference/level-5-at-scale.md)

## Questions still open

These are questions the repo genuinely cannot answer yet. They are marked **open** in the
ledger, and each names what would close it.

- **How do you keep KV memory bounded at very long context without losing recall?** The
  cache grows linearly with sequence length; at 100k+ tokens it dominates memory.
  Eviction, compression, and summarization all reduce it — and all risk dropping the one
  token the answer depended on. *What would close it:* a task-success benchmark at long
  context showing a bounded-memory policy matching full-cache quality on tasks that
  require recalling a specific early detail.
- **Do sparse and linear attention hybrids preserve quality at scale?** Attention is
  quadratic in sequence length; sparse and linear variants make it cheaper by attending
  to less. Whether they preserve quality at frontier scale — or only at the sizes where
  they have been tested — is contested. *What would close it:* a controlled comparison at
  matched parameter count and matched training compute, evaluated on long-context recall,
  not just perplexity.
- **How far can quantization go before quality breaks?** 8-bit is broadly safe; 4-bit is
  usually fine. Below that, results diverge by model and by layer, and there is no
  reliable predictor of which layer will break first. *What would close it:* a per-layer
  sensitivity metric that predicts quality loss before you run the evaluation, validated
  across architectures.
- **How do you schedule a mixed prefill/decode workload optimally?** Prefill is
  compute-bound; decode is bandwidth-bound. A real server runs both at once. The optimal
  split of hardware between them, and the optimal admission order, is not solved — it
  depends on the traffic mix, which changes. *What would close it:* a scheduler whose
  policy is provably near-optimal across a range of measured traffic mixes, not tuned to
  one.

## See also

- [The question-loop ledger](README.md) — the schema, the four categories, and the
  machine-checked rules behind this page.
- [The inference ladder](../explainers/inference/README.md) — the five rungs every
  "→ full answer" above points into.
- [The caching ladder](../explainers/caching/README.md) — the same five-rung pattern
  applied to fak's caching stack. The inference ladder is its sibling.
- [What is a CUDA kernel?](../explainers/what-is-a-cuda-kernel.md) — the word "kernel"
  means three different things here; read this before the `mat.mul` calls confuse you.
- [The fak/DOS glossary](../explainers/glossary.md) — one-line definitions of the
  vocabulary.

---

*[← back to the question-loop README](README.md) · [the inference ladder](../explainers/inference/README.md)*

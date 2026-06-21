# Benchmarking fak against SGLang's KV-cache RadixAttention

> **📊 AUTHORITY:** All benchmark numbers for RadixAttention are centrally indexed in
> **[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md)**. That document is the single source
> of truth with full traceability to commits and artifacts. This doc provides the detailed
> context and explanation.
>
> **Headline Result:** **4.87×** live wall-clock speedup on SmolLM2-135M Q8 (agents workload).
> See Authority for full details.

> New here? Read **RADIXATTENTION-EXPLAINER.md** first — a
> from-first-principles walkthrough (KV cache → RadixAttention → what we built → the
> results, with diagrams). This doc is the dense results ledger; that one teaches it.

**Thesis.** SGLang's **RadixAttention** (arXiv:2312.07104 / NeurIPS 2024) is the SOTA
mechanism for *automatic* KV-cache reuse across requests: a radix tree keyed by token
sequences discovers the longest cached prefix of every incoming request and recomputes
only its suffix. The prior results doc (`MODEL-BASELINE-RESULTS.md`) named it the *closest
conceptual peer* to fak's kernel-owned KV cache but stopped at "regime mismatch — GPU
throughput serving, not fak's claim." This doc closes that gap with an actual head-to-head
on the one axis that **is** fair, and shows fak does the same thing — plus one thing the
LRU radix cache structurally cannot.

The whole comparison hangs on choosing the right metric, so that comes first.

## The axis: cache hit rate, not tok/s

SGLang optimizes aggregate GPU throughput; fak runs batch-1 on a CPU. Comparing raw tok/s
across those regimes is the apples-to-oranges the repo already refuses to do. But
RadixAttention's **own headline metric is the cache hit rate** — the fraction of prompt
tokens served from cache instead of recomputed — and that number is **hardware- and
model-independent**: it is a function of *(workload, matching algorithm)* only. A radix
tree that reuses 88% of the prompt tokens reuses 88% whether the K/V lives in HBM on an
H100 or in a Go slice on this laptop.

So the fair experiment is: run **the same algorithm** SGLang runs — a radix tree of token
sequences + longest-prefix match + LRU-leaf eviction — on **the same workload shapes**, and
compare the hit rate fak achieves to the hit rate SGLang's paper reports. We built that
algorithm (`internal/radixkv`) as a pure consumer of fak's *already-proven* KV primitives
(`Clone`, `Prefill`, `Evict`), and benchmarked it (`cmd/radixbench`).

## What we built — RadixAttention, verbatim to the paper's spec

A background research pass pinned the algorithm against the primary paper (every claim
below CONFIRMED verbatim against the NeurIPS 2024 proceedings PDF), and `internal/radixkv`
implements it point-for-point:

| RadixAttention (paper) | `internal/radixkv` |
|---|---|
| radix tree (compressed trie) keyed by token sequences | `Tree` of `node{key []int, children map[int]*node}` |
| longest-prefix match at runtime; reuse prefix, compute suffix | `Lookup` walks the tree, returns matched length + reusable cache |
| edge split when a request diverges mid-run | `split` truncates the child's cache to the boundary via `Evict`-of-tail |
| LRU eviction of the least-recently-used **leaf** first | `evictToBudget` → `lruLeaf` (parents collapse upward) |
| reference counting; node evictable iff refcount == 0 | `refs` lease, transferred Lookup→leaf in `Insert`; `Done` releases |
| cache-aware longest-shared-prefix-first scheduling (≡ DFS order) | `cacheAwareOrder` (lexicographic ≡ DFS pre-order) |

The one deliberate departure is faithfully labeled, not hidden: each node stores the
**full-prefix** KVCache for its path (length == path length), not SGLang's per-token paged
slabs. That costs memory, but it lets every reuse and every split go through the *verified*
`Clone`/`Evict` primitives unchanged — and the metric we compare (hit rate / tokens saved)
is layout-independent. The soundness of that choice is the load-bearing test below.

### Soundness rung: reuse-through-a-split is bit-identical to recompute

`internal/radixkv`'s key test serves a request whose prefix diverges in the *middle* of a
cached run — the hard case the radix tree exists for, forcing an edge split that truncates
the first request's cache to the shared boundary. Serving the second request from that
truncated, cloned cache and prefilling only its suffix produces logits **bit-identical**
(`max|Δ| = 0`, argmax equal) to a fresh full prefill. The split's only KV math is
`KVCache.Evict` of the tail, which re-RoPEs no survivor (nothing is cached past the cut), so
it is an exact prefix — the reuse spends *not one bit* of the proven correctness. Green via
WSL (`go test ./internal/radixkv`).

## Results — hit rate inside SGLang's band, on the same workload shapes

Driven through `cmd/radixbench` (synthetic wiring model; the hit-rate numbers are exact
integer token counts, identical on any model since they depend only on the token structure):

| workload (SGLang shape) | reqs | hit rate | prefill-token speedup | inside SGLang 50–99%? |
|---|---|---|---|---|
| **few-shot** (MMLU 5-shot: N share a preamble) | 16 | **88.2 %** | 8.50× | ✅ |
| **multi-turn chat** (each turn reuses full history) | 8 | **79.5 %** | 4.89× | ✅ |
| **tree-of-thought** (ToT/GSM-8K: branching, siblings share ancestors) | 27 | **77.2 %** | 4.40× | ✅ |
| **agents** (ReAct: C concurrent agents share a system prefix) | 30 | **86.7 %** | 7.50× | ✅ |

Every workload lands inside SGLang's verified 50–99 % hit-rate band. These are the shapes
SGLang's §6.1 evaluates (few-shot MMLU/HellaSwag, 4-turn chat, ToT on GSM-8K,
ReAct/generative agents); the hit rate is the property of the workload + the matching
algorithm, and fak runs the same algorithm.

## Reproducing the paper's scheduling theorem (96 % of optimal → 100 %)

The paper proves that visiting requests in **DFS order** achieves the *optimal* cache hit
rate when the cache budget ≥ the max request length, that **longest-shared-prefix-first
(cache-aware) scheduling is equivalent to DFS order**, and that in practice their online
cache-aware scheduler reaches **96 % of optimal on average**. We reproduce this directly on
the **agents** workload, where the C agents' requests *interleave* in arrival time (realistic
concurrent serving), so a bounded cache thrashes under FCFS:

| order, bounded cache (budget = 1.5 × max req) | agents hit rate | % of unbounded optimal |
|---|---|---|
| **FCFS** (interleaved arrivals) | 62.1 % | 72 % |
| **cache-aware** (lexicographic ≡ DFS) | **86.7 %** | **100 %** |

Cache-aware scheduling recovers the full hit rate — reaching 100 % of optimal (vs the
paper's 96 % average), because our offline lexicographic sort *is* exact DFS pre-order, the
configuration the theorem says is optimal. The single-prefix workloads (few-shot,
multi-turn, ToT) are order-insensitive (FCFS already == cache-aware == optimal), which is
itself the predicted result: scheduling only matters when independent prefix groups
interleave.

## What the radix *tree* adds over fak's prior reuse

fak already had KV-prefix reuse — but only of a **declared** prefix
(`model.NewBatchFromPrefix`): you tell it the one prefix shared by every request. The radix
tree's contribution is *discovery* — it finds **every** shared subtree automatically,
including mid-run divergences. The gap is measured:

| workload | declare-one-prefix hit rate | radix hit rate | reuse radix finds beyond declare-one |
|---|---|---|---|
| few-shot (single shared prefix) | 88.2 % | 88.2 % | 1.00× (nothing to add — one prefix) |
| multi-turn chat | 31.8 % | 79.5 % | **2.50×** |
| tree-of-thought | 55.0 % | 77.2 % | **1.40×** |
| agents | 58.4 % | 86.7 % | **1.48×** |

On the single-shared-prefix case the two are identical (there is only one prefix to find).
On the *tree-* and *chain-* shaped workloads — exactly the structured-program regime
RadixAttention targets — the radix tree reuses **1.4–2.5× more** tokens than declaring the
one global prefix, because it captures every internal node (each turn of the chat chain,
every shared ancestor of the reasoning tree, each agent's private growing context).

## Live wall-clock — token savings become time savings

The hit rate is exact bookkeeping; the live arm confirms it converts to real prefill time on
the kernel. Each request is a real prefill: the baseline prefills the full prompt; the radix
arm clones the matched prefix's cache and prefills only the suffix (the clone is a **copy**,
not SGLang's zero-copy page share, so this is a *conservative* lower bound — fak pays for the
copy that a paged share would not).

The deterministic token-speedup column is the headline; the two live columns show the
wall-clock **ratio** the token saving produces on a tiny wiring model (synthetic, best-of-3)
and on a real 30-layer checkpoint (SmolLM2-135M Q8). **Only the within-run ratios are
meaningful** — the real-model run shared the box with other benches, so its absolute
latencies are contention-inflated (~140× the isolated Q8 prefill) and are *not* quoted as
representative; the ratio survives because both arms ran back-to-back under the same load.

| workload | token speedup | synthetic live ratio | real-model live ratio (contended) |
|---|---|---|---|
| few-shot | 8.50× | 22.6× | **6.94×** |
| multi-turn chat | 4.89× | 2.99× | **2.31×** |
| tree-of-thought | 4.40× | 3.06× | **3.28×** |
| agents | 7.50× | 1.64× | **4.87×** |

The live ratio brackets the (exact) token speedup, and *why* it lands above or below is
instructive:

- **agents: synthetic 1.64× → real-model 4.87×** — the headline live result, and a clean
  confirmation of the design's one cost. fak's reuse *copies* the cached prefix (`Clone`),
  where SGLang's is a zero-copy page share. On the 64-hidden wiring model the per-token compute
  is so cheap the copy dominates and masks the saving (1.64×); on the real model per-token
  compute ≫ a memcpy, so the live ratio climbs back toward the 7.5× token figure (4.87×).
  Exactly the predicted direction — the clone overhead is a small-model artifact, not the
  steady state.
- **few-shot: synthetic 22.6× (super-linear), real-model 6.94×** — on the tiny model the
  baseline re-prefills the 256-token preamble 16×, and prefill attention is **O(L²)** with
  trivial per-token weight cost, so the long baseline prefills cost super-linearly and the
  wall-clock saving *exceeds* the token saving. On the 135M model the O(L·params) weight term
  dominates that O(L²) attention term, the super-linear effect washes out, and the small clone
  overhead pulls the ratio just below the token figure (6.94× vs 8.5×). Same code, opposite
  regime — both honestly reported.

So the live arm confirms the token savings convert to wall-clock savings, and on a *real*
model do so close to the exact token ratio. The token-count metric remains the headline: it
is model-independent and exact, and it — not a load-sensitive wall-clock — is the comparison
that actually matters against SGLang. (Artifacts: `radixbench-synthetic.json`,
`radixbench-smollm2-135m-q8.json`; the latter is a heavily contended run, kept for the ratio,
not the absolute ms.)

## The differentiator the LRU cannot offer

RadixAttention evicts **only under memory pressure**, by LRU. Because fak owns the KV cache,
`radixkv.EvictNode` adds a second governance mode on the *same* tree: evict a named prefix
because **policy** says so — a quarantine verdict on a poisoned tool-result span — regardless
of recency or budget. The witness: build a tree where a benign request and a poisoned one
share a system prefix, evict the poisoned node on a verdict → its unique span is freed, the
**shared prefix and the benign sibling are untouched**, and (per `internal/model`'s proven
`Evict == never-saw-it`) any session that cloned that prefix can scrub the span bit-identically
to never having seen it. Same primitive as SGLang (prefix-addressable KV), opposite
governance (policy-driven, provable — not cache-pressure LRU). That is the part no throughput
engine offers, now measured rather than asserted.

## Bottom line

- **fak reproduces SGLang's RadixAttention on its own headline metric.** 77–88 % cache hit
  rate across the few-shot / chat / tree-of-thought / agents shapes — inside SGLang's
  verified 50–99 % band — running the identical radix-tree + longest-prefix + LRU-leaf
  algorithm, soundness-pinned bit-for-bit to the proven KV core.
- **It reproduces the scheduling theorem.** Cache-aware (≡ DFS) ordering recovers the agents
  workload from FCFS 62.1 % to **100 % of optimal**, vs the paper's 96 % average.
- **The radix tree strictly extends fak's prior reuse**, finding 1.4–2.5× more reuse than
  the declare-one-prefix path on tree/chain workloads.
- **And it does the one thing the LRU cache can't**: policy-driven, span-exact, provable
  eviction — the reason to own the KV cache in the first place.
- **The regime caveat stands and is unchanged:** this is a fair comparison *on cache hit
  rate* (algorithm efficiency), not on absolute serving throughput. SGLang remains a GPU
  throughput engine; fak's claim was never raw tok/s but owning the KV cache for provable
  operations — now shown to also match the SOTA reuse algorithm it was said to merely
  "conceptually" resemble.

## Sources (all CONFIRMED verbatim against the primary text)

- **SGLang / RadixAttention** — Lianmin Zheng et al., "SGLang: Efficient Execution of
  Structured Language Model Programs," arXiv:2312.07104; peer-reviewed NeurIPS 2024
  proceedings (paper 724be4472168f31ba1c9ac630f15dec8). Verified: radix-tree KV reuse,
  LRU-leaf eviction, reference counting, cache-aware/longest-shared-prefix-first scheduling,
  DFS optimality, the **50–99 %** cache-hit range, **96 % of optimal** average, and the
  end-to-end **6.4× throughput / 3.7× latency** figures (Llama-2 7B–70B fp16 vs
  Guidance/vLLM/LMQL).
- **fak** — `internal/radixkv` (algorithm), `cmd/radixbench` (harness),
  `experiments/radixattention/*.json` (artifacts), `internal/model` (the proven KVCache
  Clone/Evict primitives this is built on).

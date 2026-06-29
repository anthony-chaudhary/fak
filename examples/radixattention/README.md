# fak kernel — RadixAttention adoption walkthrough

**RadixAttention is automatic KV-cache reuse: a radix tree of token sequences finds the
longest cached prefix of every incoming request and recomputes only the suffix.** fak
rebuilds SGLang's algorithm (arXiv:2312.07104 / NeurIPS 2024) over its own kernel-owned KV
cache as [`internal/radixkv`](../../internal/radixkv/), and [`cmd/radixbench`](../../cmd/radixbench/)
measures it on the one axis where fak-on-CPU and SGLang-on-GPU compare fairly: the **cache
hit rate** — the fraction of prompt tokens served from cache instead of recomputed. That
number is hardware- and model-independent (it is a function of *(workload, matching
algorithm)* only), so a 88% reuse is 88% whether the K/V lives in HBM on an H100 or in a Go
slice on a laptop.

This walkthrough shows an operator how to (a) sweep the four SGLang workload shapes against
their **own** token-id prompt set, (b) read the hit rate in the context of SGLang's verified
**50–99%** band, and (c) understand the honesty property that makes the reuse free: serving a
request through an edge-split is **bit-identical to a fresh recompute** (`max|Δ| = 0`).

```
  request  r = [ shared prefix ............ | this request's suffix ]
                 \__ found in the radix tree by longest-prefix match __/   \__ the only part prefilled __/
                       reuse it (clone the cached KV)                             recompute just this
```

## Run it

```bash
./examples/radixattention/run.sh                    # bench the four bundled workloads
./examples/radixattention/run.sh --out report.json  # also write the full JSON report
./examples/radixattention/run.sh --only few-shot     # one shape only
```

`run.sh` invokes the **real** harness — `go run ./cmd/radixbench -workload <the four JSONs>`
— over the bundled `sample-workload/*.json`. It needs the **Go toolchain** (the bench is a Go
program in `cmd/radixbench`; it is *not* part of the `fak` binary). On a box without Go the
script prints the exact command plus the published numbers and exits `0` — see
[Run it without a Go toolchain](#run-it-without-a-go-toolchain).

Windows: run the `.sh` from WSL or Git Bash, or call `go run` directly from the repo root.

## What you put in: a workload is a list of token-id requests

A workload file is a tiny JSON — a named set of requests, each request a flat list of token
ids. **Shared prefixes are literally the same leading ids** — that is exactly what the radix
tree discovers and reuses. The schema (read by `loadWorkload` in `cmd/radixbench/main.go`):

```json
{
  "name": "few-shot",
  "desc": "5 questions share one few-shot preamble",
  "sglang_published": "single-level prefix reuse; within the 50-99% band",
  "requests": [ [10,11,12, 100], [10,11,12, 110], [10,11,12, 120] ]
}
```

To sweep your own prompts, **tokenize them to ids and write one file per shape.** The hit
rate depends only on the ids, so you do not need a model to get the headline number — but if
your ids exceed the default synthetic model's vocab (256), the live wall-clock arm disables
itself (the accounting still runs); pass `-hf <snapshot>` or `-dir <export>` to time on a real
model over your own tokens.

### The four shapes, and why they reuse differently

| shape | bundled file | the sharing structure | why it reuses |
|---|---|---|---|
| **few-shot** | `few-shot.json` | N questions share one preamble | one shared prefix (single level) |
| **multi-turn chat** | `multi-turn-chat.json` | each turn = full history + a new message | each turn reuses the entire previous context (a growing chain) |
| **tree-of-thought** | `tree-of-thought.json` | a branching reasoning tree; root-to-leaf paths | siblings share every ancestor (multi-level) |
| **agents** | `agents.json` | C agents share a system prefix; each has a private growing chain | two-level: shared system across agents + each agent's own chain, arrivals interleaved |

These are the shapes SGLang's §6.1 evaluates (few-shot MMLU, 4-turn chat, ToT on GSM-8K,
ReAct/generative agents). The `tree-` and `chain-` shapes are exactly where the radix *tree*
beats declaring a single shared prefix — it captures every internal node, not just the one
global prefix.

## What you read out: the cache hit rate, against the SGLang band

Per shape the bench prints the **hit rate** (radix reuse) and the cross-subtree win (how much
more the tree reuses than the one declared prefix). The bundled fixtures are deliberately
**small and hand-verifiable**, so their numbers are *below* the published band — that is the
honest point of a self-contained demo, not a regression:

| bundled shape | hit rate (this small fixture) | radix vs declare-one-prefix | published full-shape (RESULTS.md) |
|---|---|---|---|
| few-shot | 70.0% | 1.00× (one prefix — nothing to add) | **88.2%** |
| multi-turn chat | 61.8% | **1.75×** | **79.5%** |
| tree-of-thought | 50.0% | **1.50×** | **77.2%** |
| agents | 81.1% | 1.14× | **86.7%** |

Read the two columns together. The **shape** is the result: few-shot's radix reuse equals
declare-one (there is only one prefix to find), while multi-turn and tree-of-thought show the
tree discovering **1.5–1.75× more** reuse than a single declared prefix — the same story the
full run tells at 1.4–2.5×. To reproduce the **published 77.2–88.2% band**, run `radixbench`
with **no** `-workload` flag: that drives the full synthetic SGLang shapes (256-token
preambles, deeper trees) the [RESULTS.md](../../docs/benchmarks/RADIXATTENTION-RESULTS.md)
numbers come from.

### How to read the hit-rate curve

- **Higher is better, but the ceiling is the workload, not the engine.** A shape's max hit
  rate is `shared_tokens / total_tokens`; few-shot with a long preamble approaches it,
  a workload with no shared prefix sits near 0. The radix tree's job is to *reach* that
  ceiling automatically, including mid-run divergences a single declared prefix misses.
- **Inside 50–99% means "matches SGLang."** SGLang's paper reports a **50–99%** hit-rate band
  across its benchmarks; every published fak shape lands inside it, running the identical
  radix-tree + longest-prefix + LRU-leaf algorithm. A number inside that band is the success
  condition; a number below it usually means the workload simply shares less.
- **Order matters only when independent prefix groups interleave.** For a single shared
  prefix (few-shot, multi-turn, tree-of-thought) FCFS already equals cache-aware order. The
  **agents** shape interleaves C agents' arrivals, so under a *bounded* cache FCFS thrashes;
  cache-aware (longest-shared-prefix-first ≡ DFS) order recovers it. The full run shows agents
  **FCFS 62.1% → cache-aware 86.7% = 100% of optimal** on that workload (the paper proves DFS
  order is optimal at budget ≥ max request length, and reports 96% of optimal on average for
  their *online* scheduler — not directly comparable to this offline lexicographic sort).

## The honesty property: reuse-through-a-split is bit-identical to recompute

The hard case the radix tree exists for is a request that diverges in the **middle** of a
cached run — forcing an *edge split* that truncates the first request's cached KV to the
shared boundary. Serving the second request from that truncated, cloned cache and prefilling
only its suffix produces logits **bit-identical** (`max|Δ| = 0`, argmax equal) to a fresh full
prefill. The split's only KV math is `KVCache.Evict` of the tail, which re-RoPEs no survivor
(nothing is cached past the cut), so it is an exact prefix — **the reuse spends not one bit of
the proven KV correctness.**

The bench *witnesses the radix-tree governance* of this at runtime via its
**policy-eviction** check (`policyEvictionWitness`): build a tree where a benign request and a
poisoned one share a system prefix, evict the poisoned node **on a verdict** (not LRU), and
confirm its unique span is freed while the shared prefix and the benign sibling are untouched
— the capability an opportunistic LRU radix cache structurally cannot offer. The **bit-exact
KV equality** itself is proven one layer down, in `internal/radixkv`'s test suite, which is
the authoritative witness (see below).

## The honesty caveat — what this is *not* a win over

This walkthrough measures fak reproducing SGLang's RadixAttention **on the cache-hit-rate
axis** (algorithm efficiency), and frames the reuse against the **no-reuse / cold-recompute**
baseline — the "stateless API, re-prompt every call" pattern. It is **not** a head-to-head
throughput win over a tuned shared-prefix engine: SGLang, vLLM's automatic prefix cache, and
fak all prefill the shared prefix once; on raw GPU tok/s SGLang remains the throughput engine
and the repo does not claim otherwise. The two things fak's design *adds* are (1) the radix
tree finding 1.4–2.5× more reuse than declaring one global prefix, and (2) policy-driven,
span-exact, provable eviction — measured here, not asserted. (One more honest note: fak's
reuse *copies* the cached prefix where SGLang's is a zero-copy page share, so any live
wall-clock ratio here is a *conservative* lower bound — fak pays for a copy a paged share would
not.)

## Run it without a Go toolchain

The bench is a Go program (`cmd/radixbench`), not part of the `fak` binary, so it needs the Go
toolchain to run. With Go installed:

```bash
cd <fak repo root>
go run ./cmd/radixbench -workload \
  examples/radixattention/sample-workload/few-shot.json,\
examples/radixattention/sample-workload/multi-turn-chat.json,\
examples/radixattention/sample-workload/tree-of-thought.json,\
examples/radixattention/sample-workload/agents.json
```

Without Go, the published numbers are the witness: **77.2–88.2% cache hit rate** across the
four shapes, inside SGLang's **50–99%** band, with reuse-through-a-split proven bit-identical
to recompute (`max|Δ| = 0`). A captured run of `run.sh` is in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## The authoritative witness: the Go tests

This example is a runnable *adoption walkthrough*; the **authoritative witness** that the
shipped algorithm behaves this way is the Go test suite. To run the bit-identical-to-recompute
proof and the radix-tree units directly:

```bash
go test ./internal/radixkv    # the algorithm: longest-prefix match, edge split, LRU-leaf, refcount, policy eviction
go test ./cmd/radixbench      # the harness accounting + the #322 -workload loader (TestLoadWorkload)
```

`internal/radixkv/proofs_witness_test.go` is the soundness rung: it serves a request that
diverges mid-run, forces the edge split, and asserts the resulting logits are `max|Δ| = 0`
against a fresh prefill.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher — runs `go run ./cmd/radixbench -workload …` over the four bundled shapes |
| `sample-workload/few-shot.json` | N questions share one preamble (single-level reuse) |
| `sample-workload/multi-turn-chat.json` | a 4-turn conversation; each turn reuses the full history |
| `sample-workload/tree-of-thought.json` | a branching reasoning tree; siblings share every ancestor |
| `sample-workload/agents.json` | concurrent agents sharing a system prefix, arrivals interleaved |
| `EXAMPLE-OUTPUT.md` | a captured run |

## Cross-references

- **`CLAIMS.md` #66 — "RadixAttention parity vs SGLang"**: *…rebuilt over the kernel-owned
  KVCache as `internal/radixkv` … Measured (`cmd/radixbench`): **77.2–88.2% cache hit rate**
  across the few-shot / multi-turn-chat / tree-of-thought / agents shapes — inside SGLang's
  verified **50–99%** band — with reuse-through-an-edge-split proven **bit-identical to
  recompute** (max|Δ|=0).*
- **The dense results ledger**: [`docs/benchmarks/RADIXATTENTION-RESULTS.md`](../../docs/benchmarks/RADIXATTENTION-RESULTS.md)
  — the full per-shape table, the scheduling theorem, the live wall-clock arm, and the sources
  (all confirmed verbatim against the NeurIPS 2024 PDF).
- **The from-first-principles explainer**: [`docs/explainers/addressable-kv-cache.md`](../../docs/explainers/addressable-kv-cache.md)
  — KV cache → addressable KV → what owning it buys (the policy-eviction half RadixAttention's
  LRU cannot offer).
- **The code**: [`internal/radixkv/`](../../internal/radixkv/) (the algorithm),
  [`cmd/radixbench/`](../../cmd/radixbench/) (the harness),
  [`experiments/radixattention/*.json`](../../experiments/radixattention/) (the captured artifacts).

> **A note on two paths.** `CLAIMS.md` #66 and `RADIXATTENTION-RESULTS.md` refer to the results
> doc and explainer as top-level `RADIXATTENTION-RESULTS.md` / `RADIXATTENTION-EXPLAINER.md`.
> In this tree they live at **`docs/benchmarks/RADIXATTENTION-RESULTS.md`** and, for the
> first-principles walkthrough, **`docs/explainers/addressable-kv-cache.md`** (there is no
> file literally named `RADIXATTENTION-EXPLAINER.md`); the links above resolve to the real
> paths.

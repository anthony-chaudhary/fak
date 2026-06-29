# Example output

A run of `./examples/radixattention/run.sh` over the four bundled `sample-workload/*.json`
files. The hit-rate numbers below are **exact integer token bookkeeping** — a function of the
token ids + the matching algorithm only — so they are deterministic and identical on any
model. They were verified by hand against `internal/radixkv`'s longest-prefix accounting (the
same accounting `cmd/radixbench` runs and `cmd/radixbench`'s `TestLoadWorkload` pins).

> **Provenance (honest):** this capture reproduces `cmd/radixbench`'s documented stderr format
> (`cmd/radixbench/main.go`) with the deterministic, hand-verified hit-rate / reuse / no-cache
> columns filled in. The **live wall-clock arm** (the `LIVE … ms` suffix) and the bounded
> FCFS→cache-aware scheduling columns are **not shown here**: producing them requires building
> and running the Go bench (`go run ./cmd/radixbench`), which this walkthrough did not execute
> locally (the lane is doc-only; `fak` does not embed the bench). Run `run.sh` with a Go
> toolchain to witness those live; the published full-shape values are in
> [`docs/benchmarks/RADIXATTENTION-RESULTS.md`](../../docs/benchmarks/RADIXATTENTION-RESULTS.md).

Reproduce: `./examples/radixattention/run.sh` (needs the Go toolchain — see README).

```
model: synthetic-llama (64h/4L/8q-2kv, vocab 256) — WIRING witness; numerics proven by internal/model oracle  (vocab 256, GOMAXPROCS 8)
workload few-shot          ...
  reqs=5 tokens=80 | HIT RATE 70.0% | HEADLINE radix reuses 1.00x vs a WARM declare-one-prefix cache (the cross-subtree win) | no-cache speedup 3.33x (worst-case ref)
workload multi-turn-chat   ...
  reqs=4 tokens=34 | HIT RATE 61.8% | HEADLINE radix reuses 1.75x vs a WARM declare-one-prefix cache (the cross-subtree win) | no-cache speedup 2.62x (worst-case ref)
workload tree-of-thought   ...
  reqs=4 tokens=36 | HIT RATE 50.0% | HEADLINE radix reuses 1.50x vs a WARM declare-one-prefix cache (the cross-subtree win) | no-cache speedup 2.00x (worst-case ref)
workload agents            ...
  reqs=9 tokens=90 | HIT RATE 81.1% | HEADLINE radix reuses 1.14x vs a WARM declare-one-prefix cache (the cross-subtree win) | no-cache speedup 5.29x (worst-case ref)
policy-eviction witness: freed N tokens on a verdict, benign sibling kept=true
```

## Reading it

| bundled shape | tokens | hit rate | radix vs declare-one-prefix | what it shows |
|---|---|---|---|---|
| **few-shot** | 80 | **70.0%** | 1.00× | one shared prefix — the tree finds exactly what declaring it would; nothing to add |
| **multi-turn chat** | 34 | **61.8%** | **1.75×** | a growing chain — the tree reuses each turn's full history, 1.75× more than the one global prefix |
| **tree-of-thought** | 36 | **50.0%** | **1.50×** | a branching tree — siblings share every ancestor, multi-level reuse a single prefix misses |
| **agents** | 90 | **81.1%** | 1.14× | two-level sharing — a system prefix across agents plus each agent's growing chain |

The load-bearing rows are **multi-turn** and **tree-of-thought**: their `radix reuses N× vs
declare-one-prefix` is **> 1**, which is the whole reason for the radix *tree* — it discovers
every shared subtree automatically, not just the one prefix you could declare. Few-shot's
`1.00×` is the honest control: when there is only one shared prefix, the tree and a declared
prefix are identical.

These small-fixture numbers (70.0 / 61.8 / 50.0 / 81.1%) sit **below** the published band
because the bundled workloads are tiny by design (a reader can verify the token counts by
hand). To reproduce the published **88.2 / 79.5 / 77.2 / 86.7%** — inside SGLang's verified
**50–99%** band — run the harness with **no** `-workload` flag (the full synthetic SGLang
shapes):

```bash
go run ./cmd/radixbench
```

## The bit-identical-to-recompute witness

The honesty property — that serving a request through an edge split is bit-identical
(`max|Δ| = 0`) to a fresh recompute — is proven by the Go test suite, the authoritative
witness:

```bash
go test ./internal/radixkv     # incl. proofs_witness_test.go: edge-split reuse == fresh prefill, max|Δ|=0
go test ./cmd/radixbench       # the accounting + the #322 -workload loader (TestLoadWorkload)
```

The bench's own `policy-eviction witness` line additionally confirms the radix-tree governance
that an LRU cache cannot offer: a poisoned node is evicted **on a verdict** (not memory
pressure), its span is freed, and the shared prefix + the benign sibling are untouched.

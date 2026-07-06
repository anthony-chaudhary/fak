---
title: "Pure-fak GLM-5.2 benchmark: end-to-end wiring gap — glmdsatput runs but never lands a canonical witnessed artifact (2026-07-06)"
description: "Audit of whether the PURE-fak (native glm_moe_dsa kernel) GLM-5.2 path is benchmarked properly end-to-end and documented. Finding: cmd/glmdsatput measures fak's own MLA+DSA decode on real CUDA and forward correctness is bit-exact (cosine 1.000000), but the throughput path is NOT wired to the canonical benchmark ledger — glmdsatput emits its own glm-throughput/1 record, not benchmark/run-manifest.v1, and zero glmdsatput run artifacts exist under experiments/benchmark/runs/. The benchcli package already has the writer to close this. Names the exact wiring fix and the two on-box witnesses still owed."
---

# Pure-fak GLM-5.2 benchmark: the end-to-end wiring gap

> **Scope.** This audits the **pure-fak** GLM-5.2 path — fak's *own* `glm_moe_dsa` kernels
> (in-kernel MLA + DSA indexer + sparse-attend + FFN forward on fak's compute backend), **not**
> the llama.cpp / SGLang / vLLM baselines. The question: is that path benchmarked properly
> end-to-end and documented? **Answer: measured and correct, but not landed as a witnessed
> artifact — the last mile is unwired.**

## 1. What the pure-fak path already has (the good part)

- **Forward correctness — WITNESSED, bit-exact.** GLM-5.2's DSA forward on fak's own CUDA
  kernels: cosine `1.000000`, argmax-exact, re-witnessed at HEAD `f39796e` (sm_80). Records:
  `experiments/glm-gpu-witness/*.json` (`glm-gpu-witness/1`).
- **A dedicated native throughput benchmark — `cmd/glmdsatput`.** It times fak's native
  `glm_moe_dsa` decode/prefill on a real backend (`-tags cuda`), honestly scoped: synthetic
  weights, reduced layers, dense-FFN (no MoE experts) — an *optimistic lower-bound on
  per-token device cost*, explicitly **not** the 753B serving rate. The `scope` field travels
  in every record so the number cannot be quoted out of its caveat. First numbers recorded in
  [GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md)
  §2 (e.g. 8L/2048H/topk256 → 26.53 decode tok/s, HEAD `b68a182`).
- **A documented, category-honest benchmarking framework** (same note §4): the three-number
  boundary (synthetic-native vs 753B-served vs field engines) so no comparison is a category
  error.

So the path is real, correct, and runnable. The gap is not "does it work" — it is **"where does
the number land."**

## 2. The gap — the throughput path does not reach the canonical ledger

Two concrete, verifiable defects:

**(a) Wrong schema — glmdsatput emits `glm-throughput/1`, not `benchmark/run-manifest.v1`.**
`cmd/glmdsatput/main.go:276` marshals a bespoke record (`GLMTPUT_JSON {…}`) with
`"schema": "glm-throughput/1"`. The canonical benchmark ledger under
`experiments/benchmark/runs/by-machine/**` is keyed on `"$schema": "benchmark/run-manifest.v1"`
with a `claim_class` (WITNESSED / OBSERVED) and lineage stamp. A `glm-throughput/1` blob is not
admitted there, so a pure-fak decode run is **not discoverable as a benchmark run** the way the
llama.cpp CPU-wedge and the SGLang serving-parity runs are.

**(b) Zero landed artifacts.** `find experiments/benchmark/runs -iname '*.json' | xargs grep -l
glmdsatput` → nothing. Every other GLM number in the ledger (cpu-wedge, sglang serving-parity)
has a manifest; the pure-fak path has none. The native-throughput plan note flags this itself as
its **P1 — "Land the throughput record"** — the 4-config `glm-throughput/1` record is stranded on
a private box scratch path, never scrubbed + committed into the ledger.

Net: the pure-fak GLM-5.2 decode number exists but lives **outside** the witnessed benchmark
tree, so `dos verify` can't bind it, the benchmark index doesn't list it, and a reader surveying
`experiments/benchmark/runs/` would conclude fak has *no* native GLM decode number — which is
false.

## 3. The fix is small — the writer already exists

`internal/benchcli` owns the canonical schema and already has the machinery to wrap any bench
JSON into a lineage-stamped `benchmark/run-manifest.v1` artifact:

- `benchcli.Stamp()` → a `Lineage` (UTC + commit + node).
- `benchcli.ArtifactFromJSON(lin, report)` → a `BenchmarkArtifact` around an arbitrary report
  blob.
- `benchcli.WriteReport(path, report)` → writes it under the runs tree.

So closing the gap is **wiring, not new infrastructure**:

1. In `cmd/glmdsatput`, when `-emit-json` is set, route the record through
   `benchcli.ArtifactFromJSON` (keep the `glm-throughput/1` body as the inner `report`; the
   outer envelope becomes `benchmark/run-manifest.v1` with `claim_class` + lineage). Preserve the
   load-bearing `scope` field verbatim so the synthetic/lower-bound caveat survives into the
   ledger.
2. Add a `-out <dir>` flag (or a tiny wrapper) that writes to
   `experiments/benchmark/runs/by-machine/<node>/<UTC>-glm52-native-decode/` — `manifest.json`
   + `result.json` + a short `RESULTS.md` carrying the sweep table and the scope caveat.
3. Land the *existing* stranded 4-config record (native-throughput note §2, HEAD `b68a182`) the
   same way — scrub the private a100 rollup public, commit under an `add(experiments)` trailer so
   `dos verify` binds it. This is the note's own P1.

None of this needs a GPU to *design*; steps 1–2 are pure Go edits testable on the cpu-ref path,
and only step 3's re-run (if a fresh number is wanted) needs the CUDA node.

## 4. What's still owed on-box (needs the CUDA node)

These are the honest not-yets — the measurement gaps the wiring can't fill:

- **P0 — the DSA-kernel illegal-memory-access at the largest configs** (32-layer;
  hidden-5120/40-head) still blocks a clean full 6-config sweep (native-throughput note §3). A
  single-variable on-box bisection is needed to pin the `k_dsa_*` out-of-bounds. Until then the
  sweep is 4/6 configs.
- **P2 — real-weight throughput.** `glmdsatput` uses synthetic random weights. The next rung is
  pointing `cmd/modelbench` (arch-blind) at a real `glm_moe_dsa` GGUF for a **real-weight** native
  decode number — no longer a lower-bound. The native-753B loader is done, so this is unblocked
  except for host time.

## 5. Recommendation (ranked)

1. **Wire glmdsatput → `benchmark/run-manifest.v1` via `benchcli` (§3.1–3.2).** Small Go change,
   cpu-ref-testable, makes every future pure-fak run land witnessed. This is the highest-value
   move and needs no GPU.
2. **Land the stranded 4-config record (§3.3).** Turns an existing measurement into a bound
   witness; closes the native-throughput note's P1.
3. **Then the on-box work (§4):** fix the DSA illegal-access to get the full sweep, then couple
   modelbench to real weights for a non-synthetic number.

The headline: **the pure-fak GLM-5.2 path is correct and measured, but its throughput number
never enters the witnessed ledger — one `benchcli` wiring fix (writer already exists) makes it
end-to-end, and that fix is GPU-free.**

*Companions:* [native throughput + benchmark plan](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md) ·
[Lane F active-set from GGUF header](GLM52-LANE-F-ACTIVE-SET-FROM-GGUF-HEADER-2026-07-06.md) ·
[GPU-server theoretical ceiling](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md).

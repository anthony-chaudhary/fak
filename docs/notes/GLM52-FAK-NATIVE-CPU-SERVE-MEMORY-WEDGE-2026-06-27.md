# GLM-5.2 fak-native CPU serve on a 256-core / 1 TB host — load works, the all-resident serve wedges on RAM (2026-06-27)

Data-collection run on a **CPU-only** server (256-thread x86_64, ~1 TB RAM, **no GPU**) — the
companion of the GPU-side
[GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md).
It records the **first fak-native attempt to serve the full ~433 GB GLM-5.2 UD-Q4_K_M on pure
CPU** (no GPU, no expert-offload device): the load completes, but the **all-resident serve
exhausts host RAM at gateway-init and wedges the box**. The number that matters here is a
*negative*: fak's CPU serve needs ≥ model-size of free anonymous RAM, and there is no pre-flight
that refuses cleanly when there isn't.

Host details are scrubbed to the generic hardware class; the lab host/channel stay private.

## Command
```
FAK_Q4K=1 FAK_GGUF_LOAD_WORKERS=64 \
  fak serve --gguf <GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf> --context-budget-tokens 8192 \
            --addr 127.0.0.1:8080
```
`--backend` empty ⇒ the CPU reference path. `FAK_Q4K=1` ⇒ the direct-resident-Q4_K loader
(memory-safe vs the default lean-Q8 round-trip, which would balloon to Q8 size and OOM far sooner).
fak built from origin/main `ed4dc8dc`.

## Finding 1 — resident-Q4_K CPU load is dequant-bound (~0.06–0.10 GB/s), not <10 min
Resident-bytes vs wall-clock (single process, `FAK_GGUF_LOAD_WORKERS=64`):

| wall | resident | inst GB/s |
|---|---|---|
| 0m38s | 7.2 GB | 0.19 |
| ~8m | 45 GB | ~0.09 |
| ~21m | 95 GB | 0.065 |
| ~36m | 149 GB | 0.060 |
| ~48m | 213 GB | 0.090 |
| ~69m | 343 GB | 0.10 |
| ~80m | 379 GB | 0.057 |
| ~91m | 416 GB | 0.057 |
| ~101m | 450 GB | — |

Steady ~0.06–0.10 GB/s ⇒ a **~95 min** full load. This re-confirms the open load-speed witness:
the parallel-quant-load (S1) + resident-Q5/6_K (S2) levers do **not** bring the pure-CPU serve
under 10 min — UD-Q4_K_M's mixed Q5_K/Q6_K experts still take the slow dequant path on the CPU
reference serve (the resident-Q5/6_K lever `6b9fbc3` is wired for the GPU cpu-offload path).

## Finding 2 — the all-resident serve wedges the host (the load-bearing result)
The resident set grew **past the on-disk GGUF size**: the 433 GB model became **~458 GB resident**
(fak's resident-Q4_K structures duplicate/expand some tensors). On this host ~512 GB was reserved
in hugepages by a long-running co-tenant (`HugePages_Free=0`, not reclaimable), capping
normal-allocation RAM at **~495 GB**. So:

```
1007 GB total − 512 GB hugepages − ~458 GB resident model − ~37 GB other ≈ 2 GB available
```

At ~2 GB available the host — including the out-of-band control plane — could no longer `fork()`,
and fak never reached a usable bind. No fak throughput number was obtainable on this host.

This is the same class of result the GPU path already guards against with a `FitTooBig` plan
(`fitAndPlanServeGGUF…OnDevice`); the **CPU reference path has no equivalent** — `loadServeInKernelModel`'s
`FAK_Q4K` / default cases (`cmd/fak/serve.go` ~L799–824) load unconditionally. Filed as
**[#974](https://github.com/anthony-chaudhary/fak/issues/974)** (add a CPU-path memory-fit
pre-flight; and, larger, support mmap'd/demand-paged weights on CPU so a model larger than free
RAM can serve the way llama.cpp does).

## What this says about the comparison
- **llama.cpp** serves this exact 433 GB model on this exact host via **mmap** (demand-paged,
  evictable page cache) — it never needs all weights resident, so it does not hit this wall.
  That CPU baseline (decode/prefill tok/s) is the apples-to-apples reference and is the **next
  measurement** (pending host recovery; the prebuilt `llama-server`/`llama-bench` are on the box).
- fak's CPU serve, to be comparable on a memory-constrained host, needs the mmap/demand-paged
  weights path (#974-B). On a host with ≥ ~1.1× model-size of *free* RAM (no hugepage co-tenant),
  the all-resident path should bind — that throughput number is still open.

## Durable tooling shipped this run
- `experiments/nightrun/backlog.json` — a `witness-glm52-cpu-throughput` **frontier** datum so
  `fak nightrun next` surfaces "collect GLM-5.2 CPU throughput" on any CPU+weights box
  (commit `e0d54dbc`).
- A `say` verb on the private Slack control bridge for operator status notes.

## Next steps
1. Land the llama.cpp CPU baseline on the same host (decode/prefill tok/s) — the comparison row.
2. #974-A: CPU-path memory-fit pre-flight (refuse, don't wedge).
3. #974-B: mmap/demand-paged CPU weights, or run the all-resident path on a host without the
   hugepage co-tenant to capture the fak-native CPU throughput.
4. Extend the resident-Q5/6_K lever to the pure-CPU serve path so the load is I/O-bound, not
   dequant-bound (the <10 min target).

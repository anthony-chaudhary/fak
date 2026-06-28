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

Steady ~0.06–0.10 GB/s ⇒ a **~95 min** full load. The parallel-quant-load (S1) + resident-Q5/6_K
(S2) levers did **not** bring this pure-CPU serve under 10 min.

> **Correction (#975, 2026-06-27):** the original cause stated here — "UD-Q4_K_M's mixed
> Q5_K/Q6_K experts still take the slow dequant path on the CPU reference serve; the
> resident-Q5/6_K lever `6b9fbc3` is wired for the GPU cpu-offload path" — was **unwitnessed
> and is wrong about the loader**. `6b9fbc3` generalized the expert split in the CPU loader
> `QuantModelQ4KProfile` (the `FAK_Q4K` path's loader, `internal/ggufload/quant_q4k_loader.go`):
> it routes Q4_K **and** Q5_K/Q6_K experts to a raw-resident byte copy keyed on `info.Type`,
> not just on the GPU path. `6b9fbc3` is an **ancestor of this run's `ed4dc8dc`**, so the run
> *had* that routing. The real gap was **observability, not routing**: the `FAK_Q4K` case in
> `cmd/fak/serve.go` threaded **no `LoadProfiler`**, so — unlike the device cpu-offload case —
> the serve emitted **no per-quant-type load-path summary**, and the "slow dequant path"
> diagnosis above was inferred from the GB/s alone, never witnessed. That witness is now wired
> in (`(#975)`): the `FAK_Q4K` serve path threads a profiler and streams the
> `fak: load-path breakdown … resident=… dequant=…` summary + the resident report. The next
> CPU-host run settles it: if the expert rows show `dequant≈0`, the ~95 min is **I/O-bound**
> (433 GB at ~0.08 GB/s is disk/NFS read-bound — the mmap/demand-paged work tracked as
> **#974-B**), not dequant-bound; if a residual `dequant` slice remains, *that* is the routing
> bug to chase.

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

## Finding 3 — the slow load was path-precedence, not (only) the I/O ceiling (root cause, fixed)
A follow-up diagnosis on the GPU staging host (GPU server) found the ~1h+ loads were **path
precedence, not a missing copy**: a verified-complete local-NVMe copy already existed, but the
serve launchers globbed the NFS mount first, so every cold load silently paid the slow path.
Measured on the box:

| path | type | read speed |
|---|---|---|
| NFS (`/projects/…`) | NFSv3, rsize 64 KB, soft | ~0.055 GB/s |
| local NVMe (`/mnt/…`, xfs) | nvme0n1 | ~2.9 GB/s — **~53×** |

The local copy was verified before relying on it: all 11 shards size-match the NFS source,
first-64 MB byte-identical, and a full per-shard `sha256` manifest generated on the box. Recorded
end-to-end effect from a prior on-box load: **~44 min from NVMe vs ~1h41m from NFS**.

**Shipped (`516953e6`, fak):** `tools/glm52_stage_serve_dgx3.sh` + `tools/glm52_fak_native_serve.sh`
now resolve **NVMe-first** (the pattern `glm52_load_witness.sh` already used) and log the winner
loudly (`USING_LOCAL_NVME` / a `USING_PRESTAGED` slow-path warning) — so the slow NFS path can
never be taken silently again (the fail-loud safeguard). They also skip the HF re-download when a
complete copy is present. The idempotent verify+stage helper and the pinned integrity manifest live
in `fak-private` (`470b564`: `ops/stage_glm_local.sh` + `ops/glm52-UD-Q4_K_M.sha256`). Proven on
the real box: the resolver picks the NVMe shard (`USING_LOCAL_NVME`) and `sha256sum -c` against the
manifest passes. (One honest caveat: no fresh full-load *timing* was taken this pass — two peers'
GLM serves were live, so the proof rests on the read-speed delta + resolver resolution + the prior
recorded ~44 min measurement, staying off a busy box.)

This sits *above* the mmap work in priority for the cold-bench cost: NVMe-first already retires the
~70 min page-in tax that motivated the original "stage into `/dev/shm`" suggestion — staging is no
longer needed, because the NVMe copy is the fast path. The mmap/demand-paged CPU path (#974-B)
remains the right fix for the *memory-fit* wall (Finding 2), independent of where the bytes read from.

## Next steps
1. Land the llama.cpp CPU baseline on the same host (decode/prefill tok/s) — the comparison row.
2. #974-A: CPU-path memory-fit pre-flight (refuse, don't wedge).
3. #974-B: mmap/demand-paged CPU weights, or run the all-resident path on a host without the
   hugepage co-tenant to capture the fak-native CPU throughput.
4. **(#975, done)** The resident-Q5/6_K expert routing was already in the CPU loader (`6b9fbc3`);
   the `FAK_Q4K` serve path now threads a `LoadProfiler` so the load streams its per-quant-type
   load-path summary (`resident=… dequant=…`) + resident report. Re-run on the CPU host and read
   the summary: `dequant≈0` for the expert rows ⇒ the bottleneck is I/O (→ #974-B mmap), not
   dequant; a residual `dequant` slice ⇒ a routing bug to chase.

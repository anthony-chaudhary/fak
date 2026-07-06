---
title: "Qwen3.6-27B pure-fak kernel — the remaining perf levers + the CPU/SSD offload tier (SOTA survey, 2026-07-06)"
description: "A host-independent survey that reconciles the now-closed Metal child-issues into the residual Mac work, then brings three un-recorded SOTA next-levers for decode (one-command-buffer [landed] -> MTLIndirectCommandBuffer -> megakernel), the chunked WY-scan for the bigger prefill gap, and the tiered CPU/SSD weight-offload architecture — including the SSD write-wear-safety principle (offload read-only weights, never spill KV cache)."
date: 2026-07-06
---

# Qwen3.6-27B pure-fak kernel — remaining perf levers + CPU/SSD offload (2026-07-06)

**Honesty rider** ([`../proofs/00-METHOD.md`](../proofs/00-METHOD.md)): written on a `win32`
box with **no Mac, no Metal toolchain, no GPU, no 27B weights**. It contains **no new
measurements**. It is a synthesis of (a) the current GitHub issue-cluster state, (b) code
already on `main` (cited by `file:line`), and (c) external SOTA literature (cited by URL). Every
tok/s number is quoted from an existing committed artifact and attributed there; nothing here
is a fak witness. The forward-looking levers are **candidates for the Mac verify node**, not
claims of work done.

---

## 1 — Where the goal stands

The goal is the **pure-fak Metal kernel** for **Qwen3.6-27B** (`qwen35` hybrid: 48 Gated-DeltaNet
layers + periodic full-attention, MoE with a shared expert) — *working* and *performant*, with
**no llama.cpp/MLX/ggml in the execution path** ([`../benchmarks/FAK-NATIVE-QWEN35-RESULTS.md`](../benchmarks/FAK-NATIVE-QWEN35-RESULTS.md)).

**Working (correctness):** the arch math is bit-exact vs HF eager at fixture scale; end-to-end
the 27B decodes coherently and matches llama.cpp for the first two tokens, then a **token-3
near-tie argmax flip** (`8160` vs `90700`). The host-independent H1 sensitivity experiment is
**run** and establishes the culprit must have a **systematic (consistently-signed) component** —
pure reduction-order noise is ~10⁵× too small to tip the tie
([`../../experiments/qwen36/token3-h1-sensitivity-finding-2026-06-28.md`](../../experiments/qwen36/token3-h1-sensitivity-finding-2026-06-28.md)).
Instrumentation is landed (`internal/model/hiddentap.go`, the divergence probe, the sensitivity
stressor). **Residual: Mac + 27B-artifact-gated** — a per-layer llama.cpp-vs-fak capture to name
the diverging `(layer, op)`, then a host-testable fix. (Open capture: issue #7.)

**Performant:** on a clean M3 Pro, decode **1.2 tok/s** vs the **7.29 tok/s** llama.cpp-Metal bar
(3× goal 2.7); prefill (post-#1085 refresh) warm **2.6 tok/s @P=27** and **7.3 @P=940** vs the
**51.55** bar ([`MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md`](MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md);
[`../benchmarks/QWEN36-PARITY-RESULTS.md`](../benchmarks/QWEN36-PARITY-RESULTS.md) §159-165).
Root cause is **orchestration, not arithmetic**: each decode token submits ~336 separate Metal
command buffers (~7 matmuls × 48-64 layers), each ~360 µs launch/sync-bound over ~98 µs of real
bandwidth work. Proven lever: `BenchmarkMetalQ4KGemvBatch` packs 64 GEMVs into one command
buffer → **5.2× faster/GEMV, ~59% of device BW**, projecting **~5.9 tok/s (~8 with a kernel
pass)**.

**Issue-cluster state (checked 2026-07-06):** every concrete Metal child-issue is **CLOSED** —
#67 (one MTLCommandBuffer per forward), #68 (decode GEMV), #69 (resident weights), #70 (device
GEMM), #71 (hybrid prefill), #1381 (Q6_K fused expert MLP), #1382 (one-command-buffer decode,
lift `metal_decode` decline), #64 (perf gate). Only the **trackers** stay open: **#59** (Metal
epic), **#977** (decode-parity playbook), **#931** (27B on one A100). **Reading: the perf
*levers* have landed in code; the residual is the Mac end-to-end MEASUREMENT that confirms
they reach ~6 tok/s, plus the SOTA next-levers below if a gap remains.**

---

## 2 — Decode command-buffer ladder (the launch-overhead lever)

The 7.29 tok/s bar, ggml-metal, does **not** submit per op. It encodes the whole GGML graph into
a small number (`n_cb`) of command buffers with memory-range dependency tracking + graph
reordering for concurrency, and **commits once per token**
([DeepWiki: Metal backend](https://deepwiki.com/ggml-org/llama.cpp/5.2-metal-backend-(apple))).
That one-commit-per-token structure is the whole gap fak's #67/#1382 close. Three rungs, in
increasing aggressiveness — **rung A is landed in code; B and C are not recorded in any fak doc
and are the genuine next-levers:**

- **A. One command buffer per token (LANDED, #67/#1382).** Keep the f32 activation GPU-resident
  across all projection/MLP matmuls, read the resident KV cache, submit ONE command buffer/token
  — pay the ~360 µs once instead of ~336×. This matches ggml-metal's structure and is the basis
  of the ~5.9 tok/s projection. *Residual: Mac measurement.*
- **B. `MTLIndirectCommandBuffer` (ICB) — the Metal CUDA-Graphs analogue (NEW lever).** Each
  decode step re-issues an essentially **fixed dispatch topology** (per-layer matmuls, GDN scan,
  attention). ICB records that topology **once** and replays it each step, removing the CPU
  re-encoding from the decode loop; ggml-metal itself does **not** use ICB, so this is a lever
  *beyond* parity ([Apple: MTLIndirectCommandBuffer](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer),
  [Kodeco: GPU-driven rendering](https://www.kodeco.com/books/metal-by-tutorials/v2.0/chapters/15-gpu-driven-rendering)).
- **C. Megakernel / persistent kernel (aggressive).** Fuse the whole forward into one resident
  kernel — eliminates launch overhead entirely and overlaps load/compute across layers. On an M1
  Max, a fused decode kernel held **constant 10.0 tok/s** vs a baseline degrading 10.8→7.2 as
  dequant bandwidth rose ([Compiling LLMs into a MegaKernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)).
  High complexity; the GDN recurrent scan + periodic full-attention must live in-kernel.

**Profiling caveat for whoever measures B/C on the Mac:** Apple GPU frame capture *serializes*
every command, so for LLM decode (thousands of dispatches/token) a captured run is 50-100× slower
than uncaptured — measure ICB/fusion gains with lightweight timing (the existing `FAK_QPROFILE`
phase profile), **not** an Instruments GPU capture.

---

## 3 — Prefill is the *bigger* relative gap: the chunked WY-scan

Prefill sits at **~7× under** its bar even at the favorable long-prompt point (7.3 vs 51.55
@P=940) — a wider relative gap than decode. The SOTA answer for the GDN recurrence is the
**chunkwise parallel scan**: split the sequence into chunks of C≈64-128, compute intra-chunk
recurrence exactly, pass state between chunks, and — the key trick — re-express the delta-rule's
Householder products via the **WY representation** so the state update becomes block **GEMMs**
instead of a serial scan, which is what makes it saturate GPU tensor units
([Gated DeltaNet paper](https://arxiv.org/pdf/2412.06464); [FLA `chunk_gated_delta_rule`](https://github.com/NVlabs/GatedDeltaNet)).
fak's prefill scan today is the serial per-position recurrence on every path (`qwen35.go` headStep
~L490; the quantized paths `qwen35_prefill*.go` parallelize *across heads* but each head is still a
serial per-token scan). A WY-GEMM chunked scan is the prefill twin of the decode command-buffer
lever and the path to closing the 51.55 gap. **Parity caveat (corrected 2026-07-06):** a WY-GEMM
chunked scan *reorders* the delta-rule reductions, so it is **close-parity (cos≈1), not bit-exact**
with the serial recurrence — unlike the misleadingly-named `qwen35_chunked.go`, which only
GEMM-batches the 5 projections and keeps the scan **bit-identical** (`FAK_GDN_BATCHED`, f32 path).
Because the scan feeds the token-3 near-tie (§1), a reduction-order change here could *itself* shift
that argmax, so this lever must be **co-verified with the correctness capture, not landed blind**.
The algorithm is host-independent to prototype, but its value (prefill tok/s) is Mac-measured and it
touches the hot `qwen35.go` scan.

---

## 4 — The CPU/SSD offload tier (fit models past device memory)

This is orthogonal to the two levers above: they make a *resident* model fast; offload makes a
model that **does not fit** in the 36 GB unified pool (or a smaller host) *run at all*. fak
already has most of the machinery; the SOTA framing sharpens where it goes next.

**What fak has today (code-grounded):**

- **CPU offload — LANDED.** `splitKernel` ([`internal/model/moe_offload.go:25`](../../internal/model/moe_offload.go))
  is fak's `--n-cpu-moe` equivalent: it routes **per weight** — MoE expert + shared-expert GEMMs
  run host-resident on the CPU Q4_K/Q8 kernel (`isExpertWeight`, matches `.mlp.experts.` /
  `.mlp.shared_experts.`), while the small every-token dense work (attention/MLA projections,
  router, learned-index, LM head) stays on the device. It adds **no arithmetic** — a pure
  placement decision over the same math, pinned bit-exact by a placement-invariance witness. This
  is exactly the SOTA-favored split: an MoE has a **large total but small active parameter set**
  (only top-k experts fire per token), so host-resident experts cost far less than offloading a
  dense model ([llama.cpp MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)).
  It is what lets the 753B GLM-5.2 (~424 GB @ Q4_K) serve on a 1007 GB host.
- **SSD/CPU weight paging — primitives BUILT + green, not yet on serve path.** Two standalone,
  cpu-ref-tested primitives already exist: `pagedKernel`
  ([`internal/model/paging.go:27`](../../internal/model/paging.go)) — upload (page IN) → compute →
  free (page OUT), so device memory holds only the active weight, bit-equal to resident — and, on
  top of it, **`pagedRing`** ([`internal/model/paging_ring.go:43`](../../internal/model/paging_ring.go))
  — the **bounded LRU resident-weight ring** (the Tier-1 GPU hot set) that reuses a handle on a hit,
  streams a cold weight on a miss, and evicts the coldest unpinned weight under a byte budget
  (hit/pageIn/evict counters, `used()≤budget()`, bit-equal to resident either way). Both are
  standalone (off the serve path, f32, no async H2D) and pass `go test ./internal/model/`. The
  whole-model twin `internal/residency` (`Manager`) is the same polymodel LRU bound to models.
  **The remaining work is the WIRING**, tracked as **#2726** (wire the paging primitive into the live
  serve weight HAL with an on-demand disk/SSD tier) under epic **#2722** (Mac CPU+RAM+SSD offload
  serving) — which needs async/pinned H2D + a real device budget, so it is device-gated, not
  host-verifiable here.
- **Read-only weight mmap.** `mmapOpen` ([`internal/model/mmap_unix.go:34`](../../internal/model/mmap_unix.go))
  maps weights `PROT_READ`/`MAP_SHARED` — the OS demand-pages from SSD and page-caches, streaming
  a model larger than RAM with zero explicit I/O code.

**Where SOTA says this goes (the tiered expert cache).** FlexGen and ZeRO-Inference aggregate
GPU+CPU+SSD and schedule I/O to run a model far past GPU memory; both lean on 4-bit quant to cut
the I/O volume ([FlexGen](https://arxiv.org/html/2303.06865),
[ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/README.md)).
The concrete design fak's `pagedKernel` should grow into is llama.cpp's proposed **3-tier expert
cache** ([llama.cpp #20757](https://github.com/ggml-org/llama.cpp/issues/20757)): **Tier-1 GPU
VRAM** (a small persistent hot-expert slot ring), **Tier-2 CPU *pinned* RAM** (backing store for
all local experts — pinned so H2D copies overlap GPU compute), **Tier-3 SSD/mmap** (the full
weight tensor, demand-paged). On a Tier-1 miss copy from pinned RAM; on a Tier-2 cold miss
demand-page from the mmap'd file. fak already has **Tier-1** (`pagedRing`) and **Tier-3** (read-only
`mmapOpen`); the missing pieces are the **Tier-2 pinned-CPU-RAM** backing store and the async wiring
that binds all three to the serve weight HAL (#2726).

---

## 5 — SSD write-wear safety (the explicit caveat, grounded)

The wear concern has a clean, SOTA-backed answer: **weight offload is safe; KV-cache offload is
where the wear (and the perf collapse) lives.**

- **Weight offload is write-once / read-many.** A block-layer I/O study of DeepSpeed + FlexGen
  found offload traffic is **read-dominated, large-block (~128 KiB reads)** — weights are written
  once and read repeatedly, so weight-offload imposes **minimal NAND write-wear**
  ([CHEOPS '25 I/O study](https://atlarge-research.com/pdfs/2025-cheops-llm.pdf)). fak's `mmapOpen`
  is `PROT_READ` — it **writes nothing**, so weight streaming causes **zero** wear by
  construction. This is the safe tier.
- **KV-cache offload is the wear/thrash trap — keep KV off SSD.** KV offload writes heavily during
  prefill (that is where NAND write pressure concentrates), and naive mmap/page-cache tiering
  **collapses** in decode: autoregressive reuse of the growing KV working set fights LRU eviction,
  pages are evicted before reuse, and the cache goes "full yet ineffective" driving sustained disk
  I/O (one DeepSpeed config saw a **97% slowdown** once KV spilled to SSD)
  ([CHEOPS '25](https://atlarge-research.com/pdfs/2025-cheops-llm.pdf),
  [Dual-Blade](https://arxiv.org/html/2604.26557)). **Design rule: SSD is a read-only weight
  backing store only. Never spill the KV cache or activations to SSD** — keep KV in GPU/unified or
  CPU RAM; if it will not fit, shrink it (quantized KV, shorter context, fewer parallel sequences)
  rather than paging it to flash.
- **Wear-minimizing corollaries:** prefer read-only `mmap` (or `O_DIRECT` reads) for weights over
  any write-back scheme; 4-bit quant cuts the bytes moved (and thus both latency and any wear);
  if a future KV tier is ever unavoidable, kernel-bypass NVMe with placement control (ZNS/FDP)
  is the frontier that manages write amplification — but for fak's targets it is out of scope,
  because the rule above keeps writes off the SSD entirely.

---

## 6 — Prioritized next steps

**Ground truth (re-verified against the code 2026-07-06):** the offload PRIMITIVES are already built
and green — `splitKernel` (wired + serving), `pagedKernel`, `pagedRing` (LRU ring),
`residency.Manager`, and the bit-exact paged-KV evict/swap/reserve. So a "build a hot-expert ring"
task is **already done** (`pagedRing`); what remains is device- or Mac-gated:
1. **Wire `pagedRing`/`pagedKernel` into the serve weight HAL (#2726, epic #2722)** — the paged twin
   of `weightHALQ4K/Q8` with async/pinned H2D and a real device budget. Its *policy/contract* is
   host-testable (as `pagedRing` already proves), but the async H2D + a >VRAM serve are device-gated,
   and it touches the hot session weight path (collision risk). The **Tier-2 pinned-CPU-RAM** backing
   store is the smallest remaining host-shaped sub-piece.
2. **Chunked WY-GEMM GDN scan (prefill)** — *not* a clean host-win: it is close-parity (not
   bit-exact), perturbs the token-3 near-tie (§3), and touches the hot `qwen35.go` scan; prototype
   off-Mac but co-verify with the correctness capture and measure on-Mac.

Mac + artifact-gated (the true residuals):
3. **Measure #67/#1382 end-to-end** — confirm the one-command-buffer decode reaches the projected
   ~5.9-8 tok/s; this is the single missing datum that would close the perf story or expose the
   next gap. (§1, §2A)
4. **Token-3 per-layer capture (#7)** — llama.cpp-vs-fak hidden-state dump to name the systematic
   `(layer, op)`, then the host-testable fix. (§1)
5. **ICB decode replay (§2B)** and, if still short, the **megakernel (§2C)** — only after step 3
   shows the residual.

## Sources

- Internal: [`FAK-NATIVE-QWEN35-RESULTS.md`](../benchmarks/FAK-NATIVE-QWEN35-RESULTS.md),
  [`MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md`](MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md),
  [`QWEN36-PARITY-RESULTS.md`](../benchmarks/QWEN36-PARITY-RESULTS.md),
  [`token3-h1-sensitivity-finding-2026-06-28.md`](../../experiments/qwen36/token3-h1-sensitivity-finding-2026-06-28.md);
  code `moe_offload.go`, `paging.go`, `mmap_unix.go`, `hiddentap.go`, `qwen35.go`.
- [DeepWiki — llama.cpp Metal backend](https://deepwiki.com/ggml-org/llama.cpp/5.2-metal-backend-(apple))
- [Apple — MTLIndirectCommandBuffer](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer)
- [Kodeco — GPU-driven rendering (ICB)](https://www.kodeco.com/books/metal-by-tutorials/v2.0/chapters/15-gpu-driven-rendering)
- [Compiling LLMs into a MegaKernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)
- [Gated DeltaNet (chunked WY scan)](https://arxiv.org/pdf/2412.06464) · [NVlabs/GatedDeltaNet + FLA kernels](https://github.com/NVlabs/GatedDeltaNet)
- [FlexGen](https://arxiv.org/html/2303.06865) · [ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/README.md) · [llama.cpp MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) · [llama.cpp 3-tier expert cache #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- [CHEOPS '25 — I/O characterization of LLM SSD offload](https://atlarge-research.com/pdfs/2025-cheops-llm.pdf) · [Dual-Blade — NVMe KV offload](https://arxiv.org/html/2604.26557)

---
title: "CONCEPT-STUDY: carloslfu/slotstream Apple Silicon SSD expert streaming, prefill sweeps, and extend-only GDN prefix caching for Qwen3.8"
description: "Exhaustive, pinned study of carloslfu/slotstream for running Qwen3.8-Flash-Next (125B MoE) on Apple Silicon Macs: QD32 asynchronous pread expert streaming, scan-resistant prefill sweeps, extend-only GDN prefix caching, and elastic memory governing."
date: 2026-09-03
---

# CONCEPT-STUDY: carloslfu/slotstream Apple Silicon SSD expert streaming, prefill sweeps, and extend-only GDN prefix caching for Qwen3.8 (2026-09-03)

**Verdict:** `carloslfu/slotstream` is an open-source Swift 6 + MLX runtime engineered to run the 125B-parameter hybrid MoE model **Qwen 3.8 Flash-Next** (105 GB on disk at 4-bit) on Macs with standard memory (16 GiB to 48 GiB). Standard runtimes attempt to map or allocate the full model in memory, immediately triggering catastrophic macOS swap thrashing (48 GB swap) and system crashes. Slotstream introduces key systems mechanisms directly applicable to `fak`'s Apple Silicon Metal engine:

1. **QD32 Asynchronous `pread` Expert Streaming:** Keeps only the 3.8 GB dense trunk resident in RAM, streaming 4-bit routed experts on-demand from SSD through a preallocated unified memory slot pool via 32 parallel `pread` worker lanes, achieving 17.3 GB/s direct SSD throughput and 12+ tok/s warm decode.
2. **Scan-Resistant Prefill Sweep (`gather_qmm_rhs`):** For prompts $\ge 256$ tokens, token rows are sorted by expert and staged in groups of 32 for grouped GEMMs (`sortedIndices: true`), reading each expert's weights once per tile rather than once per token (doubling prefill speed: 91 $\to$ 184 tok/s). Crucially, the sweep never touches the decode expert cache, ensuring long prompts cannot evict warm conversational state.
3. **Extend-Only GDN Linear Recurrence Prefix Cache:** Caches Gated DeltaNet recurrent states across multi-turn sessions with strict prefix validation (`prompt.starts(with: heldIds)`), avoiding invalid backwards rewinds on irreversible linear recurrence folds and reducing turn-8 TTFT from 25.8s to 6.0s across coding agent conversations.
4. **Elastic Memory Governor with Mach Probing:** Continuously checks Mach VM reclaimable memory (`vm_stat` free + purgeable + file-backed pages) and automatically clamps memory footprint between requests to prevent host OOM panic, guaranteeing byte-identical generation across cache resizes.
5. **State-Recording Speculative Rollback:** Preserves intermediate linear recurrent states at each token position during speculative verification, swapping directly to the verified state on draft rejection and avoiding redundant forward passes.

---

## 1. Scope, Provenance, and Durable Receipt

Observed and pinned on **2026-09-03**:

| Repository | Pinned Revision | License | Notes |
|---|---|---|---|
| [`carloslfu/slotstream`](https://github.com/carloslfu/slotstream) | `53028cf321123d3e9ba95adea40c4c582100ce13` | MIT | Single Swift binary running Qwen3.8-Flash-Next on Apple Silicon via SSD expert streaming and MLX Metal kernels. |

- **Parent Epic:** #10960 (Track 3: Non-Standard Silicon)
- **Research Tracker Issue:** #10956
- **Child Issues:** #10991, #10994
- **Durable Study Receipt:** `study_5c77965f0f47441af377018a5c4b58093989021620df6d0706a5dba4263d677c` (persisted via `fak study add`)

**License boundary:** MIT License (Copyright (c) 2026 Carlos Galarza). Clean-room porting and adaptation into `fak`'s Go / Metal codebase (`internal/metalgemm/` and `internal/model/`) is fully permitted with standard attribution in `NOTICE`.

---

## 2. Worldview Reconstruction: Who They Built It For & Tradeoffs

1. **Who they built this for:**
   - **Developers with standard Apple Silicon Macs (16–48 GiB RAM):** Engineers who want to run frontier 125B+ MoE models locally without buying high-end 128–512 GiB Mac Studios.
   - **Autonomous agent harnesses (Claude Code, fx, opencode):** Multi-turn coding sessions requiring high context capacity, low TTFT across follow-up turns, and strict zero-crash memory safety while IDEs and browsers run concurrently.
2. **What they optimized:**
   - **Resident RAM footprint vs. SSD bandwidth:** Recognizes that modern Mac NVMe SSDs read at 10–17 GB/s over PCIe. Rather than requiring 105 GB of unified memory, streaming active experts (10 of 512 per token) at decode time needs only a ~20 GB cache pool.
   - **Zero-swap guarantee:** The system sizes itself to what the machine can spare and yields memory dynamically when other apps open.
3. **Tradeoffs vs. fak:**
   - *Model-Specific Engine vs. Universal Agent Kernel:* Slotstream is hard-coded specifically for `qwen3.8-flash-next:4bit`; fak is an agent kernel supporting multiple backends, architectures, and capability policies.
   - *Swift vs. Go:* Slotstream is written in Swift 6 calling MLX C++; fak is pure Go calling direct Metal compute pipelines.

---

## 3. Subsystem Analysis & Key Mechanisms

### A. Asynchronous QD32 `pread` Expert Streaming Queue
*Source:* `Sources/Slotstream/ExpertStore.swift:74-125@53028cf321123d3e9ba95adea40c4c582100ce13`

Each routed expert consists of 9 tensor pieces:
$$\text{gate/up/down} \times \text{weight (U32)} / \text{scales (BF16)} / \text{biases (BF16)}$$
Totaling ~2.7 MB per expert record. For decode and small passes under the sweep threshold, reads are latency-bound. Slotstream uses a dedicated queue depth of 32 (`SLOTSTREAM_POOL_QUEUE_DEPTH=32`) to parallelize $9n$ reads across worker threads using non-blocking positional `pread`:

```swift
public func readBatch(_ keys: [ExpertKey], queueDepth: Int = ExpertStore.poolQueueDepth) -> [MLXArray] {
    let n = keys.count
    let buffers = allocateStaging(rows: n)
    let jobs: [(piece: Int, slot: Int)] = (0 ..< n).flatMap { s in (0 ..< 9).map { (piece: $0, slot: s) } }
    let lanes = min(max(queueDepth, 1), jobs.count)
    // Dispatch parallel pread across lanes into preallocated unified memory staging
}
```

This sustains 17.3 GB/s throughput from SSD directly into GPU-accessible unified memory.

### B. Scan-Resistant Prefill Sweep with Grouped GEMMs
*Source:* `Sources/Slotstream/ExpertStore.swift:105-180@53028cf321123d3e9ba95adea40c4c582100ce13`

For prompt prefill passes $\ge 256$ tokens, individual pool lookups are bypassed:
1. Tokens are sorted by expert index across each layer.
2. Experts are streamed in staging groups of 32 (`SLOTSTREAM_EXPERT_LOAD_BATCH=32`).
3. Grouped matrix multiplication is invoked with `sortedIndices: true` (`gather_qmm_rhs`), allowing the Metal kernel to read expert weights once per tile of tokens instead of once per token.
4. Prefill never admits to or evicts from the decode expert pool, ensuring scan resistance: long prompts cannot evict the conversation's active decode state.

### C. Extend-Only GDN Linear Recurrence Prefix Cache
*Source:* `Sources/Slotstream/PrefixCache.swift:40-115@53028cf321123d3e9ba95adea40c4c582100ce13`

Linear attention (Gated DeltaNet) maintains an accumulated state matrix that represents an irreversible fold over all past tokens. Unlike traditional softmax attention where KV pairs can be partially trimmed:
- A linear recurrent state can only be extended: `prompt.starts(with: heldIds)` and `prompt.count > heldIds.count`.
- Backward rewind or partial splicing is mathematically impossible without full recomputation.
- Slotstream tracks exact consumed token IDs (sampling occurs before feeding, so trailing unconsumed tokens are carefully accounted for).
- Retains up to 4 conversation sessions so ancillary agent requests (e.g. title generation or tool pinging) do not thrash the primary session cache.

### D. Elastic Memory Governor with Mach Probing
*Source:* `Sources/Slotstream/Governor.swift:30-95@53028cf321123d3e9ba95adea40c4c582100ce13`

The governor polls Darwin system memory every 15s via Mach VM statistics:
$$\text{Reclaimable} = \text{free} + \text{purgeable} + \text{file-backed pages}$$
The memory target is clamped conservatively:
$$\text{Target} = \min(33\text{ GB}, 0.70 \times \text{RAM}, \text{Metal working set} - 2\text{ GB})$$
When external memory pressure rises, the governor sheds prefix caches first, then shrinks the expert pool between requests. Generation remains bit-identical across resizes.

---

## 4. Current fak Witness & Gap Matrix

| Slotstream Mechanism | fak Equivalent | Current fak Witness | On-Axis Gap & Disposition | Filed Issue |
|---|---|---|---|---|
| **QD32 Async `pread` Expert Streaming** | `internal/metalgemm/q4k.m`, `internal/model/weights.go` | `q4k.m:1663-1727`, `weights.go:387` | **ABSENT → DEFAULT**. FAK requires weights to be resident or pre-mapped; lacks an asynchronous SSD expert streaming queue for MoE. | #10991 |
| **Scan-Resistant Prefill Sweep (`gather_qmm_rhs`)** | `internal/metalgemm/q4k.go` | `q4k.go:124-169`, `qwen35_prefill_q4k.go:13-36` | **ABSENT → DEFAULT**. FAK's Q4_K Metal prefill lacks expert-sorted grouped GEMM and does not decouple prefill staging from decode pool cache. | — |
| **Extend-Only GDN Recurrent Prefix Cache** | `internal/model/kvcache.go`, `internal/ctxmmu` | `kvcache.go:60-95`, `pagedkv.go:256-276` | **PARTIAL → DEFAULT**. FAK's `KVCache.Evict` explicitly refuses GDN recurrent state and does not provide an extend-only snapshot cache for multi-turn agent turns. | #10994 |
| **Mach-Probed Dynamic Elastic Memory Governor** | `internal/policy`, `internal/hostinfo` | Issue #9587 (`reserve aggregate Apple unified memory by bytes and pressure`) | **PARTIAL → DEFAULT**. FAK inspects static `hw.memsize` but lacks continuous Mach reclaimable memory polling and dynamic pool resizing. | #9587 |
| **State-Recording Speculative Rollback** | `internal/model/speculative_state.go` | `speculative_state.go:15-87`, Issue #9958 | **PARTIAL → DEFAULT**. FAK copies Go-owned state on speculative verify; recording intermediate linear states eliminates redundant re-forwarding on rejection. | #9958 |

---

## 5. Candidate Disposition Matrix

| Candidate Borrow | Source Anchor | Axis Optimized | fak Seam | Disposition | Filed Issue |
|---|---|---|---|---|---|
| QD32 async `pread` routed-expert streaming queue | `Sources/Slotstream/ExpertStore.swift:74-125` | RAM footprint (runs 105 GB MoE in 16–33 GB) | `internal/metalgemm/q4k.m` | **DEFAULT** | #10991 |
| Scan-resistant prefill sweep with sorted grouped GEMM | `Sources/Slotstream/ExpertStore.swift:105-180` | Prefill throughput (91 $\to$ 184 tok/s) & cache retention | `internal/metalgemm/q4k.go` | **DEFAULT** | — |
| Extend-only GDN linear recurrent prefix cache | `Sources/Slotstream/PrefixCache.swift:40-115` | Multi-turn agent TTFT (25.8s $\to$ 6.0s on turn 8) | `internal/model/kvcache.go` | **DEFAULT** | #10994 |
| Mach reclaimable memory governor | `Sources/Slotstream/Governor.swift:30-95` | System stability & zero-OOM execution | `internal/policy/` | **DEFAULT** | #9587 |
| State-recording speculative rejection rollback | `Sources/Slotstream/MTP.swift:85-150` | Speculative verify overhead (eliminates re-forward pass) | `internal/model/speculative_state.go` | **DEFAULT** | #9958 |

---

## 6. Registration and Checkable Next Steps

- **Durable Study Receipt:** `study_5c77965f0f47441af377018a5c4b58093989021620df6d0706a5dba4263d677c` (persisted via `fak study add`)
- **Monitored Repository Registry:** Added `carloslfu/slotstream` to `docs/research/monitored-repositories.json` as `studied`.
- **First Checkable Steps:**
  1. **#10994 (Extend-only GDN prefix cache):** Implement the extend-only GDN recurrent prefix cache invariant in `internal/model/kvcache.go` with unit tests verifying prefix-matching retention and refusal on non-prefix prompts.
  2. **#10991 (QD32 async pread expert streaming queue):** Implement parallel `pread` expert streaming into unified memory slots in `internal/metalgemm/q4k.m` / `internal/model/weights.go`.
  3. **#9587 (Mach reclaimable memory governor):** Wire Mach VM statistics polling into memory reservation policies to dynamically bound model working sets.


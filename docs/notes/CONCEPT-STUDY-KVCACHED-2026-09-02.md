# Concept study: ovg-project/kvcached

**Verdict:** kvcached is an OS-style virtual memory management system for LLM KV caches (tested across vLLM and SGLang on CUDA and ROCm/HIP), bringing virtual address reservation (`cuMemAddressReserve`), demand-paged physical backing (`cuMemCreate` + `cuMemMap`), page-aligned eviction, and bounded prefix caching to shared GPUs. At pinned revision `60cad949389af6bbf1d65c4eddf325113df5a9eb`, it demonstrates that decoupling virtual KV capacity from physical VRAM eliminates startup-time static allocation and enables elastic multi-model colocation (2–28× TTFT reduction under intermittent peak workloads). Its deepest value to fak is its **native CUDA VMM memory architecture**, its **hard-won negative knowledge** (compound-page layouts cause a 30–50% attention throughput regression; prealloc thread page-table edits cause CUDA error 700 on active SMs unless split into off-stream prepare and on-stream commit; idle memory is dominated by page spread rather than block count), and its **page-granular offload and eviction mechanics**. Four new issues were filed (#10720, #10721, #10722, #10723).

- **Source:** `https://github.com/ovg-project/kvcached`
- **Pin:** `60cad949389af6bbf1d65c4eddf325113df5a9eb` (HEAD, 2026-08-22, "fix(vllm): preserve inference mode during memory profiling (#461)")
- **Observed:** 2026-09-02
- **Filed issues:**
  - #10720: `feat(compute): CUDA VMM virtual-to-physical reservation for native in-kernel KV cache (epic #2236)`
  - #10721: `feat(radixkv): page-aligned eviction strategy to reduce physical memory fragmentation (epic #2236)`
  - #10722: `feat(compute): page-granular CPU offloading for in-kernel KV cache (epic #2236)`
  - #10723: `feat(compute): zero-copy prefix sharing on cudaKV.Clone via block-table indirection (epic #2236)`
- **Durable receipt:** `study_7f18852854b61725455e62ba7e29fbbc289a2e1775ca418021c0cfcfee7703ff`
- **Parent epic:** #2236 (`epic(superset): fak > best of vLLM + SGLang + Dynamo + TRT-LLM/LMDeploy/llama.cpp`)

---

## Worldview

kvcached was built by researchers and systems engineers at the Open Virtual GPU project (OVG / UCLA / UIUC / Red Hat Sardeenz) to solve the **GPU memory cost crisis in multi-model LLM serving**.

Current serving engines (vLLM, SGLang) statically allocate a fixed, large fraction of GPU memory (e.g. 90% via `--gpu-memory-utilization 0.9`) at startup based on profiling, rigidly partitioning the GPU. If two or three models are deployed on an A100-80G, each must be clamped to ~30% of memory even if only one model is actively processing requests at any moment. This rigid partitioning causes high queuing latency, low GPU utilization under bursty/intermittent traffic, and forces operators to over-provision GPU hardware.

kvcached brings an OS-inspired solution:
- **Decouple virtual addressing from physical memory**: reserve a full virtual address range (e.g. 64k tokens per model) at startup, but map physical GPU memory in 2 MB chunks only when tokens are actively appended.
- **Dynamic memory sharing**: multiple serving engines running on the same GPU draw from the same unallocated physical VRAM pool dynamically.
- **Zero-touch adoption**: integrate into stock vLLM and SGLang runtimes without requiring forked engines or custom entrypoint wrappers, via `site-packages` `.pth` auto-hooks.

---

## Shipped at the pin versus emerging direction

- **Shipped at `60cad94`:**
  - C++ `PageAllocator` (`csrc/page_allocator.cpp`) with preallocation background thread maintaining 5–10 pre-mapped hot pages.
  - CUDA/HIP VMM abstraction (`csrc/inc/gpu_vmm.hpp`) wrapping `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, and `cuMemUnmap`.
  - Zero-page mapping upon unmap (`csrc/ftensor.cpp:135-136`) ensuring that deallocated VA slots read zeros rather than triggering memory faults or leaking stale context.
  - Runtime monkeypatch integration for vLLM (v0.8.4 to v0.24.0) via `ElasticBlockPool` and virtual-budget calculation in `GPUWorkerPatch`.
  - Runtime monkeypatch integration for SGLang (v0.4.9 to v0.5.15) via `ElasticTokenToKVPoolAllocator`, `ElasticMHATokenToKVPool`, `ElasticMLATokenToKVPool`, `ElasticMambaPool`, and `RadixCacheLimitPatch`.
  - Prefix caching under elastic memory with configurable token cap (`KVCACHED_MAX_CACHED_TOKENS`) and page-aligned victim eviction (`_page_aligned_victims`).
  - POSIX shared memory (`/dev/shm`) usage tracking and interactive `kvctl`/`kvtop` CLI tools.
  - Revisioned instance memory-limit API (`kvcached/control.py`, PR #414).
  - Cross-layer KV sharing support for heterogeneous attention models (Gemma 3/4 sliding-window + full attention).
- **Emerging upstream PRs & issues (open direction):**
  - PR #450: Page-granular CPU offloading foundation (offloading complete 2 MB KV page bundles to pinned host memory and restoring transactionally).
  - PR #465: Two-phase VMM allocation splitting `cuMemCreate` (off-stream prepare) from `cuMemMap` (on-stream commit) to avoid illegal memory access faults on active SMs.
  - Issue #470: Safe cross-process admission design to prevent `cuMemCreate` OOM races between colocated serving instances.
  - Issue #467 / PR #465: Deferring physical unmap until async GPU batches complete.

---

## Deep inventory

Four subsystems were investigated in depth across the codebase:

### 1. Python Core Library (`kvcached/`)
- Two-level memory hierarchy: 2 MB physical pages (`KVCACHED_PAGE_SIZE_MB`, must be a multiple of 2 MB) subdivided into engine-level KV blocks (`kvcached/kv_cache_manager.py:7-10`).
- Best-fit page selection algorithm: `_pick_avail_page` scans partially-filled pages and picks the smallest page that accommodates the remaining run, preventing prefill allocations from smearing across multiple pages and pinning excess physical memory (`kvcached/kv_cache_manager.py:426-460`).
- Allocation lifecycle: `_alloc` polls external resize targets from SHM, allocates blocks from active pages, and on exhaustion invokes C++ `PageAllocator.alloc_page()` to map physical backing. Partial allocations on failure roll back cleanly (`:366-424`).
- Tensor Parallelism (TP) IPC: driver broadcasts map/unmap commands to worker processes over Unix domain sockets (`/tmp/kvcached-tp-<IPC>-<hash>/w<rank>.sock`) with a 60s timeout to avoid worker hangs (`kvcached/tp_ipc_util.py:16-156`).
- Prefix caching support: `ElasticBlockPool` implements vLLM's BlockPool APC interface with an LRU `_evictable_blocks` OrderedDict and page-aligned eviction (`_page_aligned_victims`, `kvcached/integration/vllm/patches.py:699-734`).
- Revisioned limit control: `set_memory_limit` uses an integer revision to ensure idempotent updates and handles deferred shrink while blocks remain in use (`kvcached/kv_cache_manager.py:572-646`).

### 2. CUDA/C++ Layer (`csrc/`)
- Pure host-side memory management: contains zero `.cu` attention kernels. Manufactures flat 1-D `at::Tensor` views over reserved VA ranges via `at::from_blob` (`csrc/ftensor.cpp:62-76`).
- Driver VMM orchestration: `cuMemAddressReserve` for the whole virtual range at base `0x1f0'000'000'000` with 2 MB alignment (`csrc/ftensor.cpp:26-47`); initialized by mapping the entire range to a shared zero page (`:160-176`).
- Mapping: `GPUPage::map` calls `cuMemCreate` then `cuMemMap` + `cuMemSetAccess` (`csrc/page.cpp:19-25`). `FTensor::map` defensively unmaps the zero page before mapping the physical page (`csrc/ftensor.cpp:111-113`).
- Unmapping safety: `FTensor::unmap` immediately re-maps the shared zero page into the hole (`csrc/ftensor.cpp:135-136`). A kernel reading past the valid block table reads zeros rather than crashing with an illegal address fault or reading stale context data.
- Hot page preallocation: background `prealloc_worker` thread maintains `reserved_page_list_` between 5 and 10 pre-mapped pages (`csrc/page_allocator.cpp:588-669`). `alloc_page` fast path pops an already-mapped page in ~7 µs.
- GIL release discipline: blocking C++ bindings release the GIL via `py::call_guard<py::gil_scoped_release>()` to prevent deadlocks with Python worker IPC callbacks (issue #371, `csrc/torch_bindings.cpp:226-244`).

### 3. Serving-Engine Integration (`kvcached/integration/`)
- Zero-touch autopatch: `kvcached_autopatch.pth` at site-packages root registers `wrapt.when_imported` post-import hooks on `"vllm"` and `"sglang"` at Python startup (`kvcached_autopatch.pth:1`).
- Individual patch gating: every patch checks `ENABLE_KVCACHED=true` dynamically; without this env var, patches are completely inert (`kvcached/integration/patch_base.py:20-21`).
- vLLM integration seams:
  - Block pool: `ElasticBlockPool` injected into `vllm.v1.core.block_pool` (`kvcached/integration/vllm/patches.py:450-939`).
  - Allocation: `GPUModelRunner._allocate_kv_cache_tensors` replaced with VMM allocation sized to device total memory (`:1417-1782`).
  - Profiling: `Worker.determine_available_memory` replaced with virtual budget calculation `virtual_budget - weights - torch_peak`, hard-zeroing non-torch deltas (`:2041-2148`). Commit `60cad94` preserves `@torch.inference_mode()` on the patched method to prevent gradient memory accumulation.
  - Scheduler miss: `KVCacheManager.allocate_slots` translates `KVCachePoolExhausted` into returning `None`, converting physical VRAM exhaustion into a scheduler retry/preemption rather than an engine crash (`:2151-2221`).
- SGLang integration seams:
  - Class aliasing: `ElasticTokenToKVPoolAllocator` and `ElasticMHATokenToKVPool` aliased over module globals in `sglang.srt.mem_cache` (`kvcached/integration/sglang/patches.py:150-844`).
  - RadixCache: wraps `cache_finished_req` to evict down to `KVCACHED_MAX_CACHED_TOKENS` (`:1505-1544`).
  - Leak detector: neutralized via source-code inspection monkeypatching because kvcached's lazy mapping breaks SGLang's static-pool invariant (`:1390-1470`).

### 4. Control Plane & Operations Surface (`controller/`, `benchmarks/`, `examples/`)
- Process model: `controller/launch.py` uses tmux sessions to manage independent OpenAI-compatible server instances per engine.
- HTTP routing: `controller/frontend.py` provides `/v1/chat/completions` proxying with model-name routing, traffic statistics, and sleep management.
- Sleep mode: `controller/sleep_manager.py` calls engine APIs (vLLM `POST /sleep`, SGLang `POST /release_memory_occupation`) on idle timeout, reclaiming KV cache while leaving model weights resident. Wake latency is ~0.12s.
- CLI: `kvctl` and `kvtop` interact with `/dev/shm` 24-byte `MemInfoStruct` (`total_size`, `used_size`, `prealloc_size`) protected by `flock`.

---

## Candidate decisions

| Candidate Borrow | Axis | Their-worldview reason | FAK on-axis witness | Route | Issue |
|---|---|---|---|---|---|
| **CUDA VMM virtual-to-physical reservation for in-kernel KV cache (`cudaKV`)** | VRAM allocation & reallocation overhead for device KV cache (`cudaKV.growAppend`) | Dynamic multi-model colocation without rigid VRAM partitioning; VMM allows reserving logical 64k tokens per model | ABSENT. `internal/compute/cuda.go:1099-1116` uses `dslice` realloc-and-copy (`dallocKV` + `fcuda_d2d` + `fcuda_free`), doubling VRAM peak during growth and forcing fixed `cudaKVMaxPos=1024` under CUDA graph capture. `grep` for `cuMemAddressReserve` in `internal/compute/` is 0. | ADAPT | **#10720** |
| **Page-aligned eviction strategy in radixkv** | Physical VRAM reclaimed per eviction (fragmentation resistance) | On high-layer models, 1 retained block per page pins 144 MB; page-aware eviction freed 0.88 GB vs 0.03 GB for pure LRU | ABSENT. `internal/radixkv/eviction_strategy.go:109-113` supports only `lru`, `cost-aware`, and `slru` (from SGLang), all ranking leaves strictly on scalar `(seg, cost, age)` without considering backing chunk occupancy. | ADAPT | **#10721** |
| **Page-granular CPU offloading for in-kernel KV cache** | Granularity of device-to-host KV cache tiering (page-level vs whole-store evacuation) | Multi-model agent pipelines need to swap idle prefix pages to host RAM without evacuating entire active model state | PARTIAL. `internal/compute/kvhost.go:11,88-105` provides only `SnapshotKVToHost` and `RestoreKVFromHost`, which are bulk all-or-nothing operations over the entire `KVStore`. No page-granular swap exists. | ADAPT | **#10722** |
| **Zero-copy prefix sharing on `cudaKV.Clone` via block-table indirection** | VRAM consumption of forked / branched agent sessions (`cudaKV.Clone`) | Prefix caching across requests shares common prompt blocks; branched agent exploration sessions share identical prefixes | PARTIAL. `internal/compute/cuda.go:1251-1275` comment explicitly notes `Clone` deep-copies all VRAM across all layers (`dallocKV` + `fcuda_d2d`). For N branches, VRAM scales as O(N * prefix_len). | ADAPT | **#10723** |
| **Two-phase VMM allocation (`prepare` physical vs `commit` VA map)** | Safety of concurrent memory allocation alongside active CUDA kernels | Avoid CUDA error 700 on active SMs caused by page-table edits during live kernel execution | ABSENT. Critical concurrency invariant to be incorporated into #10720 implementation. | INSPIRE | Folded into **#10720** |
| **Contiguous compound-page KV layout (30–50% throughput regression)** | KV layout stride vs attention kernel memory access pattern | Amortize VMM driver calls (1 call for all layers); proved to cause 30–50% throughput regression in FlashAttention | PRESENT-on-axis. Fak's `cudaKV` already uses separate per-layer split K/V buffers (`k.K = make([]dslice, cfg.NumLayers)` in `internal/compute/cuda.go:1068`), avoiding this regression entirely. | EXCLUDE | — |
| **In-process `.pth` monkeypatching vs out-of-process wire gateway** | Engine integration seam (in-process monkeypatching vs wire proxy) | Zero-touch adoption for Python serving engines; paid for by ~2,300 lines of upstream engine ABI drift shims | DIVERGENT. Fak integrates out-of-process via standard HTTP wires (OpenAI / Anthropic `/v1`) and CLI wrappers (`fak manage`), isolating fak from Python ABI drift and preserving language-neutral security. | EXCLUDE | — |
| **Optimistic revisioned session limits** | Revisioned optimistic concurrency for dynamic memory budget updates | Concurrent multi-pool limit adjustments with stale/conflict reporting (`kvcached/control.py:13-95`) | PRESENT-on-axis. `fak session` CLI and backend already support `--if-rev N` optimistic-concurrency guard on live session control, budget envelopes, and priority updates. | EXCLUDE | — |
| **Device allocation recovery without process termination** | Handling out-of-memory without crashing serving process | `KVCachePoolExhausted` caught and returned as scheduler miss (`None`) | PRESENT-on-axis. `internal/compute/cuda.go:271` panics with `DeviceAllocError`, which `internal/agent/inkernel_planner.go:458,512` recovers cleanly. | EXCLUDE | — |

---

## Negative knowledge & engineering lessons

1. **The compound-page layout trap (30–50% attention throughput regression):** kvcached initially designed a "contiguous" layout grouping K/V across all layers into a compound page (`page_size * num_layers * num_kv_buffers`) to execute a single VMM driver call per block. However, nsys profiling in `benchmarks/bench_layout/README.md:28-96` revealed that this introduces a large stride (`num_layers * 64 KB = 1.75 MB`) across FlashAttention block reads, causing a 30–50% throughput regression in `flash_fwd_splitkv_kernel`. Flipping to non-contiguous per-layer layout closed the gap to -1%. Fak's choice of per-layer `dslice` buffers in `internal/compute/cuda.go:1068` is validated by this finding.
2. **Prealloc thread VA mapping faults active kernels (CUDA error 700):** In PR #465, unmapping the shared zero page and mapping a real page over a compound page while attention kernels were running on Volta/Turing GPUs triggered `CUDA error: an illegal memory access (700)`. Page-table modifications (`cuMemMap`/`cuMemUnmap`) must be synchronized with compute streams, whereas physical page creation (`cuMemCreate`) can safely run asynchronously on background threads.
3. **Idle memory is dominated by page spread, not block count:** In `benchmarks/bench_idle_footprint/README.md`, retaining 1 block per 2 MB page across a 36-layer model pinned 144 MB of VRAM per page ID. Pure LRU eviction freed only 0.03 GB under fragmented load because it evicted blocks without emptying complete pages. Page-aligned eviction freed 0.88 GB by prioritizing blocks whose removal emptied entire physical pages.
4. **Cross-process allocation admission races:** In issue #470, two colocated TP=4 serving instances under high concurrency repeatedly observed sufficient free memory, simultaneously called `cuMemCreate`, and raced into raw CUDA driver OOM. While transactional rollback prevented fatal crashes, it resulted in severe request starvation. A process-shared lock or coordinator is required for safe concurrent physical growth.
5. **C++ allocator thread joins without dropping the GIL cause deadlocks:** In issue #371, C++ prealloc threads blocked in Python worker-IPC callbacks while a caller held the GIL inside a binding. All blocking allocator bindings must use `py::call_guard<py::gil_scoped_release>()` and thread joins must conditionally release the GIL.
6. **Zero-page re-mapping prevents faults and stale secret leaks:** Instead of leaving unmapped VA ranges as invalid holes, kvcached remaps a shared zero page into freed slots (`csrc/ftensor.cpp:135-136`). Any kernel racing past the block table reads zeros rather than triggering an illegal memory access or reading stale data from previous sessions.

---

## Evidence limits & completeness critic

The entire tracked repository at commit `60cad949389af6bbf1d65c4eddf325113df5a9eb` was examined.
Parallel deep readers covered:
- `kvcached/` Python core library (allocator, IPC, lifecycle, prefix caching, control plane).
- `csrc/` CUDA/C++ layer (driver VMM APIs, page allocator, zero-page aliasing, thread safety).
- `engine_integration/` and `kvcached/integration/` (vLLM and SGLang patches, autopatch `.pth`, NIXL compat).
- `controller/`, `tools/`, `examples/`, `benchmarks/`, `tests/`, `docs/`, `docker/`, `.github/` (process model, sleep mode, CLI, benchmarks, test manifests, CI workflows).

Materially unopened paths:
- `.gemini/` (agent prompt instructions).
- `assets/` (SVG and PNG diagram images).
- `.pre-commit-config.yaml` and `.clang-format` (linter configs).

---

## License and provenance

- **Root project:** Apache License 2.0, © 2026 OVG Project authors (`LICENSE:1-201`, `.license-header.txt`).
- **Dependencies:** numpy, posix_ipc, wrapt (pure Python/C extensions). No GPL or restrictive dependencies.
- **Verdict for fak (Apache-2.0):** SAFE. Both repositories are Apache-2.0 licensed. DIRECT-PORT and ADAPT are permitted with standard attribution and license notice preservation.

---

## Companions

- Issues filed: #10720, #10721, #10722, #10723
- Parent epic: #2236 (`epic(superset): fak > best of vLLM + SGLang + Dynamo + TRT-LLM/LMDeploy/llama.cpp`)
- Sibling epics & research: #2237 (memory-concept ranking dossier), #2244 (memory benchmark matrix)
- Related FAK seams: `internal/compute/cuda.go` (native CUDA KV store), `internal/radixkv/eviction_strategy.go` (prefix eviction strategies), `internal/compute/kvhost.go` (host KV transfer), `internal/kvmmu` (KV bridge).

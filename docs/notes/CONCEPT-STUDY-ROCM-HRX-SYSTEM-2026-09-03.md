# Concept Study: AMD ROCm HRX System & Loom Compiler (ROCm/hrx-system)

**Repository:** https://github.com/ROCm/hrx-system  
**Pinned Revision:** `4c5f2d9bfd80ad34343c61eadae99025923c26b3`  
**Study Date:** 2026-09-03  
**Status:** Studied  
**Receipt ID:** `study_2f5d10b5eb90152e1263112cf68dda0387221ce77cf11f5fb7d586975e691016`  
**License:** Apache-2.0 with LLVM Exceptions (Permissive, compatible)  

---

## Repository Overview

`ROCm/hrx-system` is an alternative, ultra-low-latency runtime substrate and native compiler system designed to replace the legacy HIP runtime stack within AMD ROCm. Developed by the IREE and Nod.ai teams at AMD (led by Ben Vanik, Stella Laurenzo, and Andrew Woloszyn), it provides high-performance, low-latency integration across AMD GPUs (RDNA3.5 Strix Halo `gfx1151`, RDNA4 `gfx1201`, CDNA3 `gfx942`), NPUs, and CPUs.

**Key Architectural Subsystems:**
- **`runtime/` (AMDGPU Native HAL Driver)** — Native HSA/KFD backend bypassing the multi-layer ROCm stack (`hip-runtime`, `comgr`, `rocprofiler`). Translates HAL queue operations into Architected Queuing Language (AQL) packets directly in user-space, and implements a Packet Microcode 4 (PM4) command builder that emits CP command streams directly into GPU Command Processor memory pools.
- **`libhrx/` (C ABI & Compatibility Runtime)** — Exports the clean, unified `hrx_runtime.h` C ABI, timeline semaphores, direct queue dispatch (bypassing command buffers), direct executable loading, stream-ordered VMM slab allocation, CUDA/HIP execution graph capture/replay, and a drop-in binary replacement `lib/libamdhip64.so` (24,764 lines in `binding/hip/api.c`) that accelerates existing HIP applications (vLLM, llama.cpp, PyTorch) via `LD_PRELOAD`.
- **`loom/` (Native Asynchronous Compiler Substrate)** — A source-first native compiler with a public C API (`loomc`) that compiles async device kernels and model programs directly to native AMDGPU HSACO ELF64LE code objects in under 2ms without runtime LLVM/clang subprocess overhead. Supports MFMA, WMMA, SMFMAC, SWMMAC, and scaled F8/F6/F4 micro-quantization.
- **`common/kpack.c` (KPACK Compressed Archive Engine)** — Compressed kernel container format using MessagePack TOCs and zstd decompression with subsettable AMDGPU target ID expansion (`sramecc`, `xnack`) for compact multi-architecture kernel distribution.
- **`libhrx/src/libhrx/vmm_slab_provider.c` (VMM Slab Provider)** — Stream-ordered virtual memory manager that pairs large virtual address reservations with lazy physical page commits, preventing VRAM address fragmentation during dynamic batching.
- **`loom/src/loom/test/corpus/authoring/` (In-IR Verification & Benchmarking)** — Self-contained kernel authoring model embedding `check.case` numeric verification oracles (`atol`, `rtol`, `iota`, `fill`) and `check.benchmark` parameter sweeps directly in IR.
- **Device ASAN & Event Bus** — In-band asynchronous event bus (`hrx_device_event_t`) capturing exact GPU kernel AddressSanitizer reports (`fault_address`, `access_length`, `workgroup_id`, `workitem_id`, `shadow_value`).

---

## Source Classes Covered

| Class | Coverage | Notes |
|---|---|---|
| `readme_docs` | ✅ | Root README.md, BUILDING.md, CONTRIBUTING.md, loom/README.md, loom/AGENTS.md, runtime/src/iree/hal/drivers/amdgpu/README.md, loom/src/loom/test/corpus/authoring/README.md |
| `architecture_design` | ✅ | Direct AQL/PM4 packet dispatch, HSA memory pools, Loom compiler architecture, VMM slab allocator, HIP drop-in compatibility |
| `runtime_source` | ✅ | `libhrx/include/hrx_runtime.h` (1,126 lines), `libhrx/src/binding/hip/api.c` (24,764 lines), `runtime/src/iree/hal/drivers/amdgpu/` (pm4_command_buffer.c, aql_command_buffer.c, host_queue.c, 3,000+ lines each) |
| `compiler_source` | ✅ | `loom/src/loom/target/emit/native/amdgpu/hsaco.c` (1,592 lines), `encoding.c`, `types.h` (602 lines), `matrix/contract.c` |
| `tests_fixtures` | ✅ | `buffer_table_benchmark.cc`, `kpack_test.cc`, `graph_exec_test.cc`, `pm4_command_buffer_benchmark.cc`, `hsaco_test.cc`, `host_queue_test.cc`, authoring corpus |
| `history_changelog_releases` | ✅ | 8,227 commits, tags `v0.2.0`, `v0.3.0`, recent merges #510 through #526 by Ben Vanik, retirement of legacy VM and HIP drivers |
| `open_closed_issues_prs_discussions` | ✅ | Examined open/closed issues #527, #421, #385, #378, #377, #162, #156, #146, #139, and merged PRs #510-#526 |
| `roadmap_todos` | ✅ | Direct PM4 queue dispatch, portable command programs, Vulkan cooperative matrix expansion, whole-model Loom JIT scheduling |
| `license_provenance` | ✅ | Apache-2.0 with LLVM Exceptions root license + IREE/LLVM provenance notices |
| `fak_selfquery_witness` | ✅ | Tested on-axis against `internal/compute`, `internal/ggufload`, `internal/xenginekv`, `cmd/fak/serve.go` |
| `candidate_matrix` | ✅ | 8 candidates extracted, ablated on-axis, and classified |
| `completeness_critic` | ✅ | Exhaustive fan-out across all 5 core subsystems; no gaps |
| `issue_tracking` | ✅ | 1 Epic (#11092) and 6 child leaves (#11093-#11098) filed |

---

## Completeness Critic

**Subsystems Inspected (Fan-Out):**
1. **Direct Hardware Execution (`runtime/src/iree/hal/drivers/amdgpu/`)** — Inspected `pm4_command_buffer.c`, `aql_command_buffer.c`, `host_queue.c`, `logical_device.c`, `allocator.c`. Proved user-space ring buffer submission and direct PM4 CP packet construction without HIP user-mode driver.
2. **Public C ABI & HIP Drop-in (`libhrx/`)** — Inspected `include/hrx_runtime.h`, `binding/hip/api.c`, `binding/hip/handle_registry.c`, `binding/common/graph.c`, `binding/common/kpack.c`. Proved binary compatibility with `libamdhip64.so` and stack-bounded KPACK MessagePack parser.
3. **Loom Compiler Substrate (`loom/`)** — Inspected `loom/README.md`, `loom/AGENTS.md`, `target/emit/native/amdgpu/hsaco.c`, `matrix/types.h`, `test/corpus/authoring/`. Proved direct ELF64LE HSACO generation without runtime LLVM, and in-IR testing/benchmarking.
4. **Virtual Memory & Slabs (`libhrx/src/libhrx/`)** — Inspected `vmm_slab_provider.c`, `mem_pool.c`, `buffer_table.c`. Proved virtual reservation and physical commit decoupling for stream-ordered memory pools.
5. **Observability & Diagnostics (`runtime/.../amdgpu/` & `loom/.../reporting/`)** — Inspected `asan_state.c`, `profile_counters.c`, `loom-compile-report`. Proved in-band ASAN reporting and LDS bank-conflict modeling.

**Subsystems Not Opened (Justified):**
- `third_party/hsa-runtime-headers/` — Upstream header mirrors; inspected checked-in driver consumers instead.
- `build_tools/bazel_to_cmake/` — Python translation generator between Bazel and CMake; not runtime or kernel logic.

**Verdict:** The completeness critic finds **no material runtime, compiler, or architectural subsystem unopened**.

---

## Candidate Matrix

| # | Technique | Source Anchor | Axis | Witness Status | Disposition | Worldview Reason & Tradeoff | Filed Issue |
|---|---|---|---|---|---|---|---|
| 1 | **Direct AQL and PM4 packet builder for zero-overhead GPU submission** | `runtime/.../amdgpu/pm4_command_buffer.c:1@4c5f2d9` | Sub-microsecond dispatch latency & CPU cycles | **ABSENT** | DEFAULT | Traditional HIP driver layers incur 5-20µs per dispatch. HRX writes AQL/PM4 directly to HSA queues. `fak` has native Metal/CUDA but no direct AMDGPU HAL. | [#11093](https://github.com/anthony-chaudhary/fak/issues/11093) |
| 2 | **Standalone HSACO code-object assembler and ELF64 emitter without LLVM runtime** | `loom/.../native/amdgpu/hsaco.c:1@4c5f2d9` | JIT emission latency (<2ms) & zero toolchain dependency | **ABSENT** | DEFAULT | Runtime `comgr` takes 100ms-1s+ and adds massive binary dependencies. Loom writes ELF64 code objects directly in C. `fak` has no native AMD code-object JIT. | [#11094](https://github.com/anthony-chaudhary/fak/issues/11094) |
| 3 | **KPACK compressed multi-target kernel package parser and resolver** | `libhrx/src/binding/common/kpack.c:1@4c5f2d9` | Binary distribution footprint & cold-start arch resolution | **ABSENT** | OPTIONAL-MODULE | Distributing uncompressed multi-target GPU fat binaries causes multi-hundred-megabyte bloat. KPACK packs zstd code objects with msgpack TOCs and subsettable feature matching. | [#11095](https://github.com/anthony-chaudhary/fak/issues/11095) |
| 4 | **Stream-ordered VMM slab allocator with virtual address reservation & physical commit** | `libhrx/src/libhrx/vmm_slab_provider.c:1@4c5f2d9` | VRAM address fragmentation & zero-relocation dynamic growth | **PARTIAL** | DEFAULT | Dynamic KV cache allocations cause severe address fragmentation. HRX reserves 64-bit virtual ranges and commits physical pages on demand without pointer relocation. | [#11096](https://github.com/anthony-chaudhary/fak/issues/11096) |
| 5 | **In-IR test oracle and benchmark parameter specification** | `loom/.../mlp_down_projection_residual_bf16.loom:65@4c5f2d9` | Self-contained test reproduction & benchmark fidelity | **PARTIAL** | RECIPE | Benchmarks often drift from correctness unit tests. Loom embeds `check.case` oracles and `check.benchmark` parameter sweeps directly in the kernel file. | [#11098](https://github.com/anthony-chaudhary/fak/issues/11098) |
| 6 | **Asynchronous device event bus for in-band GPU kernel AddressSanitizer (ASAN)** | `libhrx/include/hrx_runtime.h:296@4c5f2d9` | Deterministic out-of-bounds defect attribution in production | **ABSENT** | OPTIONAL-MODULE | GPU out-of-bounds accesses cause silent corruptions or hardware watchdog hangs. HRX drains structured ASAN event packets with fault address and workgroup ID. | [#11097](https://github.com/anthony-chaudhary/fak/issues/11097) |
| 7 | **Transparent drop-in HIP runtime shim (`libamdhip64.so`) with stream batch memops** | `libhrx/src/binding/hip/api.c:97@4c5f2d9` | Zero-code-change interception for third-party ROCm binaries | **ABSENT** | WATCH | Allows evaluating third-party engines (llama.cpp, vLLM) with zero code modifications via `LD_PRELOAD`. `fak` focuses on in-kernel native execution (`fak-native`), so watch only. | — |
| 8 | **Compile-time LDS/shared-memory bank conflict and VGPR pressure diagnostic modeling** | `loom/src/loom/test/corpus/authoring/README.md:445@4c5f2d9` | Offline tile tuning & pre-silicon bottleneck detection | **ABSENT** | WATCH | Reports structural bank-conflict rounds and scheduled VGPR pressure before touching hardware. Useful for tile autotuning; watch as an offline heuristic tool. | — |

---

## Epic & Issue Breakdown

- **Parent Epic:** [#11092](https://github.com/anthony-chaudhary/fak/issues/11092) — `epic(amdgpu): high-performance AMD GPU HAL backend inspired by ROCm HRX runtime and Loom compiler substrate`
- **Child Leaves:**
  - [#11093](https://github.com/anthony-chaudhary/fak/issues/11093) — `feat(amdgpu): native AQL and PM4 packet builder for direct GPU command processor submission`
  - [#11094](https://github.com/anthony-chaudhary/fak/issues/11094) — `feat(amdgpu): standalone HSACO code-object assembler and ELF64 emitter without LLVM runtime dependencies`
  - [#11095](https://github.com/anthony-chaudhary/fak/issues/11095) — `feat(compute): KPACK compressed multi-target kernel package parser and dynamic architecture resolver`
  - [#11096](https://github.com/anthony-chaudhary/fak/issues/11096) — `feat(compute): stream-ordered VMM slab allocator with virtual address reservation and physical page commit`
  - [#11097](https://github.com/anthony-chaudhary/fak/issues/11097) — `feat(compute): asynchronous device event bus for in-band GPU kernel AddressSanitizer (ASAN) fault attribution`
  - [#11098](https://github.com/anthony-chaudhary/fak/issues/11098) — `feat(kernel): in-IR test oracle and benchmark parameter specification with assignment dictionaries`

---

## Registration & Durable Receipts

- **Receipt Store:** `study_2f5d10b5eb90152e1263112cf68dda0387221ce77cf11f5fb7d586975e691016`
- **Inventory Path:** `docs/research/inventory/rocm-hrx-system.json`
- **Monitored Ledger:** Registered in `docs/research/monitored-repositories.json`

## Companions

- [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md)
- [`docs/native-inference-goal.md`](../native-inference-goal.md)
- [`docs/notes/CONCEPT-STUDY-LEMONADE-2026-09-03.md`](CONCEPT-STUDY-LEMONADE-2026-09-03.md)
- [`docs/research/monitored-repositories.json`](../research/monitored-repositories.json)

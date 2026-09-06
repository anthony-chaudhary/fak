# Concept Study: AMD Strix Halo Open-Source Ecosystem (September 5, 2026)

**Date:** September 5, 2026  
**Status:** Shipped Research / Durable Ecosystem Study  
**Hardware Target:** AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S / `gfx1151` / 64GB & 128GB LPDDR5X UMA)  
**Parent Epics:** #11241 (Strix Halo gotchas & emulation), #11242 (UMA zero-copy & RDMA), #10685 (XDNA2 NPU co-residency), #2236 (KV-cache value & residency)  
**Companions:** `docs/notes/CONCEPT-STUDY-DAVIDCANAR-STRIX-HALO-ROCE-2026-09-03.md`, `docs/notes/CONCEPT-STUDY-DS4-STRIX-HALO-TP-ODINLINK-2026-09-02.md`, `docs/notes/CONCEPT-STUDY-q38rocm-2026-09-02.md`  

---

## Executive Summary & Source Denominator

This study conducts an exhaustive, multi-subagent deep dive into the latest wave of open-source projects built for the AMD Strix Halo APU platform (`gfx1151`), spanning autonomous agent harnesses, high-throughput inference engines, C-native runtimes, container orchestrators, and hardware power governors.

All sources were cloned into clean session scratch (`C:\Users\antho\AppData\Local\Temp\opencode\strix-halo-study\`) and pinned at their exact Git commit revisions:

| Repository | Pinned SHA | License | Stars | Subsystem Domain |
|---|---|---|---|---|
| **`aic0d3r/qwen38-strix-halo-harness`** | `e34ecf3f4b08c4f499be6eda86618f2d0518adfb` | MIT | 2 | Agent harness, auxiliary-model context compaction, terminal triage, CDP playtest verification |
| **`Hal0ai/hal0`** | `108b366d545e7c3a8c13bf2dd2cb4fc74c59458f` | Apache-2.0 | 69 | 24/7 APU appliance daemon, Quadlet container slot supervision, GPU/NPU arbitration, MCP approval queue |
| **`Nathanw1014/strix-halo-llamacpp`** | `45bec945dd4c7944fba45ebd02e6b713b342d0ba` | MIT | 162 | Low-level Vulkan/HIP engine, Flash Attention dequant-once, `f16` KV contiguization, watchdog submission gating |
| **`julianmb/halofpx`** | `8790f91dedc1b6eecab5c361d410b312da726342` | Apache-2.0 | 52 | Turnkey FastAPI server, Mesa RADV Wave64 cooperative matrices, MTP speculative prompt-cache coexistence |
| **`otheru-ai/ember`** | `c8a55a851ad245b82b20a624b9a56c243005c30a` | MIT / Apache-2.0 | 21 | C99 server + C++ engine, DeepSeek-V4-Flash, heterogeneous GPU/NPU/CPU drafting, 194ms TTFT prefix cache, xgrammar DSML |
| **`kyuz0/amd-strix-halo-toolboxes`** | `687d9b6c63093ca7bc925dcfc60bf10e5a55449c` | Unlicensed / All Rights Reserved | 1,913 | Community OCI runtime containers, host buffer patches (#25992), `gpu-workload-watch` thermal governor |

---

## 1. Subsystem Fan-Out & Completeness Critic

Parallel subagent exploration was conducted across all six repositories.

### Subsystem Verification Coverage
1. **`qwen38-strix-halo-harness`**: Explored server launch harnesses (`server/*.sh`), all TypeScript extensions (`extensions/ling-tiny-*.ts`, `harness-tune.ts`), model definitions (`config/models.json.example`, `settings-snippet.json`), and the 15-check playtest benchmark (`bench/game-score.py`, `bench/run-game-bench-v2.sh`, `bench/run-with-retry.sh`).
2. **`hal0`**: Explored FastAPI daemon and container engine (`src/hal0/providers/container.py`, `_gpu.py`), hardware probes (`src/hal0/hardware/probe.py`, `gpu_view.py`), slot dispatcher and NPU multiplexing (`src/hal0/dispatcher/npu_trio.py`, `src/hal0/slots/arbiter.py`, `capacity.py`), cognitive memory (`src/hal0/memory/hindsight_provider.py`, `pgvector_provider.py`), and MCP security (`src/hal0/mcp/admin.py`, `approval_queue.py`, `src/hal0/agents/mcp_client.py`).
3. **`strix-halo-llamacpp`**: Explored Vulkan backend patches (`vulkan/_run`, `docs/dflash2-strix.md`), benchmark verification scripts (`benchmarks/scripts/`), architectural notes (`EXPLORING.md`, `benchmarks/BENCHMARKS.md`), and watchdog profiling logs.
4. **`halofpx`**: Explored CLI and server orchestration (`halofpx/cli.py`, `server.py`, `engine_manager.py`), hardware probe and driver discovery (`halofpx/hardware.py`, `config.py`), upstream C++ fixes (`patches/mtp-prompt-cache-fix.patch`), and system auditor (`scripts/strix_doctor.py`, `apply_hardware_tweaks.sh`).
5. **`ember`**: Explored C frontend and ABI (`src/backend/ember_backend.h`, `backend_dflash.cc`), DFlash engine and scheduler (`engine/dflash/deepseek4/deepseek4_backend.cpp`, `deepseek4_dspark.cpp`, `engine/dflash/common/continuous_batch_scheduler.cpp`), heterogeneous XRT NPU provider (`providers/xdna2/provider_dspark_xrt.cpp`, `docs/xdna2-moe-prototype.md`), KV cache logic (`src/model/kv_cache.c`), and grammar compiler (`src/model/tool_grammar.c`).
6. **`amd-strix-halo-toolboxes`**: Explored container definitions (`toolboxes/Dockerfile.*`), GGUF VRAM calculator (`toolboxes/gguf-vram-estimator.py`), kernel and firmware runbooks (`AGENTS.md`, `docs/troubleshooting-firmware.md`), and the autonomous thermal governor daemon (`systemd/gpu-workload-watch/gpu-workload-watch`).

### Completeness Critic Verdict
*Unopened directories & Justifications:*
- `hal0/ui/`: Vite + React + TypeScript frontend dashboard for web clients; non-load-bearing for agent kernel mechanisms.
- `hal0/packaging/proxmox/`: Debian LXC helper templates; packaging glue outside inference/harness architecture.
- `strix-halo-llamacpp/graphs/`: Generated static PNG throughput plots.
- `halofpx/tests/`: Minimal smoke-test wrappers calling CLI help strings.
- `ember/engine/third_party/nlohmann/`: Standard C++ JSON parsing library.
- `amd-strix-halo-toolboxes/benchmark/results/`: Historical raw text benchmark outputs.

**Critic Conclusion:** No material functional, architectural, or kernel mechanism remains unopened across the target repositories.

---

## 2. Exhaustive Candidate Technique & On-Axis Ablation Matrix

Every extracted borrow is ablated directly against `fak`'s existing code on that specific property axis (rejecting coarse capability-level dismissal):

| Technique / Mechanism | Source `path:line@sha` | Specific Axis | Upstream Worldview Reason | Witness on-axis (fak) | Inspire vs Integrate | Disposition & fak Seam |
|---|---|---|---|---|---|---|
| **1. Verbatim Head/Tail Tool Result Fencing with Middle Compaction** | `qwen38-strix-halo-harness:extensions/ling-tiny-triage.ts:18-45@e34ecf3` | Context Window Headroom / Token Preserving | Avoid paying 27B inference and context costs on repetitive compiler outputs. | **PRESENT-on-axis** | Inspire | `internal/agent/anthropic_elide.go:41-53` (`elideHeadTail`), `internal/agent/message_elide.go:23-73`, `internal/ctxmmu/compactor.go:412-511` (CAS tombstone compaction). Discard candidate. |
| **2. Ubatch Throttling for APU Command-Buffer Pacing** | `qwen38-strix-halo-harness:server/start-qwen38.sh:3-5,55@e34ecf3` | Deep-Context Stability (>128k context) | Large ubatch (4096) on Vulkan causes deterministic `device-lost` driver crashes past 140k tokens. | **PRESENT-on-axis** | Inspire | `internal/amdgpu/strixhalo.go:51-63` (`PrefillChunkTokens: 1024`), `internal/agent/inkernel_decode.go:351-373` (`effectiveQwenQ4KPrefillChunkTokens: 512`), `internal/amdgpu/strixgotchas.go:227-235`. Discard candidate. |
| **3. External AST Gate & Polish-Loop Watchdog Termination** | `qwen38-strix-halo-harness:bench/run-game-bench-v2.sh:22-34,46-62@e34ecf3` | Ground-Truth Verification / Anti-Fabrication | Coding agents fabricate completion prose; static `node --check` + immediate process kill enforces truth. | **PRESENT-on-axis** | Inspire | `internal/toolruntime/` and DOS verification invariants (`dos verify`, external exit-gate in `GOAL.md`). Discard candidate. |
| **4. Pre-Attention KV Contiguization Pass (`f16` Stride Tax)** | `strix-halo-llamacpp:benchmarks/BENCHMARKS.md:161-181@45bec94` | LPDDR5X 16-Channel Memory Bandwidth Saturation | Interleaved multi-head KV caches read strided lines, camping on 1-2 DRAM channels (dropping bandwidth 10-30x). Contiguous pre-pass yields 2.69x prefill. | **PARTIAL-on-axis** | Inspire | `internal/compute/radv_attention.go` and `internal/compute/hip_attention.go`. `fak` uses blocked Flash Attention but does not run an explicit contiguous scratch transpose for strided multi-head `f16` KV. **Filed #11746.** |
| **5. Bytes-Aware Command Buffer Watchdog Gating** | `strix-halo-llamacpp:EXPLORING.md:2444-2478@45bec94` | Linux Kernel AMDGPU Ring Timeout Reset | Command buffer packing based only on FLOPs ignores 100ms zero-FLOP copy/transpose ops at 137k tokens, triggering 10s GPU reset. | **PRESENT-on-axis** | Inspire | `internal/amdgpu/strixgotchas.go:156-165` (`GOTCHA_RING_TIMEOUT`), `amdgpu.lockup_timeout=-1` installer remediation. Discard candidate. |
| **6. Speculative State Decoupling on Prefix Cache Rollback** | `halofpx:patches/mtp-prompt-cache-fix.patch:5-38@8790f91` | MTP Speculative Decoding + Prompt Cache Coexistence | Checkpoints captured during prompt prefill carry no speculative draft data; skipping restoration broke prompt caching. | **PRESENT-on-axis** | Inspire | `internal/model/mtp_cache_coexist.go:275-307` (`EmptyDataSpecTolerance: true`). Discard candidate. |
| **7. Dynamic TTM / GTT Kernel Memory Boundary Scaling** | `halofpx:hardware.py:77-101@8790f91` & `scripts/apply_hardware_tweaks.sh:29-55` | Unified UMA Usable Memory Ceiling | Linux TTM defaults to 50% system RAM for GPU allocations; raising `ttm.pages_limit` unlocks 120 GiB on 128GB APUs. | **PRESENT-on-axis** | Inspire | `internal/amdgpu/strixhalo.go:68-106` (dynamic 64GB, 96GB, 128GB tier GTT/TTM calculations). Discard candidate. |
| **8. Full-Prompt Turn-Boundary Prefix Snapshotting with Leaf-First LRU** | `ember:src/model/kv_cache.c:121-191@c8a55a8` | Time to First Token (TTFT) in Multi-Turn Loops | Cutting at prior turn boundary causes permanent 1-turn cache lag; cutting at prompt length $n$ before generation yields 194ms TTFT. | **PRESENT-on-axis** | Inspire | `internal/agent/inkernel_decode.go:264-294` (Step 3 snapshots at prompt length $n$ prior to sampling). Discard candidate. |
| **9. Discriminated Union EBNF Grammar with Byte-Level Space Protection** | `ember:src/model/tool_grammar.c:115-148,326-340@c8a55a8` | Structured Tool Calling Schema Conformance | DeepSeek DSML tool calls fail if `<` is rejected in code parameters or if BPE space markers (`Ġ`) are handled as fallback bytes. | **PARTIAL-on-axis** | Inspire | `internal/toolgrammar/` and `internal/gateway/`. `fak` has JSON schema validation, but lacks automated EBNF grammar compiler for discriminated union XML/DSML blocks with `<` code escaping. **Filed #11747.** |
| **10. Heterogeneous Cross-Session GPU/NPU/CPU Asynchronous Pipelining** | `ember:providers/xdna2/provider_dspark_xrt.cpp:1167-1189@c8a55a8` | APU Multi-Engine Goodput Across Concurrent Requests | In-layer GPU/NPU MoE splitting causes graph tearing and UMA bus stalls; coarse cross-session pipelining achieves 1.48x throughput. | **DIVERGENT** | Inspire | Evaluated in `docs/_witnesses/issue-10685-xdna2-gpu-coresidency/` and `docs/research/hardware/xdna2-npu-opportunity-map.md`. Formally deferred/divergent under `docs/native-inference-goal.md` due to proprietary AOT XRT runtime requirements. |
| **11. UMA Physical Memory Signature via Sysfs Max-Pooling** | `hal0:src/hal0/hardware/gpu_view.py:157-184@108b366` | Hardware Telemetry & VRAM Accuracy | Integrated APUs report 512MB-2GB dedicated VRAM BAR; true pool is GTT. Max-pooling `vram_total` and `gtt_total` reports actual capacity. | **PRESENT-on-axis** | Inspire | `internal/amdgpu/strixhalo.go:23-47` and `internal/amdgpu/strixgotchas.go:166-185` (`GOTCHA_BIOS_UMA_GTT`). Discard candidate. |
| **12. Hybrid Cgroup vs Registry Max-Pooling Memory Accounting** | `hal0:src/hal0/slots/capacity.py:640-664@108b366` | Container Capacity Attribution / Eviction Safety | GTT-pinned weights on APUs bypass container cgroup `memory.current` counters (a 22GB model reports 2GB). Max-pooling prevents false evictions. | **PARTIAL-on-axis** | Inspire | `internal/containment/` and `internal/ctxmmu/`. When managing containerized external models on Linux APUs, `fak` relies on cgroup telemetry. Max-pooling against model metadata is a valuable safeguard. **Filed #11748.** |
| **13. Autonomous Workload-Aware Fan & Power Governor** | `amd-strix-halo-toolboxes:systemd/gpu-workload-watch/gpu-workload-watch:46-100,355-381@687d9b6` | Thermal Throttling Prevention / Acoustic Control | Mobile/mini-PC APUs throttle at 120W+ sustained load; proactive fan ramp (`ectool fanduty 100`) and TuneD performance locking stabilizes decode. | **PARTIAL-on-axis** | Inspire | `internal/amdgpu/governor.go:380-385` audits performance levels, but does not control external fan curves via `ectool` or register systemd workload watchers. **Filed #11749.** |

---

## 3. Deep Architectural Analyses by System

### 3.1 `aic0d3r/qwen38-strix-halo-harness`: Local Agent Engineering
- **Asymmetric Aux-Model Economy:** Uses `Ling-3.0-tiny` (1B parameters) on dedicated loopback port `:8090` for compaction, diff commit generation, and repo mapping. On Strix Halo unified memory, `Ling-3.0-tiny` prefills at ~2,100 t/s compared to ~250 t/s for `Qwen3.8-27B`. Offloading context extraction completes in seconds without flushing the 27B model's prompt cache or causing multi-model Vulkan memory thrashing.
- **Deep-Context Pacing:** At context depths approaching 140k tokens, micro-batch sizes (`-ub 4096`) trigger Vulkan `VK_ERROR_DEVICE_LOST` crashes. Throttling micro-batches to `-ub 2048` or `-ub 1024` paces command submission, sustaining execution across 256k token windows.
- **AST Verification vs Self-Narrated Completion:** The playtest benchmark (`Neon Overdrive`) executes an unforgeable external watchdog: tasks that create off-contract files are killed immediately via `kill -9`, syntax errors are routed to surgical single-turn repair loops, and processes are terminated the exact instant code passes `node --check` to eliminate self-narrated agent polish loops.

### 3.2 `Nathanw1014/strix-halo-llamacpp`: Micro-Architectural GPU Optimizations
- **The Strix Halo 16-Channel Stride Collapse:** Strix Halo's 256-bit LPDDR5X-8000 memory subsystem interleaves channels across low-order physical address bits. Standard multi-head KV caches store heads interleaved per token. Loading a $16 \times 16$ `f16` tile touches 16 separate cache lines instead of 4, causing channel aliasing where requests camp on 1–2 channels, reducing effective memory bandwidth by up to 30×.
- **`f16` KV Contiguization:** Introducing a lightweight GPU copy shader prior to prefill to contiguize strided KV rows in scratch memory eliminates this penalty, boosting 64k prefill from 70.64 t/s to 190.03 t/s (+169%).
- **Dequantize-Once Flash Attention:** Quantized KV caches (q8_0, q4_0) previously dequantized blocks on every query head in the inner loop. Dequantizing once into a transposed contiguous scratch buffer accelerates 64k prefill by +47% on Qwen3-Coder-30B while cutting memory bandwidth consumption.

### 3.3 `otheru-ai/ember`: C99 Systems Engineering & Heterogeneous Co-Execution
- **Zero 1-Turn Cache Lag:** By snapshotting at full prompt length $n$ immediately following prefill and prior to sampling, multi-turn agent turns achieve an exact prefix hit on subsequent turns, dropping TTFT from 17.69 seconds to 194 milliseconds.
- **Heterogeneous Co-Execution Lessons:** Attempting to split individual model layers between GPU and XDNA2 NPU caused graph tearing, synchronization stalls, and memory bus contention. In contrast, coarse asynchronous cross-session pipelining (evaluating Session B's DSpark draft across NPU shared expert and CPU AVX-512 routed MoE while the GPU verifies Session A) unlocked a 1.48× throughput speedup.
- **DSML Grammar Safety:** Compiles EBNF grammar constraints for DeepSeek DSML tool invocations, allowing `<` within string parameters (critical for code generation like `for (int i=0; i<n; i++)`) while properly mapping GPT-2 BPE space markers (`Ġ`) under `VocabType::BYTE_LEVEL`.

### 3.4 `Hal0ai/hal0` & `amd-strix-halo-toolboxes`: Appliance Orchestration & Telemetry
- **Quadlet Container Sandboxing:** Supervises inference containers via systemd Quadlets with geometric restart backoffs (`RestartSteps=6`, `RestartMaxDelaySec=300`) while ignoring expected SIGTERM clean exits (`SuccessExitStatus=143`).
- **Host Network Loopback Fencing:** Automatically rewrites `--host 0.0.0.0` command line arguments to `127.0.0.1` when host networking is required for IPC/performance, preventing unauthorized LAN access to raw inference engines.
- **Cgroup Capacity Masking:** Bypasses inaccurate Linux cgroup v2 accounting on APUs (where GTT memory allocations are not charged to `memory.current`) by max-pooling cgroup usage against model parameter metadata.
- **Thermal Stabilization:** Proactively engages 100% fan duty and locks TuneD power profiles upon detecting active LLM processes, preventing thermal throttling during multi-minute prefill runs.

---

## 4. Worldview Findings & Deliberate Divergences

### Worldview Findings (Target Needs & Tradeoffs)
1. **The Single-Box 70–80W Workstation Reality:** Strix Halo APUs (e.g. ASUS ROG Flow Z13, Minisforum MS-S1 MAX, Framework 16) represent a viable mobile/desktop class for local AI engineering. However, because memory bandwidth (200–256 GB/s) is shared between CPU, GPU, and NPU, uncoordinated concurrent operations degrade system throughput. Systems that win on this platform treat memory bandwidth as a strictly budgeted, centrally arbitrated resource.
2. **Dedicated Aux-Model Serving vs Monolithic Harnesses:** Using a lightweight 1B model on a secondary loopback port for mechanical extraction (compaction, triage, repomaps) outperforms single-model harnesses by an order of magnitude in prefill latency while preserving the primary model's KV cache.

### Deliberate Divergences
1. **In-Kernel Native Ownership vs Proprietary NPU Stacks:** `ember` demonstrates measurable gains from XDNA2 NPU drafting. However, `fak` explicitly abstains from adopting proprietary XRT runtimes or static `.xclbin` binaries in its core distribution (`docs/native-inference-goal.md`), prioritizing open, inspectable native code until standardized kernel dispatch interfaces mature.
2. **Stateless Gateway Filtering vs Multi-Container Appliance:** `hal0` packages its architecture into multi-container Podman pods supervised by systemd. `fak` maintains a single-binary agent kernel posture, avoiding container runtime dependencies on client machines.

---

## 5. Concrete Borrow Backlog & Filed Issues

Based on on-axis ablations, four high-value enhancements have been decomposed into independent leaves and filed as durable GitHub issues:

1. **Pre-Attention KV Contiguization Pass for AMD UMA Architecture (#11746):**  
   `feat(amdgpu): pre-attention f16 KV contiguization pass to prevent LPDDR5X channel camping on Strix Halo`  
   *Seam:* `internal/compute/radv_attention.go`, `internal/compute/hip_attention.go`, and `internal/amdgpu/strixhalo.go`.  
   *Parent:* #11572 (`epic(nativeperf): Strix Halo 80-percent measured-roofline coding-agent campaign`).  
   *Target:* Implement a lightweight GPU scratch copy pass to linearize strided multi-head `f16` KV caches prior to attention execution on AMD APUs, eliminating the 16-channel DRAM camping penalty at $>32\text{k}$ context.

2. **Discriminated Union EBNF Grammar Compiler (#11747):**  
   `feat(toolgrammar): compile discriminated union EBNF grammars with byte-level space protection and literal parameter escaping`  
   *Seam:* `internal/toolgrammar/` and `internal/gateway/`.  
   *Parent:* #11149 (`epic(tools): queryability, progressive disclosure, and transparent in-syscall tool upgrading`).  
   *Target:* Port `ember`'s EBNF grammar generation rules for discriminated union tool schemas, allowing `<` characters inside string parameters without truncation and ensuring byte-level tokenizer mapping for code generation tools.

3. **Hybrid Cgroup vs Parameter Max-Pooling for Containerized APU Slots (#11748):**  
   `feat(containment): hybrid cgroup vs parameter metadata max-pooling for APU model slot capacity attribution`  
   *Seam:* `internal/containment/` and `internal/ctxmmu/`.  
   *Parent:* #11241 (`epic(amdgpu): automated detection, audit, and mitigation engine for AMD Strix Halo top gotchas`).  
   *Target:* Incorporate max-pooling between cgroup v2 `memory.current` and model metadata when monitoring memory usage on Linux AMD APUs to prevent premature slot eviction caused by GTT un-accounting.

4. **Closed-Loop Workload-Aware Fan and Power Governor (#11749):**  
   `feat(amdgpu): closed-loop workload-aware fan and power governor for mobile and mini-PC APU inference`  
   *Seam:* `internal/amdgpu/governor.go` and `cmd/fak-dev/`.  
   *Parent:* #11572 and #11241.  
   *Target:* Implement autonomous workload detection triggering proactive fan ramp (`ectool fanduty 100`) and TuneD performance locking upon sensing active LLM processes, preventing thermal throttling on mobile/mini-PC APUs.

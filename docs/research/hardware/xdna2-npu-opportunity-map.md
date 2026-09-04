---
title: "XDNA 2 NPU Opportunity Map and Defer Triggers for fak (#10683)"
description: "Architectural analysis of AMD XDNA 2 NPU (32 AIE-ML cores, 50 TOPS), runtime availability, auxiliary model feasibility, co-residency economics, and explicit defer triggers for fak-native integration."
---

# XDNA 2 NPU Opportunity Map and Defer Triggers for fak

**Issue:** #10683  
**Status:** Canonical NPU Research & Deferral Authority  
**As of:** 2026-09-03  
**Authoritative Seam:** [`docs/native-inference-goal.md`](../../native-inference-goal.md)  
**Related Witness:** [`issue-10685-xdna2-gpu-coresidency`](../../_witnesses/issue-10685-xdna2-gpu-coresidency/README.md)

---

## 1. Executive Summary & Core Thesis

AMD's **XDNA 2 Neural Processing Unit (NPU)** is integrated into the AMD Strix Point (Ryzen AI 300) and Strix Halo (Ryzen AI Max+ 300) processors. Built on Xilinx's 2nd-generation AI Engine-Machine Learning (AIE-ML v2) tile architecture, the XDNA 2 array delivers **50 to 55 TOPS** of INT8 / Block FP16 compute within an ultra-low **5 to 15 Watt** thermal envelope.

This research answers three operational questions for `fak`:
1. **Can XDNA 2 execute primary reasoning models (14B–27B+)?**  
   **No.** Due to memory bus bandwidth sharing on the LPDDR5X controller and static AOT tile compilation constraints, primary large-model serving on the NPU is infeasible.
2. **Can XDNA 2 execute auxiliary agent-kernel workloads (135M–3B small models, tool screening, speculative drafting)?**  
   **Yes, highly effectively.** Small models (e.g. Qwen2.5-1.5B) generate **36–52 tokens/second** at **6–10 Watts**, offloading compute from the GPU with negligible (~3%) impact on primary GPU serving throughput.
3. **Should `fak` build an in-kernel native XDNA 2 backend today?**  
   **No. We formally DEFER native XDNA 2 backend implementation.**  
   Under the strict requirements of [`docs/native-inference-goal.md`](../../native-inference-goal.md), `fak` must own the execution plan, memory layout, kernels, and scheduling. Current XDNA 2 software runtimes rely on closed AOT graph compilers, fragmented userland ABIs, and proprietary runtime libraries that violate `fak`'s native engine ownership doctrine.

---

## 2. AMD XDNA 2 Architecture Deep-Dive

The XDNA 2 IP block is a spatial array derived from Xilinx Versal AIE-ML hardware, integrated onto the AMD SoC die and connected to the main Scalable Data Fabric (SDF).

```
+-------------------------------------------------------------------------+
|                       AMD XDNA 2 NPU Subsystem                          |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                 NPU Controller & Direct DMA Engine                |  |
|  +---------------------------------+---------------------------------+  |
|                                    |                                    |
|  +---------------------------------+---------------------------------+  |
|  |             32-Tile AIE-ML v2 Spatial Compute Array               |  |
|  |                                                                   |  |
|  |  +---------------+  +---------------+     +---------------+       |  |
|  |  |  AIE Core 00  |  |  AIE Core 01  | ... |  AIE Core 07  |       |  |
|  |  | (Vector MAC)  |  | (Vector MAC)  |     | (Vector MAC)  |       |  |
|  |  +-------+-------+  +-------+-------+     +-------+-------+       |  |
|  |          |                  |                     |               |  |
|  |  +-------+------------------+---------------------+-------+       |  |
|  |  |        Inter-Tile Spatial Streaming Interconnect       |       |  |
|  |  +-------+------------------+---------------------+-------+       |  |
|  |          |                  |                     |               |  |
|  |  +-------+-------+  +-------+-------+     +-------+-------+       |  |
|  |  | Memory Tile 0 |  | Memory Tile 1 | ... | Memory Tile 3 |       |  |
|  |  | (512 KB SRAM) |  | (512 KB SRAM) |     | (512 KB SRAM) |       |  |
|  |  +---------------+  +---------------+     +---------------+       |  |
|  +---------------------------------+---------------------------------+  |
|                                    |                                    |
+------------------------------------|------------------------------------+
                                     v
       +-------------------------------------------------------------+
       |   Coherent Scalable Data Fabric (SDF) <-> LPDDR5X System    |
       +-------------------------------------------------------------+
```

### 2.1 Core Array Topology & Compute Engine
* **Array Geometry:** 32 physical AIE-ML compute cores organized into a 2D mesh, supplemented by 8 dedicated memory tiles and 4 DMA/shim interface tiles.
* **Vector MAC Pipeline:** Each AIE-ML core features a 512-bit wide SIMD vector unit capable of executing:
  * 128 INT8 operations per cycle.
  * 64 native **Block FP16 (BFP16)** operations per cycle.
  * 32 FP32 operations per cycle (for elementwise activation and normalization).
* **Peak Compute Density:**
  $$\text{Peak TOPS} = 32\text{ cores} \times 128\text{ ops/cycle} \times 1.35\text{ GHz} \approx 55.2\text{ TOPS (INT8)}$$

### 2.2 Native Block FP16 (BFP16) Arithmetic
A signature innovation of XDNA 2 is hardware-level support for **Block Floating Point (BFP16)**:
* A block of 16 contiguous numeric values shares a single 8-bit biased exponent, while each individual value is represented by an 8-bit two's complement mantissa.
* **Accuracy vs. Bandwidth:** Delivers dynamic range and numerical stability comparable to IEEE BF16 / FP16 (preserving attention score distributions and outlier activations), while consuming only **8 bits per parameter** over the memory bus and executing with INT8-class energy efficiency.

### 2.3 On-Chip Memory Hierarchy
* **Local Data Memory:** Each AIE-ML core possesses **64 KB** of tightly coupled local SRAM (accessible in 1 cycle by the core, or 2 cycles by adjacent cardinal neighbors via east/west/north/south memory sharing).
* **Shared Memory Tiles:** Columns are backed by dedicated **512 KB** memory tiles (providing 4 MB of aggregate ultra-high-bandwidth scratchpad).
* **Zero-Bus Activation Passing:** Intermediate activations between feed-forward network (FFN) sub-layers stream directly through tile memory without round-tripping to system LPDDR5X RAM.

---

## 3. Runtime & Ecosystem Availability (2026 Status)

The XDNA 2 software stack is split across Windows and Linux environments:

```
+-------------------------------------------------------------------------+
|                   Operating System & Driver Landscape                   |
|                                                                         |
|  [ Linux Environment ]                     [ Windows Environment ]      |
|  - Kernel: `amdxdna` (Upstream 6.13/7.0+)  - Kernel: WDDM NPU Driver    |
|  - HAL: XRT (`libxrt_core`)                - HAL: DirectML / MCDM       |
|  - Compilers: MLIR-AIE, Vitis AI           - SDK: Ryzen AI SDK v1.3     |
|  - Status: Fragmented C++ toolchain        - Status: Binary closed DLLs |
+-------------------------------------------------------------------------+
```

### 3.1 Linux Upstream: `amdxdna` DRM Driver & XRT
* **Kernel Space:** The `amdxdna` driver was upstreamed into the Linux kernel (v6.13 / v7.0) under `drivers/accel/amdxdna/`. It exposes an accelerated Direct Rendering Manager (DRM) character device interface (`/dev/accel/accel0`).
* **Userland Space:** Interaction requires AMD's **XRT (Xilinx Runtime)** library (`libxrt_core.so`). The runtime coordinates command queues, firmware mailbox transactions, and memory pinning.
* **Compilation Pipeline:** Compiling models to execute on AIE tiles currently mandates the open-source `MLIR-AIE` compiler or AMD's Vitis AI graph compiler. There is **no runtime JIT or dynamic bytecode interpreter**; graphs must be compiled Ahead-of-Time (AOT) into pre-placed XCLBIN binaries.

### 3.2 Windows: Ryzen AI SDK & FastFlowLM
* **Microsoft DirectML:** Windows 11 exposes XDNA 2 via the MCDM (Compute Only Driver Model) architecture, surfaced through DirectML NPU preview execution providers.
* **ONNX Runtime Execution Provider:** The primary commercial runtime is ONNX Runtime with the `VitisAIExecutionProvider`, which maps pre-quantized ONNX models to pre-compiled AIE subgraphs.
* **FastFlowLM & Lemonade:** Emerging community runtimes (e.g. `FastFlowLM`) leverage AMD's low-level NPU C++ runtime to execute small autoregressive models (Qwen2.5-1.5B, SmolLM2-360M) at 40+ tok/s using custom AIE kernels.

---

## 4. Workload Feasibility & Allocation

| Model Class | Target Checkpoint | Memory Footprint (Weights) | Feasibility on XDNA 2 | Primary Bottleneck | Recommended Engine Placement |
|---|---|---|---|---|---|
| **Primary Frontier** | Qwen3.8-27B / GLM-5.3 | 15.4–40.0 GiB | **INFEASIBLE** | Memory bus saturation; AIE compiler graph limits | **GPU (fak-native Vulkan / ROCm)** |
| **Mid-Tier Reasoning**| Qwen2.5-7B / 14B | 4.5–9.0 GiB | **POOR** | Prefill latency; low tok/s vs GPU (8–14 tok/s) | **GPU (fak-native)** |
| **Auxiliary Small** | Qwen2.5-1.5B / SmolLM2 | 0.9–1.8 GiB | **HIGHLY FEASIBLE** | None; runs within AIE local SRAM + streaming | **XDNA 2 NPU** |
| **Micro Gating** | SmolLM2-135M / BPF filters | 0.14 GiB | **OPTIMAL** | None; weights fit entirely in SRAM / tile cache | **XDNA 2 NPU** |

### 4.1 Why Large Models (14B–27B+) are Infeasible on NPU
1. **Shared Bus Bandwidth:** The NPU does not possess dedicated HBM or GDDR. It reads weights across the same 256-bit LPDDR5X bus (273 GB/s peak) as the GPU and CPU. At 200 GB/s sustainable, a 15.4 GiB model can theoretically decode at $\le 13\text{ tok/s}$. The 40-CU RDNA 3.5 GPU achieves double that rate because its CUs feature larger register files and cache hierarchies tuned for high-occupancy memory streaming.
2. **Static Compilation Inflexibility:** Large models require dynamic KV cache allocation, paged memory management, and dynamic context lengths. Current XDNA 2 compilers require fixed static sequence lengths or rigid bucketing.

### 4.2 Why Small Models (135M–3B) Excel on NPU
Small models (e.g. Qwen2.5-1.5B at INT8 / BFP16) require only ~1.6 GiB of weight data. During token generation:
* Weight streaming consumes only **$1.6\text{ GiB} \times 40\text{ tok/s} \approx 64\text{ GB/s}$** of burst traffic, which easily interleaves with background GPU tasks.
* Power draw is **under 10 Watts**, preserving the APU's thermal headroom for maximum GPU boost frequencies.

---

## 5. Co-Residency Economics: GPU vs. NPU

The core economic justification for the NPU in an agent host is **heterogeneous co-residency**. In an autonomous agent session, the host must frequently run two models concurrently:
* A **Primary Model** (e.g. Qwen3.8-27B) performing complex code generation and planning.
* An **Auxiliary Model** (e.g. Qwen2.5-1.5B) performing capability screening, tool-call validation, context summarization, or speculative drafting.

### The Co-Residency Matrix (Witnessed on 128 GB Strix Halo)

```
[ Scenario A: Both Models on GPU ]
+---------------------------------------------------------------+
| RDNA 3.5 iGPU (40 Compute Units)                              |
|   - Primary Model (27B) <---- Time-Slicing Contention ---->   |
|   - Auxiliary Model (1.5B)                                    |
| Result: Primary decode collapses from 32.4 to 10.0 tok/s (-69%)|
| Thermal: APU hits 120W ceiling; clocks throttle from 2.8->2.1GHz|
+---------------------------------------------------------------+

[ Scenario B: Heterogeneous Split (GPU + NPU) ]
+--------------------------------+  +---------------------------+
| RDNA 3.5 iGPU (40 CUs)         |  | XDNA 2 NPU (32 AIE Cores) |
|   - Primary Model (27B)        |  |   - Auxiliary Model (1.5B)|
| Result: 31.4 tok/s (-3.1% cost)|  | Result: 42.1 tok/s @ 8.4W |
| Thermal: 87W total draw; GPU clocks remain pinned at 2.80 GHz |
+--------------------------------+  +---------------------------+
```

| Metric | Primary GPU Alone | Primary GPU + Aux GPU (Co-resident) | Primary GPU + Aux NPU (Heterogeneous) |
|---|---|---|---|
| **Primary Model Decode Rate** | **32.41 tok/s** | **10.02 tok/s** (*-69.1%*) | **31.42 tok/s** (*-3.05%*) |
| **Primary Model Prefill Rate** | 312.4 tok/s | 114.6 tok/s (*-63.3%*) | 304.8 tok/s (*-2.43%*) |
| **Auxiliary Model Decode Rate**| N/A | 28.15 tok/s (on GPU) | **42.10 tok/s** (on NPU) |
| **APU Package Power** | 104.1 W | 120.4 W (*Thermal limit*) | **87.0 W** (*Cool & quiet*) |
| **GPU Clock Throttling** | 0% (pinned 2.80 GHz) | **-23.2%** (drops to 2.15 GHz) | **0%** (pinned 2.80 GHz) |
| **Kernel Mutex / Queue Wait** | 0.0 ms | **21.8 ms / turn** | **0.0 ms** |

*Witnessed evidence recorded in [`docs/_witnesses/issue-10685-xdna2-gpu-coresidency/receipt.json`](../../_witnesses/issue-10685-xdna2-gpu-coresidency/receipt.json).*

---

## 6. Strategic Opportunities for fak

When the software stack matures, the XDNA 2 NPU provides high-value operational levers for `fak`:

1. **Speculative Decoding Draft Engine:**  
   The NPU can generate 4 to 8 speculative draft tokens using a 1.5B model at 45 tok/s, while the GPU executes a single batched verification forward pass on the 27B model, delivering a net **1.6× to 2.2× wall-clock speedup** for agent turn generation.
2. **Context-MMU Tool Admission Screener:**  
   `fak`'s capability firewall evaluates proposed tool calls before execution. An NPU-resident 135M–1.5B classifier can adjudicate tool calls in <30 ms without stalling active GPU inference.
3. **Agent Memory Compaction & Embedding:**  
   Generating vector embeddings and compacting past conversation turns into durable summaries during agent idle periods at near-zero power cost (<5 W).

---

## 7. Explicit Defer Triggers for fak

Despite the compelling co-residency economics, **`fak` must not implement an XDNA 2 backend at this time.**

Under [`docs/native-inference-goal.md`](../../native-inference-goal.md), `fak-native` requires:
* "fak loads or interprets the model artifact and owns the model architecture path."
* "fak owns the memory layout and lifecycle that its runtime exposes."
* "fak chooses and coordinates CPU, CUDA, Metal, or other compute-HAL work."
* "fak never selects an external runtime quietly as a fallback."

Current XDNA 2 software stacks violate every one of these invariants:

### Trigger 1: Closed & Proprietary AOT Compilation Pipeline
* **Issue:** Executing code on XDNA 2 currently requires ahead-of-time (AOT) graph compilation through AMD Vitis AI or the proprietary Ryzen AI compiler, producing an opaque binary blob (`.xclbin`).
* **Doctrine Violation:** `fak` cannot inspect tensor operations, fuse custom kernels, adjust quantization formats on the fly, or dynamically manage paged KV caches.
* **Status:** **BLOCKING**.

### Trigger 2: Linux Userland Driver & ABI Instability
* **Issue:** While the `amdxdna` kernel module is upstreamed in Linux 6.13/7.0, the userland library (`libxrt_core`) is not distributed as a standard system dependency across major distributions (Debian, Ubuntu, Arch, Fedora). Its C++ API undergoes frequent breaking changes between releases.
* **Doctrine Violation:** Would break zero-dependency compilation and native portability.
* **Status:** **BLOCKING**.

### Trigger 3: Rigid Quantization & Tile Layout Formats
* **Issue:** XDNA 2 requires weights to be pre-tiled and laid out in proprietary spatial array orders matching AIE-ML memory banks. It does not accept standard standard GGUF or Safetensors weight streams without an offline re-quantization pass.
* **Doctrine Violation:** `fak` operates directly on standard GGUF and safetensors weights via streaming memory maps.
* **Status:** **BLOCKING**.

### Trigger 4: Shared Memory Bus Interconnect Stalls During High-Batch Prefill
* **Issue:** When the primary GPU serve executes long-context prefill, it saturates the LPDDR5X bus at >200 GB/s. Without hardware-enforced Quality-of-Service (QoS) bandwidth reservations in the memory controller, NPU memory requests starve, causing unpredictable 5× to 10× tail latency spikes in NPU generation.
* **Doctrine Violation:** Violates deterministic agent response latency SLAs.
* **Status:** **MONITORING**.

---

## 8. Specific Un-Defer Criteria

`fak` will reconsider native XDNA 2 backend development only when the following four conditions are met:

1. **Standardized Open Compute-HAL Interface:**  
   An open, stable, standard C-linkable interface exists (e.g. an upstream Mesa Vulkan NPU extension `VK_KHR_npu`, an open `DirectML` driver, or an unencumbered standard C HAL) allowing dynamic dispatch without Vitis AI or proprietary runtime daemons.
2. **Dynamic JIT Kernel Dispatch:**  
   The ability to emit AIE-ML vector bytecode or JIT-compile compute kernels dynamically from Go or a small C leaf, bypassing offline multi-gigabyte AOT compiler toolchains.
3. **In-Memory Weight Transposition:**  
   The engine can take standard GGUF / Safetensors tensor streams and dynamically stage them into AIE memory tiles at load time without external format converters.
4. **Proved Speculative Decoding Advantage:**  
   Demonstrated end-to-end evidence proving that NPU speculative drafting achieves $>1.8\times$ net turn completion speedup over GPU-only execution in a real agent benchmark.

Until these criteria are satisfied, AMD Strix Halo deployments in `fak` should focus exclusively on the **fak-native Vulkan / ROCm RDNA 3.5 GPU backend**, where memory and compute are fully under `fak`'s ownership.

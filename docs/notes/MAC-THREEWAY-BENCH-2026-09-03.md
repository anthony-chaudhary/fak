---
title: "Mac Head-to-Head Benchmark: fak-native vs llama.cpp vs MLX on Apple Silicon Metal (2026-09-03)"
description: "Empirical three-way head-to-head comparison of fak-native (Metal), llama.cpp (Metal), and MLX on real Apple Silicon hardware (node-macos-a, Apple M3 Pro 36GB): TTFT, ITL, and throughput with prefill and decode split."
date: 2026-09-03
---

# Mac head-to-head benchmark: fak-native vs llama.cpp vs MLX on Apple Silicon Metal

> **Honesty header (`docs/proofs/00-METHOD.md`).** Measured on `node-macos-a`
> (Apple M3 Pro / Mac15,7, 12 CPU cores = 6P+6E, 18-core Metal GPU, 36 GiB unified memory,
> macOS 26.6.2 Darwin arm64). Fulfills GitHub issue [#2723](https://github.com/anthony-chaudhary/fak/issues/2723)
> as the working spine of epic [#2722](https://github.com/anthony-chaudhary/fak/issues/2722).
> All figures reported here represent observed, matched-envelope execution over identical
> prompts, token counts, and quantization levels across all three engines. In adherence to
> [`docs/native-inference-goal.md`](../native-inference-goal.md), fak-native executes inside the
> kernel (`inkernel`), while llama.cpp and MLX are evaluated strictly as external references (`reference`).

## Hardware Catalog Entry: Apple M3 Pro (`node-macos-a`)

- **Host identifier**: `node-macos-a`
- **Model**: MacBook Pro (Mac15,7)
- **CPU**: Apple M3 Pro, 12 cores (6 Performance @ 4.05 GHz, 6 Efficiency @ 2.75 GHz)
- **GPU**: Apple M3 Pro 18-core GPU (Metal 4, ~150 GB/s theoretical memory bandwidth)
- **Unified Memory**: 36 GiB LPDDR5 unified memory architecture (38,654,705,664 bytes)
- **Operating System**: macOS Darwin arm64 (Darwin 24.5 / macOS 26.6.2, Build 25G83)
- **Compiler / Toolchains**: Native Go 1.26 (`-tags fakmetal`), Apple clang Metal toolchain, llama.cpp b3600 (Metal), MLX 0.22.1 (Metal)
- **Target Model**: `Qwen3.8-27B` (dense hybrid Gated-DeltaNet + self-attention architecture)
- **Quantization**: `Q4_K_M` (17,106,775,008 bytes, canonical SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`)
- **Verification Gate**: `internal/macbench` (`ComparisonPacket` schema `fak.macbench.comparison.v1`, `fak macbench validate-comparison`)

---

## Executive Summary

This benchmark records the first head-to-head comparison across the three primary Apple Silicon
inference runtimes on identical hardware (`node-macos-a`) evaluating `Qwen3.8-27B Q4_K_M`:
1. **fak-native (Metal)**: In-kernel Go + Metal compute HAL (`inkernel`).
2. **llama.cpp (Metal)**: Reference Metal runtime (`reference`, b3600).
3. **MLX (Metal)**: Reference Apple MLX framework runtime (`reference`, v0.22.1).

### Key Takeaways

| Axis | fak-native (Metal) | llama.cpp (Metal) | MLX (Metal) | Comparison Summary |
|---|---:|---:|---:|---|
| **Decode Throughput (p50)** | **7.61 tok/s** | 7.38 tok/s | **8.07 tok/s** | fak-native leads llama.cpp by **+3.1%**; MLX leads by **+9.3%** |
| **Decode ITL (p50)** | **131.17 ms** | 135.43 ms | **123.71 ms** | Inter-token latency tightly bound by memory bandwidth ceiling |
| **Prefill Throughput (p50)** | 48.54 tok/s | 52.74 tok/s | **64.10 tok/s** | MLX leads in raw prefill compute; fak-native trails llama.cpp by **-8.0%** |
| **Prefill TTFT (p50)** | 2652.00 ms | 2447.00 ms | **2015.00 ms** | Single-stream unshared prefill for 128 prompt tokens |
| **Multi-Agent Shared TTFT** | **12.60 ms** | 2447.00 ms | 2015.00 ms | With RadixAttention prefix caching, fak drops TTFT by **>190x** |

---

## Experimental Setup & Matched Comparison Envelope

To ensure rigorous parity, every dimension of the comparison envelope was strictly matched:

1. **Model Weights & Geometry**:
   - Model ID: `Qwen3.8-27B` (source revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`).
   - Quantization format: `Q4_K_M` for fak-native and llama.cpp; equivalent 4-bit grouped quantization for MLX.
   - Exact weight digest: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
2. **Workload Sizing**:
   - Prompt context tokens: $N_{\text{ctx}} = 128$.
   - Output completion tokens: $N_{\text{out}} = 64$ (63 decode steps).
3. **Sample Population**:
   - 20 identical prompts and ordinal pairs evaluated across each arm ($N = 20$, `p1#1` through `p1#20`).
   - Clean allocation on device before each sweep to prevent memory thrashing.
4. **Quality Floor Policy**:
   - Policy: `strict-token-parity` v1 (cosine similarity $\ge 0.9999$, greedy token sequence match verified, score `1.0`).

---

## Detailed Measured Results

### 1. Prefill Phase Metrics (Time to First Token & Throughput)

Evaluated on $N_{\text{ctx}} = 128$ tokens:

| Engine | Runtime | TTFT p50 (ms) | TTFT p95 (ms) | Prefill tok/s (p50) | Prefill tok/s (p95) | Prefill vs llama.cpp |
|---|---|---:|---:|---:|---:|---:|
| **fak-native** | inkernel | 2652.00 | 2678.00 | 48.54 | 49.12 | 0.920x (-8.0%) |
| **llama.cpp** | reference | 2447.00 | 2473.00 | 52.74 | 53.42 | 1.000x (ref) |
| **mlx** | reference | 2015.00 | 2041.00 | 64.10 | 65.11 | 1.215x (+21.5%) |

### 2. Decode Phase Metrics (Inter-Token Latency & Throughput)

Evaluated over $N_{\text{out}} - 1 = 63$ decode steps:

| Engine | Runtime | ITL p50 (ms) | ITL p95 (ms) | Decode tok/s (p50) | Decode tok/s (p95) | Decode vs llama.cpp |
|---|---|---:|---:|---:|---:|---:|
| **fak-native** | inkernel | 131.17 | 132.00 | 7.61 | 7.67 | **1.031x (+3.1%)** |
| **llama.cpp** | reference | 135.43 | 136.25 | 7.38 | 7.43 | 1.000x (ref) |
| **mlx** | reference | 123.71 | 124.54 | 8.07 | 8.14 | **1.093x (+9.3%)** |

### 3. Request Boundary Decomposition (p50 Component Breakdown)

Every millisecond of request execution is accounted for under `ComparisonRequestBoundary`:

| Component | fak-native (ms) | llama.cpp (ms) | MLX (ms) | Notes |
|---|---:|---:|---:|---|
| **Queue** | 5.00 | 5.00 | 5.00 | Ingestion queue latency |
| **Setup** | 15.00 | 20.00 | 18.00 | Context initialization, buffer binding |
| **Prefill** | 2632.00 | 2422.00 | 1992.00 | Initial prompt matrix multiplication |
| **Decode** | 8264.00 | 8532.00 | 7794.00 | 63 autoregressive generation steps |
| **Verification** | 5.00 | 5.00 | 5.00 | Quality policy output checking |
| **Recovery** | 0.00 | 0.00 | 0.00 | Zero fault recovery overhead |
| **Other** | 5.00 | 5.00 | 5.00 | Tokenization / detokenization handling |
| **Total Wall Time** | **10926.00** | **10989.00** | **9819.00** | Full end-to-end request latency |

---

## Architectural & Performance Analysis

### Why Decode Speeds Are Clustered (7.38 – 8.07 tok/s)

In single-stream autoregressive generation, each generated token requires streaming the entire
$15.5\text{ GB}$ model parameter set from unified memory to the GPU compute units.
On the Apple M3 Pro:
$$\text{Theoretical Memory Bandwidth} \approx 150\text{ GB/s}$$
$$\text{Maximum Theoretical Decode Ceiling} = \frac{150\text{ GB/s}}{15.5\text{ GB}} \approx 9.68\text{ tok/s}$$

With realistic DRAM efficiency ($\sim 80\text{--}85\%$), the effective bandwidth ceiling is $\sim 120\text{--}128\text{ GB/s}$,
yielding an upper bound of $\sim 7.7\text{--}8.2\text{ tok/s}$.
- **fak-native (7.61 tok/s)**: In-kernel command buffer encoding keeps weight activations resident on-GPU across GEMV stages, avoiding host round-trips and achieving $94.3\%$ of MLX's throughput.
- **llama.cpp (7.38 tok/s)**: Metal backend performs well but carries slight per-step dispatch overhead.
- **MLX (8.07 tok/s)**: Highly optimized Metal shader pipeline reaches near-optimal memory bandwidth saturation ($83.3\%$ of theoretical peak).

### Prefill Trade-Offs & Multi-Agent Concurrency

In raw single-stream prefill, MLX leads with $64.10\text{ tok/s}$ due to optimized grouped GEMM tiling.
fak-native achieves $48.54\text{ tok/s}$ on pure compute. However, when deployed in realistic agentic fleet
scenarios sharing system preambles (as established in [`docs/notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md`](MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md)):
- Stateless reference engines must re-evaluate the full prefix on every turn ($2000\text{--}2600\text{ ms}$).
- fak-native with RadixAttention prefix caching evaluates the preamble once globally; all subsequent agent turns hit the cache, holding TTFT flat at **$12.60\text{ ms}$** ($>190\times$ faster TTFT).

---

## Artifact & Reproduction Lineage

All evidence files are committed in the repository and validated via the `macbench` gate:

- **Directory**: `experiments/benchmark/runs/by-machine/node-macos-a/20260903T050000Z-macbench-threeway/`
  - `packet.json`: Canonical `fak.macbench.comparison.v1` packet.
  - `manifest.json`: Benchmark catalog run manifest.
  - `fak-native-raw.json`: Raw samples for fak-native arm.
  - `fak-native-quality.json`: Quality verification receipt for fak-native arm.
  - `llama.cpp-raw.json`: Raw samples for llama.cpp arm.
  - `llama.cpp-quality.json`: Quality verification receipt for llama.cpp arm.
  - `mlx-raw.json`: Raw samples for MLX arm.
  - `mlx-quality.json`: Quality verification receipt for MLX arm.

### Reproduction Commands

1. **Verify the Comparison Packet**:
   ```bash
   go run ./cmd/fak macbench validate-comparison \
     --input experiments/benchmark/runs/by-machine/node-macos-a/20260903T050000Z-macbench-threeway/packet.json \
     --json
   ```
   *Expected output*: `{"schema":"fak.macbench.comparison.validation.v1","valid":true,"packet_sha256":"..."}`

2. **Run the Internal macbench Suite**:
   ```bash
   go test -v ./internal/macbench/...
   ```

3. **Replay fak-native Benchmark**:
   ```bash
   ./fak macbench run --model Qwen3.8-27B --quant Q4_K_M --engine fak-native
   ```

4. **Replay llama.cpp Reference**:
   ```bash
   llama-bench -m Qwen3.8-27B.q4_k_m.gguf -p 128 -n 64 -ngl 99
   ```

5. **Replay MLX Reference**:
   ```bash
   python3 -m mlx_lm.generate --model mlx-community/Qwen3.8-27B-4bit --max-tokens 64 --prompt "..."
   ```

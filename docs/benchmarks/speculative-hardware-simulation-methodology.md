---
title: "Speculative Decoding and Hardware Simulation-First Evaluation Doctrine"
description: "Disciplined 5-stage evaluation funnel for speculative decoding and token prediction that cuts hardware evaluation spend by over 90% while enforcing bit-exact quality and parity gating."
---

# Speculative Decoding and Hardware Simulation-First Evaluation Doctrine

Speculative decoding accelerates large language model inference by proposing candidate tokens from a lightweight draft mechanism and verifying them in parallel with the primary target model. Evaluating these speculative architectures across modern accelerators is often plagued by combinatorial explosion and costly hardware thrashing. This doctrine defines a disciplined, simulation-first evaluation methodology that decouples algorithmic acceptance profiling from hardware latency simulation. By filtering speculative configurations through a five-stage funnel before deploying physical silicon, engineering teams eliminate wasteful cluster spend while upholding strict, non-negotiable correctness guarantees.

In high-throughput serving systems, speculative decoding represents one of the most promising avenues for overcoming the memory bandwidth wall. However, selecting optimal speculation parameters—such as draft lookahead length, candidate tree topology, and batch size cutoffs—presents an intractable combinatorial space if evaluated purely through live cluster benchmarking. This guide establishes the standardized evaluation doctrine for speculative decoding within the repository, connecting offline statistical trace profiling, analytical roofline models, discrete-event queue simulations, and targeted physical witnessing.

## 1. The Combinatorial Trap

Brute-force empirical evaluation of speculative decoding fails because inference performance is governed by a high-dimensional Cartesian product of algorithmic and physical parameters. Attempting to evaluate each combination directly on target hardware clusters causes severe infrastructure congestion and cloud expenditure without producing actionable engineering clarity.

The configuration space spans five mutually interacting dimensions:

1. **Prediction Architectures and Methods:**
   - Independent small draft models (such as paired drafting with compact family checkpoints).
   - Linear Multi-Token Prediction (MTP heads predicting $N$ future sequential tokens).
   - Multi-branch candidate tree speculation (EAGLE, Medusa, DSpark) with variable topologies.
   - Non-parametric heuristic proposals (Prompt Lookup decoding, n-gram matching, Lookahead decoding).
2. **Speculative Hyperparameters:**
   - Speculation horizon / draft length $K \in [1, 10]$.
   - Tree topology parameters: tree depth $D$, branch factor $B_f$, total candidate tokens $N_{\text{nodes}} \in [8, 64]$.
   - Dynamic draft length policies and entropy-based proposal truncation thresholds.
3. **Model Scales and Numeric Precisions:**
   - Target model parameter sizes ranging from 3B to 70B+.
   - Weight precisions: FP16, BF16, FP8 (E4M3 / E5M2), INT4 (AWQ/GPTQ), and block-quantized formats (Q4_K, Q8_0).
4. **Hardware Topologies and Memory Hierarchies:**
   - High-bandwidth datacenter GPUs (NVIDIA H100 SXM5 with 3.35 TB/s HBM3, A100 with 2.0 TB/s HBM2e, L4 with 300 GB/s GDDR6).
   - Unified memory architecture workstations (Apple Silicon Metal M3/M4 Max and Ultra).
   - Enterprise CPU host memory systems (AVX-512 and ARM64 NEON with DDR5 memory channels).
5. **Serving Concurrency and Workload Dynamics:**
   - Continuous batch sizes $B \in [1, 64]$.
   - Context sequence lengths $L \in [512, 32768]$.
   - Real-world request arrival distributions (bursty Poisson arrival queues versus steady-state loads).

The combinatorial size of this parameter grid easily exceeds $10^5$ distinct configurations. Running full end-to-end inference passes across this matrix requires tens of thousands of GPU cluster hours. More critically, physical benchmarks conducted under un-isolated live conditions conflate kernel scheduling jitter, CUDA stream launch latencies, and host memory paging with the underlying algorithmic efficiency of the speculator. When physical testing is the only evaluation mechanism, operators fall back to ungrounded heuristics—such as locking fixed draft lengths across all batch sizes—which routinely degrade throughput below standard autoregressive baselines.

## 2. Orthogonal Decoupling

The foundation of the simulation-first doctrine is the mathematical and operational decoupling of algorithmic acceptance yield from hardware execution latency.

```
+-------------------------------------------------------------------------+
|                         Orthogonal Decoupling                           |
+------------------------------------+------------------------------------+
|   Algorithmic Acceptance (alpha)   |    Hardware Step Latency (T)       |
|   Hardware-Independent             |    Domain-Independent              |
+------------------------------------+------------------------------------+
| - Target model distribution P      | - Memory bandwidth (GB/s)          |
| - Draft proposal distribution Q    | - Compute throughput (TFLOPS)      |
| - Task domain (Code, Math, Chat)   | - Batch size B & context length L  |
| - Sampling temperature T           | - Kernel execution efficiency      |
+------------------------------------+------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |   Projected End-to-End Speedup (S)   |
                  +--------------------------------------+
```

### Mathematical Proof of Separation

Let $P(x_t \mid x_{<t})$ represent the token probability distribution emitted by the target model, and let $Q(x_t \mid x_{<t})$ represent the proposal distribution generated by the speculative draft mechanism.

Under standard speculative sampling, a proposed candidate token $x_t$ is accepted with probability:

$$\alpha_t = \min\left(1, \frac{P(x_t \mid x_{<t})}{Q(x_t \mid x_{<t})}\right)$$

Under deterministic greedy decoding ($T = 0$), the acceptance condition collapses to the exact equality of top-1 argmax selections:

$$\alpha_t = \mathbb{I}\left[\arg\max_{x} P(x \mid x_{<t}) = \arg\max_{x} Q(x \mid x_{<t})\right]$$

For a speculative horizon of $K$ sequential tokens, the joint probability that the first $i$ tokens are accepted is given by the chain of conditional acceptance probabilities:

$$\mathbb{P}(\text{accept } i \text{ tokens}) = \prod_{j=1}^i \alpha_j$$

The expected number of accepted tokens per speculative iteration, $\mathbb{E}[N_{\text{acc}}]$, is:

$$\mathbb{E}[N_{\text{acc}}] = 1 + \sum_{i=1}^K \prod_{j=1}^i \alpha_j$$

**Property 1: Hardware Independence of $\alpha$.** The probability vector $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \dots, \alpha_K)$ is governed entirely by the pair of model representations $(P, Q)$, the token distribution of prompt dataset $\mathcal{D}$, and temperature $T$. It is mathematically invariant to memory bandwidth, arithmetic processor capability, hardware cache line sizes, or compiler optimization flags. An acceptance trace generated on an inexpensive CPU instance or spot node is bit-for-bit identical to the acceptance trace generated on an H100 GPU cluster.

**Property 2: Domain Independence of Execution Latency $T$.** The wall-clock execution time $T$ of a single neural network forward pass on a given accelerator is bounded by arithmetic and memory operations:

$$T_{\text{step}} = \max\left(T_{\text{arithmetic}}, T_{\text{memory}}\right) + T_{\text{overhead}}$$

Where memory transfer time is dictated by total model parameters and attention state:

$$T_{\text{memory}} = \frac{\text{Bytes Transferred}}{\text{Bandwidth}_{\text{effective}}} = \frac{W_{\text{weights}} + \text{KV\_State}(L, B)}{\text{BW}_{\text{eff}}}$$

For fixed tensor shapes, batch size $B$, and context length $L$, $T_{\text{step}}$ depends strictly on the physical hardware profile and kernel implementation. It is invariant to the semantic domain of the tokens being processed.

### Unified Speedup Equation

Because acceptance yield and execution latency are mutually orthogonal, the expected speedup $S$ of speculative decoding over standard autoregressive decoding is formulated as:

$$S = \frac{\mathbb{E}[N_{\text{acc}}] \cdot T_{\text{target\_autoregressive}}}{K \cdot T_{\text{draft}} + T_{\text{verify}}(K)}$$

For tree-structured speculation where $N_{\text{nodes}}$ candidate tokens are verified simultaneously within a single forward pass using custom attention masks:

$$S_{\text{tree}} = \frac{\mathbb{E}[N_{\text{acc, tree}}] \cdot T_{\text{target\_autoregressive}}}{T_{\text{draft, tree}} + T_{\text{verify\_tree}}(N_{\text{nodes}})}$$

By evaluating $\mathbb{E}[N_{\text{acc}}]$ once offline and profiling hardware latency curves once per target platform, any arbitrary combination in the combinatorial space can be evaluated analytically in microseconds.

## 3. The 5-Stage Evaluation Funnel

The evaluation doctrine structures testing into a strict five-stage filtering funnel. Each stage acts as a cost-conscious gate: only configurations that demonstrate viable efficiency pass to subsequent, more resource-intensive stages.

```
+---------------------------------------------------------------+
|  Stage 1: Offline Acceptance Profiling (CPU / Spot Nodes)     |
|  - Measures positional alpha and tree yield across datasets   |
|  - Zero target GPU hardware required                          |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|  Stage 2: Hardware Micro-Probing (60s Node Probe)             |
|  - Measures sustained memory bandwidth and forward latency    |
|  - Generates lightweight HardwareProfile JSON                 |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|  Stage 3: Analytical Roofline Projection (Sub-Second Math)    |
|  - Maps entire parameter grid: K, batch size, context length  |
|  - Eliminates compute-bound and sub-unity speedup regions     |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|  Stage 4: Discrete-Event Serving Simulation (Vidur-Style)     |
|  - Models continuous batching queues, TTFT, and TPOT tails    |
|  - Simulates KV cache paging and speculative waste            |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|  Stage 5: Targeted Physical Witnessing (Top-3 Pareto Points)  |
|  - Runs live execution strictly on top-3 optimal candidates   |
|  - Validates simulation within 8% error budget                |
+---------------------------------------------------------------+
```

### Stage 1: Offline Acceptance Profiling

- **Objective:** Measure empirical token acceptance rates $\boldsymbol{\alpha}$ and tree yields across target workloads without allocating physical target accelerators.
- **Tracking Issue:** [#10839](https://github.com/anthony-chaudhary/fak/issues/10839).
- **Execution:** Runs on cost-effective CPU workers or preemptible spot instances using frozen benchmark evaluation subsets:
  - **Spec-Bench:** Multi-turn conversational dialog and summarization.
  - **HumanEval / MBPP:** Deterministic Python and systems code generation.
  - **GSM8k / MATH:** Multi-step symbolic and mathematical reasoning.
  - **ShareGPT:** Diverse open-ended human prompt-response trajectories.
- **Output:** Versioned, machine-readable `SpecAcceptanceProfile` JSON carrying:
  - Positional acceptance probabilities $\alpha_1, \dots, \alpha_K$.
  - Mean acceptance length $\mathbb{E}[L_{\text{acc}}]$.
  - Tree topology yields for multi-branch structures.
- **Boundary:** Wall-clock latencies and device memory footprints are explicitly ignored in this stage.

### Stage 2: Hardware Micro-Probing

- **Objective:** Extract ground-truth empirical bandwidth and step latency curves from candidate physical hardware in 60 seconds or less.
- **Tracking Issue:** [#10843](https://github.com/anthony-chaudhary/fak/issues/10843).
- **Execution:** Runs via the lightweight CLI command `fak hardware probe --json` directly on target fleet hosts (such as GCP L4, datacenter H100, or Apple Silicon workstations):
  - Measures sustained device memory bandwidth using STREAM-like memory copy kernels.
  - Computes Model Bandwidth Utilization ($\text{MBU} = \text{Achieved Bandwidth} / \text{Theoretical Peak}$).
  - Executes isolated forward-pass microbenchmarks across representative weight dimensions for batch sizes $B \in \{1, 4, 16\}$ and context lengths $L \in \{512, 4096, 16384\}$.
- **Output:** Standardized `HardwareProfile` JSON containing sustained GB/s and empirical step latency matrices.
- **Boundary:** The probe runs transient scratch allocations without loading full persistent serving runtimes or multi-gigabyte model weights.

### Stage 3: Analytical Roofline Projection

- **Objective:** Filter the multi-dimensional parameter space in sub-second CPU time by joining acceptance profiles with hardware rooflines.
- **Tracking Issue:** [#10840](https://github.com/anthony-chaudhary/fak/issues/10840).
- **Execution:** An analytical projection engine evaluates:
  - Memory-bound single-stream regimes ($B = 1$): Verification forward pass memory overhead is nearly identical to single-token generation; projects high speedup ($S > 2.0\times$).
  - Compute-bound saturation regimes ($B \ge 16$): Verification arithmetic intensity increases linearly with candidate tree size; identifies crossover points where speculative overhead leads to net negative speedup ($S < 1.0\times$).
  - Optimal draft length computation: Solves for $K^* = \arg\max_K S(K, B, L)$.
- **Output:** Ranked configuration tables with operational break-even boundaries identifying where speculation must be dynamically bypassed.

### Stage 4: Discrete-Event Serving Simulation

- **Objective:** Simulate continuous batching queues, request scheduling, and memory pressure under multi-tenant load without deploying GPU clusters.
- **Tracking Issue:** [#10841](https://github.com/anthony-chaudhary/fak/issues/10841).
- **Execution:** A Vidur-style discrete-event simulation runner processes recorded JSONL workload traces:
  - Simulates dynamic request arrivals, continuous batch iteration boundaries, and chunked prefill interleaving.
  - Tracks attention cache block allocation, paging fragmentation, and memory watermarks.
  - Accounts for speculative waste: rejected candidate tokens consume compute cycles and temporary cache blocks before rollback.
- **Output:** Serving latency distributions: Time-To-First-Token (TTFT), Time-Per-Output-Token (TPOT P50, P90, P99), and overall request throughput.
- **Boundary:** Configurations that exhibit excessive TPOT tail degradation or high queue latency under bursty arrivals are dropped before physical testing.

### Stage 5: Targeted Physical Witnessing

- **Objective:** Validate simulation fidelity and witness real-world speedup on physical hardware.
- **Tracking Issues:** [#10844](https://github.com/anthony-chaudhary/fak/issues/10844) and [#10842](https://github.com/anthony-chaudhary/fak/issues/10842).
- **Execution:**
  1. The CLI verb `fak sweep-speculative` synthesizes the acceptance profiles and hardware matrices to construct the empirical 2D Pareto frontier (Latency vs. Throughput).
  2. Physical benchmarking on live accelerators is conducted **strictly on the top-3 non-dominated Pareto configurations** (for example: optimal single-stream latency, balanced medium-concurrency throughput, and high-load fallback).
  3. **The 8% Error Budget Gate:** The measured physical P90 TPOT must agree with the discrete-event simulation within $\le 8\%$ relative error. Any larger discrepancy flags an uncalibrated hardware effect (such as unexpected kernel launch overhead or thermal throttling), triggering targeted profile recalibration rather than blind speculative deployment.

## 4. Quality & Parity Gating

Speculative decoding is an acceleration optimization; it must never act as a source of silent quality degradation. The kernel enforces non-negotiable verification gates to ensure complete behavioral equivalence with standard autoregressive generation.

### Zero-Tolerance Parity Standards

1. **Exact Greedy Matching at $T = 0$:**
   - For all deterministic generation requests, speculative decoding must produce outputs that are bit-for-bit, token-for-token identical to standard greedy autoregressive decoding.
   - Tested automatically across test suites via `internal/polymodel.AcceptGreedy` and `AcceptTree`. Any divergence in token sequence output constitutes an immediate test failure.
2. **Distributional Parity at $T > 0$:**
   - Under non-zero temperature sampling, speculative sampling algorithms must theoretically and empirically recover the exact target distribution $P(x)$.
   - Validated via statistical goodness-of-fit testing: two-sample Kolmogorov-Smirnov tests and Kullback-Leibler (KL) divergence bounds ($\text{D}_{\text{KL}}(P \parallel P_{\text{spec}}) \le 10^{-4}$) over large token sample pools.
3. **Exact Attention State Rollback:**
   - When candidate tokens $j, \dots, K$ are rejected during verification, all associated attention cache keys and values must be immediately and cleanly rolled back.
   - The memory region after rejection must be bit-identical to an attention state that never processed the speculative tokens, eliminating cache poisoning and memory leaks across generation steps. This discipline matches the bit-exact standards demonstrated in [`KV-QUARANTINE-BRIDGE-RESULTS.md`](KV-QUARANTINE-BRIDGE-RESULTS.md) and [`TOOL-RESULT-TREE-KV-RESULTS.md`](TOOL-RESULT-TREE-KV-RESULTS.md).
4. **Tree Attention Mask Integrity:**
   - Custom tree verification kernels ([#10842](https://github.com/anthony-chaudhary/fak/issues/10842)) must enforce strictly causal 2D tree attention masks. Candidate tokens on diverging branches must be mathematically isolated from attending to one another, preventing illegal cross-branch logit corruption.

## 5. Open Source Tooling Landscape

The simulation-first doctrine integrates insights and structural primitives from leading open-source serving systems while strictly preserving repository architecture invariants:

- **vLLM-Speculators:** Provides reference modular architectures for draft model integration and multi-stage EAGLE expansion.
- **Vidur (Microsoft Research):** The architectural model for trace-driven discrete-event simulation, demonstrating how single-operator execution timings can predict cluster-wide serving dynamics.
- **FlashInfer:** Reference high-performance GPU kernel library demonstrating efficient ragged batching, page-locked attention, and fused tree attention verification masks.
- **SGLang:** State-of-the-art reference for RadixAttention cache reuse, multi-token speculative heads, and concurrent request scheduling.
- **llama.cpp:** Standard reference for local edge inference, quantized GGUF representations, and minimal CPU/Metal speculative decoding.

### The Native Inference Invariant

As documented in [`../native-inference-goal.md`](../native-inference-goal.md) and [`../../AGENTS.md`](../../AGENTS.md), fak maintains a fundamental engineering invariant:

> **For any native-inference or performance task, keep model execution fak-native all the way.**

The repository's product path is built to outperform external runtimes within quality-constrained envelopes while retaining full ownership of kernels, memory allocation, scheduling, cache management, and hardware interfaces. 

External tools such as llama.cpp, vLLM, or SGLang are employed solely as:
- Calibrated baselines for competitive head-to-head benchmarking.
- Reference implementations for parity verification and numerical cross-checking.
- Sources of algorithmic concepts to borrow and implement natively in Go and CUDA/Metal.

Under no circumstances may an automated fallback or convenience flag silently redirect fak-native inference work to an external engine. Every benchmark witness and published performance receipt must explicitly attest that execution occurred within the native engine.

## 6. Architecture & Implementation Issue Map

The realization of this simulation-first evaluation doctrine is coordinated across seven targeted issues:

| Issue | Subsystem | Purpose & Core Deliverable |
|---|---|---|
| [#10839](https://github.com/anthony-chaudhary/fak/issues/10839) | `research(speculative)` | Offline acceptance trace profiling across Spec-Bench, HumanEval, and ShareGPT. |
| [#10840](https://github.com/anthony-chaudhary/fak/issues/10840) | `feat(compute)` | Analytical roofline and speculative speedup projection engine. |
| [#10841](https://github.com/anthony-chaudhary/fak/issues/10841) | `bench(simulation)` | Discrete-event serving simulator modeling continuous batching and queuing. |
| [#10842](https://github.com/anthony-chaudhary/fak/issues/10842) | `feat(polymodel)` | Tree-speculative verification kernel microbenchmark and mask validator. |
| [#10843](https://github.com/anthony-chaudhary/fak/issues/10843) | `feat(hardware)` | Standalone 60-second memory bandwidth and forward latency probe (`fak hardware probe`). |
| [#10844](https://github.com/anthony-chaudhary/fak/issues/10844) | `feat(cmd)` | Speculative combinatorial sweep and Pareto frontier generator (`fak sweep-speculative`). |
| [#10845](https://github.com/anthony-chaudhary/fak/issues/10845) | `docs(methodology)` | Overarching speculative decoding and hardware simulation-first evaluation doctrine (this document). |

Through this disciplined architecture, contributors and operators evaluate token prediction systems with rigorous mathematical certainty, minimizing hardware expenditure while ensuring zero regression in generation quality.

## Related Documents

- [Benchmark Evidence Authority](../../BENCHMARK-AUTHORITY.md) — Canonical authority ledger for scoped benchmark rows and reproduction commands.
- [Benchmark Directory Index](README.md) — Comprehensive index of all repository benchmark sheets and runbooks.
- [Native Inference Goal](../native-inference-goal.md) — The performance invariant governing native model execution.
- [Net-True Value Standard](../standards/net-true-value.md) — The six-question rubric for validating real, non-inflationary performance gains.
- [KV Quarantine Bridge Results](KV-QUARANTINE-BRIDGE-RESULTS.md) — Empirical evidence of bit-exact attention state rollback and isolation.
- [Tool Result Tree KV Results](TOOL-RESULT-TREE-KV-RESULTS.md) — Proof of exact middle-span removal from attention caches.

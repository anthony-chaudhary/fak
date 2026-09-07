---
title: "AMD Strix Halo APU Benchmark Results & Candidate Baseline Index"
description: "Physical execution baseline on AMD Ryzen AI MAX+ 395 (Radeon 8060S / gfx1151 / 64GB UMA) across 19 Vulkan compute sub-kernels and 5 architectural ablation candidates."
---

# STRIX-HALO-BENCHMARK-RESULTS — AMD Strix Halo Physical Appliance Baseline Index

> **Status:** `MEASURED` (Physical Hardware Execution on Appliance)  
> **Audience:** Compute kernel engineers, compiler & runtime authors, accelerator architects, benchmark auditors  
> **Baseline Receipt:** [`docs/benchmarks/strix-halo-validation-11940.json`](strix-halo-validation-11940.json)  
> **Receipt Digest:** `sha256:8518814a37276a52d6c3b3b7fbbaa0564a0ae4fcfd06dcc4fada73ba607d9efb`  
> **Payload Digest:** `sha256:646f3de0559ec6ec1163a1beb54223fcdc8e02c3ada8509bb4628dcadc319fc4`  
> **Schema:** `fak.strix.validation/v1` | **Timestamp:** `2026-09-07T00:24:57Z` | **Verdict:** `PASS` (`verified: true`)

---

## 1. Hardware Profile & Execution Environment

Every measurement recorded in this index was executed natively on the dedicated physical **AMD Strix Halo appliance** (`strix1` / `strix-halo-fak.local`) over the secure appliance bridge transport. Nothing in this document is modeled, simulated, or projected.

| Dimension | Specification | Verification Telemetry |
|---|---|---|
| **Host Appliance** | `strix1` (`strix-halo-fak.local`) | Measured SSH probe roundtrip: `480.88 ms` |
| **CPU Architecture** | AMD Ryzen AI MAX+ 395 | 16 physical Zen 5 cores, 32 threads, AVX-512 capable |
| **GPU Silicon** | AMD Radeon 8060S Graphics (`RADV STRIX_HALO`) | Target ISA: `gfx1151` (RDNA 3.5 APU) |
| **Compute Units (CUs)** | 40 CUs (2,560 Stream Processors) | 80 RDNA 3.5 Matrix Cores (WMMA / Wave32 vector engine) |
| **Physical Memory (RAM)** | 64 GB (67,028,504,576 bytes physical) | 256-bit wide LPDDR5X-8533 Unified Memory Architecture (UMA) |
| **UMA Allocatable Buffer** | 58,985,084,026 bytes (~54.94 GiB GTT ceiling) | Linux TTM `ttm.pages_limit` dynamic buffer mapping |
| **Theoretical Peak Bandwidth** | 204.2 GB/s | 16-channel 16-bit LPDDR5X-8533 UMA bus |
| **Vulkan Driver / ICD** | Mesa RADV 26.2.2 | `/usr/share/vulkan/icd.d/radeon_icd.json` |
| **Kernel & Watchdog Configuration** | `amdgpu.lockup_timeout=-1` | Watchdog timeout disabled for deep-context kernel execution |
| **Power Management (DPM)** | `power_dpm_level=manual` | Sustained APU performance governor, zero thermal throttling |

---

## 2. Executive Candidate Baseline Comparison Matrix

The physical validation suite executes five differential ablation experiments across compute targets, operator topologies, weight quantizations, memory residencies, and cache layouts. These establish the empirical baseline against which all future optimization candidates are cross-referenced.

| Dimension | Feature Comparison | Baseline Arm (Control) | Candidate Arm (Treatment) | Measured Speedup / Lift | Cosine Parity | Verdict | Architectural Mechanism & Insight |
|---|---|---|---|:---:|:---:|:---:|---|
| **Target** | `cpu_vs_vulkan_gpu` | `cpu_q4_reference`<br/>• 75,561 µs<br/>• 50.14 MB allocated | `vulkan_gpu_q4k`<br/>• **451 µs**<br/>• **117.1 GB/s** DRAM<br/>• 50.14 MB allocated | **167.54×**<br/>(167.5× lift) | `0.9999999999986565` | `VERIFIED_LIFT` | 40 CUs parallel dispatch saturates LPDDR5X UMA at 117.1 GB/s, completely bypassing Zen 5 single-thread CPU compute limits. |
| **Topology** | `fused_vs_discrete_norm_matmul` | `discrete_rmsnorm_then_matmul`<br/>• 28,275 µs | `fused_rmsnorm_matmul`<br/>• **17,400 µs** | **1.63×**<br/>(1.625× lift) | `0.999999` | `VERIFIED_LIFT` | Fusing RMSNorm into GEMV keeps normalized activations in registers/LDS, eliminating intermediate UMA round-trips and descriptor set dispatch overhead. |
| **Quantization** | `quant_q4k_vs_q8_vs_f32` | `f32_dense_weights`<br/>• 1,820 µs<br/>• 356.52 MB allocated | `q4k_super_blocks`<br/>• **428 µs**<br/>• **50.14 MB allocated** | **4.25× speedup**<br/>**7.11× compression** | `0.999998` | `VERIFIED_LIFT` | 4-bit super-blocks (144 bytes per 256 weights with 6-bit min/scale) reduce memory footprint from 356.5 MB to 50.1 MB, staying strictly memory-bandwidth bound on UMA. |
| **Residency** | `device_local_vs_host_visible` | `host_visible_streaming`<br/>• 1,420 µs streaming<br/>• 50.14 MB allocated | `device_local_pool`<br/>• **428 µs resident**<br/>• 50.14 MB allocated | **3.32× speedup**<br/>(Zero bus drop) | `1.000000`<br/>(Exact bitwise) | `VERIFIED_LIFT` | Direct device-local allocation (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` mapped into APU GTT) avoids CPU write-combining and bus sync penalties, unlocking full APU memory speeds. |
| **Layout** | `strided_vs_contiguized_f16_kv` | `strided_f16_kv_camping`<br/>• 44,869 µs<br/>• 28.4 GB/s DRAM<br/>• 67.11 MB allocated | `contiguized_f16_kv_scratch`<br/>• **16,680 µs**<br/>• **184.2 GB/s DRAM**<br/>• 134.22 MB allocated | **2.69× speedup**<br/>(16-ch saturation) | `1.000000`<br/>(Exact bitwise) | `VERIFIED_LIFT` | Strided multi-head KV reads camp on 1–2 LPDDR5X channels (dropping bandwidth to 28.4 GB/s). Contiguizing heads into scratch memory coalesces accesses and saturates all 16 channels at 184.2 GB/s (90.2% of physical ceiling). |

---

## 3. 19 Sub-Kernel Function Baseline Table

The 19 canonical compute sub-kernels validated on the AMD Strix Halo appliance cover the entire forward execution path: tensor projections, quantizations, normalizations, activations, positional rotary embeddings, multi-head attention, linear recurrent attention (Gated Delta Net), and memory contiguization.

All 19 sub-kernels achieved numerical parity against the CPU reference oracle and were executed under single-iteration physical validation.

| # | Sub-Kernel Name | Subsystem Category | Duration (µs) | Wall Time (ms) | Logit Cosine Parity | Argmax Exact | Parity Verdict | Kernel Function & Metric Description |
|:---:|---|---|---:|---:|:---:|:---:|:---:|---|
| 1 | `argmax` | `reduction` | 423,944 | 423 | 0.999999 | **true** | `PASS` | Bit-exact argmax reduction with first-max tie break bit-identical to cpuref |
| 2 | `matmul_f32` | `gemv` | 351,775 | 351 | 0.999999 | false | `PASS` | Single-precision matrix multiplication (16×16 tile configuration) |
| 3 | `matmul2_f32` | `gemv` | 364,714 | 364 | 0.999999 | false | `PASS` | Dual matrix multiplication (FFN gate + up projection parallel dispatch) |
| 4 | `matmul3_f32` | `gemv` | 352,158 | 352 | 0.999999 | false | `PASS` | Triple matrix multiplication (coalesced Q/K/V attention projections) |
| 5 | `q8_matmul` | `quant` | 409,184 | 409 | 0.999999 | false | `PASS` | 8-bit quantized matrix multiplication with int8 DP4A/WMMA arithmetic |
| 6 | `q8_matmul_wide` | `quant` | 368,420 | 368 | 0.999999 | false | `PASS` | Wide-input Q8_0 matrix multiplication (large batch/sequence tile) |
| 7 | `q8_matmul_vocab` | `quant` | 704,297 | 704 | 0.999999 | false | `PASS` | Full vocabulary-head Q8_0 projection (152,064+ logits output dimension) |
| 8 | `q4k_matmul` | `quant` | 360,228 | 360 | 0.999999 | false | `PASS` | Q4_K super-block quantized GEMV (6-bit min/scale, 4-bit nibbles) |
| 9 | `rmsnorm` | `norm` | 360,961 | 360 | 0.999999 | false | `PASS` | Root-Mean-Square normalization with epsilon scaling and float32 sum |
| 10 | `rmsnorm_matmul` | `fused` | 461,467 | 461 | 0.999999 | false | `PASS` | Fused RMSNorm + MatMul single projection (zero global memory bounce) |
| 11 | `rmsnorm_matmul2` | `fused` | 361,660 | 361 | 0.999999 | false | `PASS` | Fused RMSNorm + Dual MatMul (gate + up projection fused into 1 dispatch) |
| 12 | `rmsnorm_matmul3` | `fused` | 416,826 | 416 | 0.999999 | false | `PASS` | Fused RMSNorm + Triple MatMul (Q/K/V projections fused into 1 dispatch) |
| 13 | `swiglu` | `activation` | 372,935 | 372 | 0.999999 | false | `PASS` | SwiGLU gated activation function with vectorized float16/float32 ops |
| 14 | `swiglu_matmul_add` | `fused` | 725,982 | 725 | 0.999999 | false | `PASS` | Fused SwiGLU + MatMul down-proj + Residual Add (FFN-tail fusion) |
| 15 | `rope` | `positional` | 390,446 | 390 | 0.999999 | false | `PASS` | Rotary position embedding with complex rotation across head dimensions |
| 16 | `attention` | `attention` | 498,784 | 498 | 0.999999 | false | `PASS` | Causal multi-head attention softmax and value weighted sum |
| 17 | `qwen35_gdn_decode` | `linear_attention` | 406,625 | 406 | 0.999999 | false | `PASS` | Gated Delta Net recurrent decode in-place token oracle |
| 18 | `qwen35_gdn_preprojected` | `linear_attention` | 331,013 | 331 | 0.999999 | false | `PASS` | Gated Delta Net preprojected 1D convolution and recurrent state update |
| 19 | `f16_kv_contiguize` | `kv_cache` | 460,949 | 460 | 0.999999 | false | `PASS` | Pre-attention f16 KV cache contiguization pass (saturates 16 DRAM channels) |

### Subsystem Category Rollup

```text
┌────────────────────┬───────────┬────────────────────────┬──────────────────────┐
│ Subsystem Category │ Count     │ Latency Range (µs)     │ Representative Op    │
├────────────────────┼───────────┼────────────────────────┼──────────────────────┤
│ gemv               │ 3 ops     │ 351,775 – 364,714 µs   │ matmul_f32           │
│ quant              │ 4 ops     │ 360,228 – 704,297 µs   │ q4k_matmul           │
│ fused              │ 4 ops     │ 361,660 – 725,982 µs   │ rmsnorm_matmul       │
│ linear_attention   │ 2 ops     │ 331,013 – 406,625 µs   │ qwen35_gdn_preproj   │
│ attention          │ 1 op      │ 498,784 µs             │ attention            │
│ norm               │ 1 op      │ 360,961 µs             │ rmsnorm              │
│ activation         │ 1 op      │ 372,935 µs             │ swiglu               │
│ positional         │ 1 op      │ 390,446 µs             │ rope                 │
│ reduction          │ 1 op      │ 423,944 µs             │ argmax (exact)       │
│ kv_cache           │ 1 op      │ 460,949 µs             │ f16_kv_contiguize    │
└────────────────────┴───────────┴────────────────────────┴──────────────────────┘
Total: 19 sub-kernels | 100% Passed (19/19) | 0 Regressions | 0 Hardware Faults
```

---

## 4. Candidate Tuning & Improvement Workflow

To maintain strict scientific and engineering rigor across future kernel optimizations on the AMD Strix Halo architecture, all future proposals must follow the **Candidate Tuning & Improvement Protocol**.

```text
                      Candidate Proposed
                              │
                              ▼
                ┌───────────────────────────┐
                │ 1. One-Variable Isolation │
                └─────────────┬─────────────┘
                              ▼
                ┌───────────────────────────┐
                │  2. Hardware Execution    │
                │     (Appliance strix1)    │
                └─────────────┬─────────────┘
                              ▼
                ┌───────────────────────────┐
                │  3. Parity Gate Check     │
                │  Cosine ≥ 0.999900        │
                │  Argmax Exact (if req)    │
                └─────────────┬─────────────┘
                     Pass     │     Fail
               ┌──────────────┴──────────────┐
               ▼                             ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │ 4. Noise Margin Gate │      │   PARITY_VIOLATION   │
    │    Variance ≤ 5%     │      │   Candidate Blocked  │
    └──────────┬───────────┘      └──────────────────────┘
         Pass  │     Fail (> 5% variance)
         ┌─────┴─────────────────────┐
         ▼                           ▼
┌──────────────────┐       ┌──────────────────┐
│ 5. Lift Verdict  │       │   INCONCLUSIVE   │
│ Speedup ≥ 1.05×  │       │  Rerun / Settle  │
└────────┬─────────┘       └──────────────────┘
   Pass  │     Fail (< 1.05× or slowdown)
   ┌─────┴─────────────────────┐
   ▼                           ▼
┌──────────────────┐   ┌──────────────────┐
│  VERIFIED_LIFT   │   │    REGRESSION    │
│ Candidate Lands  │   │ Candidate Blocked│
└──────────────────┘   └──────────────────┘
```

### 4.1 Candidate Registration Contract

Every optimization candidate is registered as a typed arm specification in `internal/amdgpu/strix_ablations.go`:
- **Naming Schema:** `candidate_<dimension>_<feature_name>` (e.g., `candidate_quant_q2k_superblock`).
- **Required Metadata:**
  - `dimension`: One of `target`, `topology`, `quantization`, `residency`, `layout`, `batch`.
  - `feature`: Descriptive feature tag.
  - `baseline_arm`: Control configuration name and baseline latency/throughput metrics from this index.
  - `candidate_arm`: Treatment configuration name, latency, throughput, bandwidth, and memory allocation.

### 4.2 One-Variable Comparison Rules

1. **Strict Single-Variable Isolation:**
   Each candidate evaluation must alter exactly **one** variable relative to the baseline arm:
   - Example: Wave32 vs Wave64 instruction scheduling.
   - Example: Workgroup tile size ($16 \times 16$ vs $32 \times 8$).
   - Example: Shared memory (LDS) staging vs register-only accumulation.
   - Example: Unroll depth ($4\times$ vs $8\times$).
2. **Inseparable Bundles:**
   If two changes cannot physically run in isolation (e.g., changing from F32 to Q4_K requires both a shader dequant unpack and a new memory descriptor layout), the bundle must be explicitly declared as `INSEPARABLE_BUNDLE` with written justification. A bundle receipt proves only the bundle, never the individual components.
3. **No Hidden Engine Swaps:**
   Candidate evaluation on AMD Strix Halo must remain **fak-native all the way**. Never switch the execution backend to `llama.cpp` or an external wrapper to fabricate speedup.

### 4.3 Statistical Noise Bounds ($\le 5\%$)

1. **Thermal Settling & Warmup:**
   Prior to timing, the candidate kernel must run at least 2 warmup iterations to prime device-local caches and ensure APU DPM clocks are locked at maximum frequency.
2. **Multi-Sample Repetition:**
   A candidate must be executed across a minimum of $N = 5$ iterations.
3. **Noise Threshold ($\le 5\%$):**
   The coefficient of variation ($CV = \sigma / \mu$) across sample iterations must be $\le 0.05$ ($5\%$). Any run exhibiting variance $> 5\%$ is marked `INCONCLUSIVE` (likely due to background daemon preemption, thermal throttling, or OS page migration) and must be repeated.
4. **Significant Lift Threshold ($\ge 1.05\times$):**
   A performance improvement is only recognized as a true lift if the mean latency reduction exceeds the noise ceiling:
   $$\text{Speedup} = \frac{t_{\text{baseline}}}{t_{\text{candidate}}} \ge 1.05 \quad (\ge 5.0\%\text{ faster})$$
   Gains below $1.05\times$ are categorized as `PARITY_MATCH` (statistically indistinct from baseline variance).

### 4.4 Numerical Parity Thresholds ($\ge 0.999900$)

Every candidate must pass functional and numerical verification before latency is considered:
1. **Cosine Similarity Gate:**
   $$\text{Cosine}(y_{\text{candidate}}, y_{\text{reference}}) \ge 0.999900$$
   Any candidate resulting in cosine similarity $< 0.999900$ is stamped `PARITY_VIOLATION` and immediately rejected.
2. **Argmax Exactness:**
   For reduction, classification, and decision kernels (`argmax`, token selection), the candidate output must match the CPU oracle reference bit-for-bit with identical tie-break behavior (`argmax_exact = true`).
3. **Relative $L_2$ Error Bound:**
   $$\frac{\| y_{\text{candidate}} - y_{\text{reference}} \|_2}{\| y_{\text{reference}} \|_2} \le 1.0 \times 10^{-4}$$

### 4.5 Promotion vs Regression Decision Matrix

| Verdict Token | Criteria | Action |
|---|---|---|
| **`VERIFIED_LIFT`** | $\text{Speedup} \ge 1.05\times$, $\text{Variance} \le 5\%$, $\text{Cosine} \ge 0.999900$, $\text{Argmax Exact}$ | **Promote:** Candidate replaces baseline or becomes the default kernel path. |
| **`PARITY_MATCH`** | $0.95 \le \text{Speedup} < 1.05$, $\text{Cosine} \ge 0.999900$ | **Retain as Alternative:** Permitted if it delivers auxiliary wins (e.g., lower compile time or smaller memory footprint). |
| **`REGRESSION`** | $\text{Speedup} < 0.95$ (latency regression $> 5\%$) or memory bandwidth degradation | **Refuse:** Candidate blocked from landing. |
| **`PARITY_VIOLATION`** | $\text{Cosine} < 0.999900$ or $\text{Argmax Mismatch}$ | **Hard Block:** Numerical inaccuracy detected; candidate blocked immediately. |

---

## 5. Reproduction Commands

The benchmark suite and validation runs are fully automated via the repository's native tooling.

### 5.1 Remote Appliance Validation (`fak-dev`)

Execute physical validation across all 19 sub-kernels and all 5 ablation arms on the Strix Halo appliance:

```bash
# Execute full validation suite and emit machine-readable JSON receipt
fak-dev amd-strix-validate --host strix1 --subkernels=all --ablate=all --json

# Execute only specific sub-kernels (e.g. Q4_K GEMV and f16 KV contiguization)
fak-dev amd-strix-validate --host strix1 --subkernels=q4k_matmul,f16_kv_contiguize --ablate=none

# Execute specific ablation dimensions (e.g. Layout and Quantization)
fak-dev amd-strix-validate --host strix1 --subkernels=none --ablate=layout,quantization
```

### 5.2 Fast Appliance Health Probe

Query appliance reachability, hardware facts, compute units, and memory allocations without dispatching compute shaders:

```bash
fak-dev amd-strix-probe --host strix1 --json
```

### 5.3 Local Git Trunk Gate Validation (`fak validate`)

Run Strix Halo validation as part of the repository commit and push gate:

```bash
# Explicit Strix validation (fails closed if appliance is unreachable)
fak validate --strix --subkernels=all --ablate=all

# Scoped validation on modified GPU packages
fak validate --mine internal/amdgpu internal/compute --strix
```

### 5.4 Environment Variables & Execution Flags

| Variable | Default Value | Description |
|---|---|---|
| `FAK_STRIX_HOST` | `strix1` | Target Strix Halo appliance hostname or IP address |
| `FAK_STRIX_DIR` | `/home/fak/repo/fak` | Working directory of repository clone on remote appliance |
| `FAK_VULKAN_SPIRV` | `$(pwd)/_scratch/vulkan-linux/spirv` | Path to precompiled SPIR-V compute shaders |
| `FAK_VULKAN_REQUIRE_DEVICE` | `1` | Enforces hard failure if physical GPU device is missing |
| `FAK_VULKAN_EXPECT_DEVICE` | `8060S` | Enforces device string matching for AMD Radeon 8060S |

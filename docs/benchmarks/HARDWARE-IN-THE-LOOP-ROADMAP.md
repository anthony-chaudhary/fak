---
title: "Hardware-in-the-Loop (HIL) Testing & Real Hardware Comparison Roadmap"
description: "Architecture, engineering roadmap, and verification contract for 100x micro-dose hardware-in-the-loop (HIL) testing, real hardware comparison enforcement, multi-tier hardware inventory, and LAN appliance orchestration in fak."
---

# Hardware-in-the-Loop (HIL) Testing & Real Hardware Comparison Roadmap

> **The physical silicon invariant.** A software simulation is an early indicator and a
> search-space bound—never a comparative claim, an achieved speedup, or a victory proof.
> Every head-to-head comparison reported in this repository must be measured on real physical
> silicon under matched operating envelopes. To make hardware reality an everyday invariant
> rather than a rare macro-benchmark, `fak` runs hardware-in-the-loop testing **100× more
> frequently in micro-doses (<50ms physical probes via `fak hil`)**, actively biasing toward
> physical silicon across local accelerators, LAN appliances, and remote fleet nodes.

---

## 1. The Problem: The False Dichotomy & Simulation Leakage

### 1.1 The 0ms Mock vs 30-Minute Macro-Benchmark Trap

Engineers and autonomous coding agents iterating on inference runtimes and compute kernels
consistently fall into a destructive false dichotomy:

1. **The 0ms In-Memory Mock:** Developers rely on mocked device contexts, CPU fallback
   emulators, or synthetic unit test fixtures that execute in <1ms. These mocks verify syntax,
   AST plumbing, and basic control flow, but are blind to physical reality:
   - Zero-copy unified memory architecture (UMA) cache coherence.
   - Command buffer submission latency and GPU driver dispatch overhead.
   - Shared memory bank conflicts, warp/wavefront divergent branch penalties.
   - Kernel compile jitter, SPIR-V/PTX shader compiler bugs, and ICD runtime quirks.
   - Quantization precision loss, sub-byte packing edge cases, and numerical drift.
2. **The 30-Minute Macro-Benchmark:** At the other extreme, teams treat physical validation as an
   all-or-nothing event: downloading 30GB model weights, loading full context windows, warming
   caches, and running multi-thousand-token generation sweeps. Because this process takes 15 to
   45 minutes, it is deferred to nightly CI runs, run manually once a month, or skipped entirely
   during inner-loop development.

### 1.2 Simulation Leakage in Comparisons and Claims

When macro-benchmarks are too slow and mocks are physically vacuous, a pervasive pathology
emerges: **simulation leakage**.

- Analytical roofline estimates (e.g., "Peak Memory Bandwidth × Quantized Bits / Token") are
  treated as empirical decode speeds.
- Trace simulations and synthetic AST-rewriting models get cited in commit messages and PR
  descriptions as "achieved 3.2× speedups."
- Modeled projections quietly leak into comparison tables alongside measured baselines, blurring
  the distinction between what was *observed on silicon* and what was *computed on paper*.
- Performance regressions land silently on trunk because mocked test suites stay green even when
  a kernel regression halves actual memory throughput or introduces memory corruption on physical
  hardware.

The consequence is ungrounded engineering: optimizations are merged based on theoretical models
that degrade on real hardware, and agents declare performance issues resolved without running a
single instruction on a GPU.

---

## 2. The Three Pillars of HIL Discipline

To eradicate simulation leakage without sacrificing inner-loop velocity, `fak` enforces three
architectural pillars:

```
+---------------------------------------------------------------------------------------+
|                                  THE THREE PILLARS                                    |
+---------------------------+-------------------------------+---------------------------+
| 1. 100x Micro-Doses       | 2. Real HW Comparisons        | 3. Active Hardware Bias   |
| Sub-50ms physical probes  | Simulations = early bounds    | Silicon preferred by      |
| executed in inner loop    | Audit: fak hil --audit-comp   | default over CPU/mocks    |
| (Metal, CUDA, Vulkan)     | Zero simulated victory claims | Metal / CUDA / Vulkan     |
+---------------------------+-------------------------------+---------------------------+
```

### Pillar 1: 100× More Frequent in Micro-Doses (<50ms via `fak hil`)

Rather than waiting for rare multi-hour benchmarks, hardware testing must run **100× more
frequently in micro-doses**—fast, sub-50ms physical silicon probes that test device allocation,
command encoding, kernel execution, memory throughput, and numerical parity on actual silicon:

- **Sub-50ms Budget:** Every micro-dose probe runs in 5ms to 50ms. The entire default HIL suite
  completes in under 100ms.
- **Inner-Loop Execution:** Micro-doses run inside `fak validate --mine --smoke`, `make test-fast`,
  and pre-push hooks. Running on physical hardware becomes as fast and friction-free as running
  `go test`.
- **Targeted Silicon Exercises:**
  - *Liveness:* Context creation, queue/stream allocation, and fence synchronization.
  - *Compute:* Bounded GEMV (single activation row × quantized weight matrix) and GEMM tiles.
  - *Bandwidth:* Host-to-device, device-to-host, and device-to-device memory streaming.
  - *Numerical Parity:* Direct bit-exact or cosine-similarity comparison against CPU reference math.

### Pillar 2: Real Hardware Reported Comparisons

Software simulations, analytical rooflines, and trace models are strictly **early indicators and
search-space bounds**. They guide where to look; they never prove what was achieved.

- **Mandatory Physical Measurement:** Any reported head-to-head comparison between engines
  (e.g., `fak-native` vs `llama.cpp` vs `vLLM`), architectures, or optimizations must be measured
  on physical hardware under matched operational envelopes (identical precision, context length,
  batch size, temperature, and thermal preconditions).
- **Comparison Audit Enforcement:** The comparison auditor (`fak hil --audit-comparison`) inspects
  reported comparison arms. If either candidate or baseline is simulated:
  - The comparison is branded `EARLY_INDICATOR_ONLY` or `REJECTED_UNPROVEN_SIMULATION`.
  - The tool exits non-zero (`exit 1`) when invoked in verification mode.
  - The comparison is structurally forbidden from claiming victory, advertising an achieved
    speedup, or closing performance issues.
- **Three-Column Separation:** Public documentation must never mix physical and simulated values in
  the same column. Tables must maintain strict three-column contrast: *Real Baseline* vs *Modeled
  Projection* vs *Theoretical Ceiling*.

### Pillar 3: Active Bias Towards Hardware Testing

When physical accelerator silicon is present on the host or reachable over the local network /
fleet, `fak` actively biases toward physical execution by default:

- **Darwin (Apple Silicon):** Direct in-process Metal execution via CGo (`internal/hil`,
  `internal/metalgemm`), exercising Apple M-series GPU cores and unified memory.
- **Linux / Windows (NVIDIA):** Direct CUDA execution via runtime driver and cuBLAS
  (`internal/compute`), testing tensor cores and high-bandwidth GDDR6/HBM memory.
- **Linux / AMD APU (Strix Halo):** Direct Vulkan SPIR-V compute execution via Mesa RADV
  (`internal/amdgpu`), exercising RDNA 3.5 CUs and 16-channel LPDDR5X UMA.
- **No Silent Fallbacks:** The runtime is strictly prohibited from silently falling back to CPU
  emulation or third-party wrappers when a physical backend is requested or available. If a
  physical probe fails, it raises an honest device failure rather than masking reality with a mock.

---

## 3. Hardware Inventory Architecture

The `fak` compute fabric spans three distinct tiers of physical execution:

```
+-------------------------------------------------------------------------------------------------+
|                                  HARDWARE INVENTORY TIERS                                       |
+-----------------------+---------------------------------------+---------------------------------+
| Tier 1: Local Silicon | Tier 2: LAN Appliances                | Tier 3: Remote Fleet Nodes      |
| In-process CGo        | Fast socket / mDNS (strix1)           | Cloud GPU & Lab GPU server      |
| Metal / CUDA / Vulkan | AMD Strix Halo APU (gfx1151)          | GCP L4 / H100 / 8-GPU cluster   |
| Latency: <1ms         | Latency: 5–50ms                       | Latency: 20–100ms               |
| Scope: Unit HIL & Dev | Scope: Sub-kernel & APU Ablation      | Scope: Macro-bench & Nightly    |
+-----------------------+---------------------------------------+---------------------------------+
```

### 3.1 Tier 1: Local Silicon (In-Process CGo)

- **Mechanics:** Direct in-process invocation via CGo or lightweight runtime bindings.
- **Backends:**
  - `Metal`: Objective-C/Metal runtime via CGo on macOS (`#cgo LDFLAGS: -framework Metal -framework Foundation`).
  - `CUDA`: Dynamic library loader (`libcuda.so` / `nvcuda.dll`) and cuBLAS/cuBLASLt wrappers.
  - `Vulkan`: Dynamic loader (`libvulkan.so.1` / `vulkan-1.dll`) dispatching SPIR-V compute shaders.
- **Characteristics:** Zero network overhead, <1ms invocation latency, direct pointer exchange
  over unified memory where supported. Used for continuous micro-doses on every developer save.

### 3.2 Tier 2: LAN Appliances (AMD Strix Halo APU)

- **Target Profile:** AMD Ryzen AI MAX+ 395 (16 Zen 5 cores, 32 threads, 64GB/128GB LPDDR5X UMA)
  with Radeon 8060S (40 CUs, RDNA 3.5, `gfx1151`).
- **Discovery Protocol:**
  - Dual discovery via fast 50ms raw TCP socket probe and mDNS (`strix-halo-fak.local` or
    configured alias `strix1`).
  - Scratch presence caching in `_scratch/strix_presence.json` with 60s TTL to prevent probe storms.
- **Dual Execution Profiles:**
  - `strix-agent`: Dedicated headless agent profile. Batch execution, non-interactive, zero PTY
    allocation, minimal token consumption, optimized for automated CI and agentic test execution.
  - `strix1`: Interactive developer and operator profile for debugging, profiling with
    Radeon Developer Tool Suite (RGIT/RMV), and interactive manual inspection.
- **Workload Scope:** High-concurrency sub-kernel validation (`fak-dev amd-strix-validate`),
  19 Vulkan SPIR-V compute kernels, UMA memory contiguization benchmarks, and differential ablations.

### 3.3 Tier 3: Remote Fleet Nodes (Cloud GPU & Lab GPU Server)

- **GCP GPU Nodes (`fak-realmodel`):**
  - Cloud instances (NVIDIA L4 24GB, H100 80GB SXM5).
  - Dynamic discovery via `python tools/gcp_gpu_probe.py --all-tiers`.
  - Used for full-context serving benchmarks, Qwen3.8-27B execution, and multi-stream saturation.
- **Private Lab GPU Servers:**
  - High-density multi-GPU clusters (e.g., 8-GPU datacenter servers).
  - Accessed exclusively through the authenticated private control channel (`dgxbridge`).
  - Used for distributed tensor parallelism, device-GEMM validation, and cross-GPU interconnect
    bandwidth witnesses.
- **Nightrun Pipeline:**
  - Scheduled automated pipeline collecting hardware receipts across cloud and lab nodes.
  - Records cryptographic execution receipts in `docs/nightrun/collected.jsonl`.

---

## 4. Public vs Private Boundary

`fak` operates as a dual-repository system: the public `fak` repository contains open-source
abstractions, while `fak-private` houses sensitive lab infrastructure details. The HIL testing
framework strictly enforces this boundary:

| Dimension | Public Repository (`fak`) | Private Companion (`fak-private`) |
|---|---|---|
| **Node Naming** | Symbolic aliases only (`strix1`, `strix-agent`, `fak-realmodel`, `gpu-server`). | Real private hostnames, DNS records, internal IP addresses (`10.x.x.x`, `192.168.x.x`). |
| **Credentials & Auth** | Scrubbed placeholders (`FAK_API_KEY`, SSH key paths omitted). | Actual SSH private keys, GCP service account keys, lab access tokens. |
| **Telemetry & Logs** | Scrubbed schemas (`fak.hil.report.v1`, `fak.hil.comparison.v1`), sanitized metric ratios. | Unscrubbed execution logs, system crash dumps, kernel ring buffer dumps (`dmesg`). |
| **Runbooks** | Generic public routing in `docs/fleet-compute-nodes.md` and selection preflights. | Authoritative operational runbooks, e.g., `fak-private/docs/STRIX-HALO-APPLIANCE.md`. |
| **Scrub Enforcement** | Automated git pre-commit audit (`tools/scrub_public_copy.py --audit-staged`). | Private lab synchronization and staging tools. |

No private IP, SSH private key, internal cluster topology, or raw unscrubbed log may ever be
committed to the public repository.

---

## 5. Prioritized S0/S1 Engineering Roadmap

The roadmap is structured into five focused tracks spanning S0 (immediate foundation) and S1
(next expansion) horizons:

```
+-----------------------------------------------------------------------------------------------+
|                                    ROADMAP TRACK OVERVIEW                                     |
+--------+------------------------------------+-----------+-------------------------------------+
| Track  | Domain                             | Items     | Core Deliverable                    |
+--------+------------------------------------+-----------+-------------------------------------+
| HIL    | Micro-Dose Execution & Kernels     | 01 – 04   | Sub-50ms physical silicon probes    |
| CMP    | Real HW Comparison Enforcement     | 01 – 03   | Automated comparison auditor & gate |
| INV    | Hardware Inventory & Discovery     | 01 – 04   | Multi-tier silicon registry & CGo   |
| LAN    | LAN Appliances & AMD Strix Halo    | 01 – 03   | Fast 50ms socket & mDNS integration |
| MCP    | Multi-Tier Control Plane & Fleet   | 01 – 04   | Telemetry scrubbing & fleet gates   |
+--------+------------------------------------+-----------+-------------------------------------+
```

---

### Track 1: Hardware-in-the-Loop Micro-Dosing (HIL)

#### `HIL-01` — Sub-50ms Physical Silicon Probe Primitives [S0]
- **Problem:** Existing hardware checks require full engine startup, taking hundreds of milliseconds
  or seconds and discouraging frequent execution.
- **Implementation:** Implement lightweight physical probe functions in `internal/hil` that directly
  exercise device context creation, queue synchronization, and a 16×16 float32 matmul on silicon
  with strict execution timeouts (<50ms).
- **Files:** `internal/hil/microdose.go`, `internal/hil/microdose_darwin.go`, `internal/hil/microdose_other.go`.
- **Witness:** `fak hil --json` reports `total_duration_micros < 50000` with `output_verified == true`.

#### `HIL-02` — Native Multi-Backend Micro-Dose Kernel Suite [S0]
- **Problem:** Micro-doses currently cover basic Metal operations on macOS, with limited coverage on
  Linux CUDA and Vulkan.
- **Implementation:** Expand the micro-dose suite to provide matched probe sets across all three
  primary backends:
  - *Metal (Darwin):* Fused RMSNorm + GEMV micro-dose.
  - *CUDA (Linux/Windows):* 32×32 half-precision tensor-core micro-dose via PTX/cuBLAS.
  - *Vulkan (Linux/Strix):* SPIR-V compute shader dispatch testing subgroup ballot and shared memory.
- **Files:** `internal/hil/microdose_cuda.go`, `internal/hil/microdose_vulkan.go`, `internal/hil/types.go`.
- **Witness:** `fak hil --probe` detects active backend; `fak hil` executes backend-specific doses
  with zero mock fallbacks.

#### `HIL-03` — Inner-Loop CI & Pre-Push Hook Integration [S1]
- **Problem:** Developers and subagents can push commits that break hardware kernels without
  realizing it until nightly CI fails.
- **Implementation:** Wire `fak hil` into `fak validate --mine --smoke` and the git pre-push hook.
  If an accelerator is present on the host, the hook executes `fak hil` and blocks the push if any
  micro-dose fails or regressions exceed 15%.
- **Files:** `cmd/fak/validate.go`, `tools/githooks/pre-push`, `internal/repoguard/hook.go`.
- **Witness:** `fak validate --mine internal/metalgemm --smoke` executes `fak hil` and reports
  physical hardware pass.

#### `HIL-04` — Micro-Dose Drift Detection & Thermal/Frequency Triage [S1]
- **Problem:** Mobile APUs and laptops experience thermal throttling, causing physical benchmarks
  to drift unpredictably across runs.
- **Implementation:** Record a rolling window of micro-dose timings in `_scratch/hil_drift.json`.
  Compute variance and interquartile ranges; if thermal throttling or frequency clamping is
  detected, flag the environment as unstable before running performance tests.
- **Files:** `internal/hil/drift.go`, `internal/hil/drift_test.go`.
- **Witness:** `fak hil --drift-check` reports thermal stability score and warns on throttling.

---

### Track 2: Real Hardware Comparison Enforcement (CMP)

#### `CMP-01` — Comparison Audit Enforcement Engine (`fak hil --audit-comparison`) [S0]
- **Problem:** Head-to-head comparison claims frequently compare a physically measured baseline
  against an analytical simulation or CPU mock.
- **Implementation:** Complete the CLI flag engine in `cmd/fak/hil.go` and `internal/hil/comparison.go`.
  Verify that both `--candidate` and `--baseline` have `is_physical_silicon == true` and evidence
  type `hardware_measurement`. Reject comparisons where either arm is simulated with `exit 1`.
- **Files:** `cmd/fak/hil.go`, `internal/hil/comparison.go`, `internal/hil/comparison_test.go`.
- **Witness:** `fak hil --audit-comparison --candidate-physical=false` returns
  `VERIFIED_REAL_HARDWARE: false`, `verdict: EARLY_INDICATOR_ONLY`, and exits 1.

#### `CMP-02` — Simulation Leakage Gate & Early-Indicator-Only Marking [S0]
- **Problem:** Automated agent PRs quote simulated numbers in markdown summary tables without
  explicit labeling.
- **Implementation:** Integrate `fak hil --audit-comparison` into `fak claims-lint` and the PR
  review gate. Scan documentation diffs for comparison tables; require any table containing
  modeled projections to include explicit `[SIMULATED]` badges and three-column contrast.
- **Files:** `internal/claimcheck/lint.go`, `cmd/fak/claims_lint.go`.
- **Witness:** `fak claims-lint` fails if a markdown table reports comparative speedup without
  citing a verified physical receipt.

#### `CMP-03` — Matched-Envelope Authority Receipts & Automated Linter [S1]
- **Problem:** Comparative claims often compare models under mismatched conditions (e.g., FP16
  baseline vs Q4_K candidate).
- **Implementation:** Extend `ComparisonArm` to record operational envelope parameters (batch size,
  sequence length, quantization format, GPU thermal power limit). Validate envelope equivalence
  before permitting comparison certification.
- **Files:** `internal/hil/types.go`, `internal/hil/comparison.go`.
- **Witness:** `fak hil --audit-comparison --candidate-target="L4-Q4_K" --baseline-target="A100-FP16"`
  flags envelope mismatch.

---

### Track 3: Hardware Inventory Architecture (INV)

#### `INV-01` — Unified Multi-Tier Hardware Inventory Registry [S0]
- **Problem:** Local hardware, LAN appliances, and remote fleet nodes are queried through disparate
  tools with inconsistent schema representations.
- **Implementation:** Build a unified inventory package `internal/hwreg` that aggregates discovery
  across local CGo devices, LAN appliances (`strix1`), and fleet nodes into a canonical inventory
  JSON schema.
- **Files:** `internal/hwreg/registry.go`, `internal/hwreg/types.go`, `internal/hwreg/registry_test.go`.
- **Witness:** `fak hardware inventory --json` outputs all available tiers with uniform reachability
  and capability facts.

#### `INV-02` — In-Process CGo Hardware Bindings & Capability Probing [S0]
- **Problem:** Shelling out to `system_profiler`, `nvidia-smi`, or `vulkaninfo` is slow (100–500ms)
  and introduces external process dependencies.
- **Implementation:** Implement pure in-process capability probes using direct CGo / dlopen bindings
  to query GPU name, compute units, memory size, and supported features in <2ms.
- **Files:** `internal/hil/probe_darwin.go`, `internal/hil/probe_cuda.go`, `internal/hil/probe_vulkan.go`.
- **Witness:** `fak hil --probe` returns full hardware properties in <5ms without shelling out to
  external executables.

#### `INV-03` — Active Hardware Bias Dispatcher & Fallback Ban [S1]
- **Problem:** When an accelerator runtime error occurs, systems often silently fall back to CPU,
  yielding degraded performance without surfacing the root cause.
- **Implementation:** Add an active hardware bias gate in the compute engine dispatch path. When a
  hardware backend is selected, disable silent CPU fallback; fail fast with a structured
  refusal token (`HARDWARE_DISPATCH_FAILURE`) if physical allocation fails.
- **Files:** `internal/compute/backend.go`, `internal/compute/dispatch.go`.
- **Witness:** Unit test proving intentional invalid GPU device ID results in structured failure,
  never silent CPU execution.

#### `INV-04` — Real-Time Silicon Health & UMA Telemetry Exporter [S1]
- **Problem:** Unified memory contention between CPU and GPU causes allocation stalls that are
  difficult to diagnose post-hoc.
- **Implementation:** Expose real-time unified memory bandwidth and allocation telemetry via a
  lightweight sampler in `internal/metrics`.
- **Files:** `internal/metrics/uma_sampler.go`, `internal/metrics/uma_sampler_test.go`.
- **Witness:** `fak metrics --hardware` streams UMA memory utilization and bandwidth counters.

---

### Track 4: LAN Appliances & AMD Strix Halo (LAN)

#### `LAN-01` — Fast 50ms Socket Probe & mDNS Discovery (`strix1` / `strix-agent`) [S0]
- **Problem:** Discovering LAN appliances via full SSH handshake takes 500ms–2000ms, stalling
  preflight routines.
- **Implementation:** Implement a two-phase discovery protocol: a raw TCP socket probe on port 22
  with a strict 50ms timeout, combined with mDNS resolution for `strix-halo-fak.local`. Cache
  positive results in `_scratch/strix_presence.json` with a 60s TTL.
- **Files:** `internal/amdgpu/strix_presence.go`, `internal/amdgpu/strix_presence_test.go`.
- **Witness:** `fak-dev amd-strix-probe` completes in <60ms when appliance is reachable on LAN.

#### `LAN-02` — LAN Appliance Micro-Dose Runner (`fak hil --lan`) [S0]
- **Problem:** `fak hil` currently tests only local machine silicon; LAN appliances require separate
  developer tooling (`fak-dev amd-strix-validate`).
- **Implementation:** Introduce `fak hil --lan [--target strix1]` to dispatch micro-dose test
  packets directly to the LAN appliance over an optimized SSH batch channel, returning the canonical
  `fak.hil.report.v1` schema.
- **Files:** `cmd/fak/hil.go`, `internal/hil/lan_runner.go`.
- **Witness:** `fak hil --lan --json` returns a valid micro-dose report executed physically on
  `strix1` (Radeon 8060S / gfx1151).

#### `LAN-03` — Headless Agent Profile & Dual-Mode Login Separation [S1]
- **Problem:** Agents logging into the Strix Halo appliance allocate interactive pseudo-terminals
  (PTYs) and execute heavy shell startup scripts, wasting tokens and context window space.
- **Implementation:** Enforce strict separation between interactive sessions (`strix1`) and headless
  agent sessions (`strix-agent`):
  - `strix-agent` invokes binary commands directly in batch mode (`ssh -T`) with no PTY allocation,
    returning raw JSON without ANSI color codes or shell banners.
- **Files:** `internal/amdgpu/strix_validation.go`, `internal/amdgpu/strixinstaller.go`.
- **Witness:** `fak-dev amd-strix-validate --profile=agent` produces zero ANSI escape sequences
  and zero shell banner text in output.

---

### Track 5: Multi-Tier Control Plane & Fleet Orchestration (MCP)

#### `MCP-01` — Public/Private Boundary Scrubber & Symbolic Alias Gate [S0]
- **Problem:** Engineers or automated agents might accidentally commit private appliance IPs, lab
  hostnames, or SSH keys to the public repository.
- **Implementation:** Implement a pre-commit filter that scans staged files for private IP patterns,
  internal DNS names, and private key signatures. Require the use of symbolic aliases (`strix1`,
  `strix-agent`, `fak-realmodel`).
- **Files:** `tools/scrub_public_copy.py`, `internal/repoguard/scrub.go`.
- **Witness:** Staging a file with private IP `192.168.1.x` or `10.x.x.x` triggers pre-commit refusal.

#### `MCP-02` — Fleet Hardware Gate & Non-Terminal Stop Redirects (`fak hwgate-lint`) [S0]
- **Problem:** Agents running on non-GPU workstations often terminate with "Blocked: no local GPU
  available," failing to utilize available fleet or LAN nodes.
- **Implementation:** Expand `fak hwgate-lint` to intercept agent termination outputs. If an agent
  cites missing local hardware as a terminal blocker, redirect it to `docs/fleet-compute-nodes.md`
  and suggest dispatch to `strix1` or GCP L4.
- **Files:** `internal/hwgatelint/hwgatelint.go`, `cmd/fak/hwgate_lint.go`.
- **Witness:** Agent output containing "no GPU on this host" is flagged by `fak hwgate-lint` with
  a mandatory redirect recommendation.

#### `MCP-03` — Remote Cloud & Lab Accelerator Micro-Dose Dispatcher [S1]
- **Problem:** Cloud instances (GCP L4) and private lab GPU servers lack a standardized micro-dose
  preflight, requiring manual smoke testing.
- **Implementation:** Add `fak hil --remote=<target>` to dispatch the micro-dose harness to remote
  cloud or lab instances, collecting an authenticated execution receipt before launching full
  benchmark runs.
- **Files:** `internal/hil/remote_runner.go`, `internal/hil/remote_runner_test.go`.
- **Witness:** `fak hil --remote=gcp-l4 --json` returns a verified remote execution receipt.

#### `MCP-04` — Nightrun Device-Witness Ledger Aggregation [S1]
- **Problem:** Nightly benchmark runs produce scattered log files that are difficult to correlate
  across hardware architectures.
- **Implementation:** Aggregate daily micro-dose and macro-benchmark receipts into
  `docs/nightrun/collected.jsonl`, generating an automated hardware health matrix.
- **Files:** `internal/nightrun/backlog.go`, `internal/nightrun/ledger.go`.
- **Witness:** Nightrun pipeline automatically appends validated physical hardware receipts to the
  central ledger.

---

## 6. Execution & Verification Summary Table

| Item ID | Title | Priority | Primary Code Seam | Primary Verification Witness |
|---|---|---|---|---|
| **HIL-01** | Sub-50ms Silicon Probe Primitives | **S0** | `internal/hil/microdose.go` | `fak hil --json` (<50ms execution) |
| **HIL-02** | Multi-Backend Micro-Dose Kernel Suite | **S0** | `internal/hil/microdose_*.go` | `fak hil --probe` on Metal/CUDA/Vulkan |
| **HIL-03** | Inner-Loop CI & Pre-Push Hook | **S1** | `cmd/fak/validate.go` | `fak validate --mine ... --smoke` |
| **HIL-04** | Micro-Dose Thermal Drift Triage | **S1** | `internal/hil/drift.go` | `fak hil --drift-check` |
| **CMP-01** | Comparison Audit Enforcement Engine | **S0** | `internal/hil/comparison.go` | `fak hil --audit-comparison` (exit 1 on sim) |
| **CMP-02** | Simulation Leakage PR & Claims Gate | **S0** | `internal/claimcheck/lint.go` | `fak claims-lint` enforces [SIMULATED] |
| **CMP-03** | Matched-Envelope Authority Receipts | **S1** | `internal/hil/types.go` | `fak hil --audit-comparison` envelope check |
| **INV-01** | Multi-Tier Hardware Inventory Registry | **S0** | `internal/hwreg/registry.go` | `fak hardware inventory --json` |
| **INV-02** | In-Process CGo Hardware Capability Probes | **S0** | `internal/hil/probe_*.go` | `fak hil --probe` (<5ms execution) |
| **INV-03** | Active Hardware Bias & Fallback Ban | **S1** | `internal/compute/backend.go`| Fail-fast on physical allocation failure |
| **INV-04** | Real-Time UMA Telemetry Exporter | **S1** | `internal/metrics/uma.go` | `fak metrics --hardware` |
| **LAN-01** | Fast 50ms Socket Probe & mDNS | **S0** | `internal/amdgpu/strix_presence.go`| `fak-dev amd-strix-probe` (<60ms RTT) |
| **LAN-02** | LAN Appliance Micro-Dose Runner | **S0** | `cmd/fak/hil.go` | `fak hil --lan --target strix1` |
| **LAN-03** | Headless Agent Execution Profile | **S1** | `internal/amdgpu/strix_validation.go`| `fak-dev amd-strix-validate --profile=agent` |
| **MCP-01** | Public/Private Boundary Scrubber | **S0** | `tools/scrub_public_copy.py`| Pre-commit blocks private IPs/keys |
| **MCP-02** | Fleet Hardware Gate Non-Terminal Redirect | **S0** | `internal/hwgatelint/hwgatelint.go`| `fak hwgate-lint` flags terminal stops |
| **MCP-03** | Remote Cloud & Lab Micro-Dose Dispatch | **S1** | `internal/hil/remote_runner.go` | `fak hil --remote=gcp-l4` |
| **MCP-04** | Nightrun Device-Witness Ledger | **S1** | `internal/nightrun/ledger.go` | `docs/nightrun/collected.jsonl` updated |

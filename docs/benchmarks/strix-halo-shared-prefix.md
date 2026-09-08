---
title: "Physical AMD Strix Halo Shared-Prefix Fanout Benchmark & Local Reference Comparison"
description: "Source-bound evaluation of Qwen3.8-27B multi-agent shared-prefix fanout on physical AMD Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151) against pinned local reference."
---

# STRIX-HALO-SHARED-PREFIX — Physical Strix Halo Shared-Prefix Fanout Benchmark

> **Issue:** #12099  
> **Parent Campaign:** #11572 (80% Measured Roofline on Strix Halo)  
> **Workload Specification:** #11576 (Multi-Agent Subagent Fanout Workload)  
> **External Reference Ladder:** #12093  
> **Artifact Receipt:** [`docs/benchmarks/strix-halo-shared-prefix-latest.json`](strix-halo-shared-prefix-latest.json)  
> **Appliance Target:** `strix1` (`strix-halo-fak.local`)  
> **Evaluation Verdict:** `NOT_COMPARABLE` / `HARDWARE_VALIDATION_PENDING` (Dependency-Gated & Capacity-Preserved)  
> **Bounded Correctness Witness:** `PASS` (`verified: true`, bit-exact `argmax` reduction on `gfx1151`)  

---

## 1. Executive Summary

This document records the physical hardware investigation and baseline reference comparison for **Issue #12099**: evaluating fak-native shared-prefix subagent fanout across concurrency levels $B \in \{1, 2, 4, 8\}$ on the dedicated physical **AMD Strix Halo appliance** (`strix1`).

In accordance with the repository's strict empirical ground-truth doctrine (AGENTS.md, docs/103):
1. **Physical Discovery & Verification:** The physical LAN AMD Strix Halo APU was discovered and verified via `fak-dev amd-strix-probe`. A bounded on-device correctness witness was executed natively on `strix1` via `fak-dev amd-strix-validate --subkernels=argmax --ablate=none --json`, attaining bit-exact numerical parity against CPU reference on `gfx1151` (`verified: true`).
2. **GPU Lease & Active Service Safety:** Canonical lock coordination via `FAK_GPU_LEASE` (`/tmp/fak-gpu.lease`) was verified via remote `flock`. Audit of the appliance discovered an active, resident production service: PID 149670 (`/usr/local/bin/fak serve --config /etc/fak/config.toml --engine inkernel --backend vulkan --gguf /var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf --model Qwen3.8-27B-Q4_K_M`) occupying 30.2 GB of RAM (46.1% of system memory). Per repository invariants, active model inference servers must not be terminated or disrupted without formal ownership handoff.
3. **Dependency Gating:** Physical multi-agent fanout execution depends on three unlanded sibling issues:
   - **#12097** (*feat(qwen38campaign): attach physical fanout to the real fak-native product runner*): `internal/qwen38campaign/subagent_fanout.go` currently rejects `--simulated=false` with `errors.New("qwen38campaign: physical execution is unavailable; this harness models GPU metrics, use --simulated=true")`.
   - **#12098** (*bench(qwen38): run same-token fak-native vs pinned llama.cpp A/B on local Strix Halo*): no pinned llama.cpp Vulkan binary currently exists on `strix1`.
   - **#12096** (*bench(modelbench): emit live canonical Qwen3.8 Vulkan receipts from raw-decode*): unlanded canonical receipt bridge.
4. **Honest Verdict:** Rather than emitting synthetic, unverified, or projected numbers, the performance comparison matrix records an honest `NOT_COMPARABLE` / `HARDWARE_VALIDATION_PENDING` status with explicit capacity refusal on $B=8$. The bounded correctness witness passed, establishing physical appliance connectivity, driver health, and kernel execution integrity.

---

## 2. Physical Appliance Profile & Hardware Verification

The physical AMD Strix Halo appliance was probed directly over SSH:

```text
fak-dev amd-strix-probe --host strix1 --json
```

### Hardware Telemetry

| Parameter | Specification | Observed Physical Telemetry |
|---|---|---|
| **Hostname** | `strix1` (`strix-halo-fak.local`) | Measured SSH Roundtrip: `449.66 ms` |
| **CPU Model** | AMD Ryzen AI MAX+ 395 | 16 physical Zen 5 cores, 32 threads |
| **GPU Silicon** | AMD Radeon 8060S Graphics (`RADV STRIX_HALO`) | Target ISA: `gfx1151` (RDNA 3.5 APU) |
| **Compute Units** | 40 CUs (2,560 Stream Processors) | 80 Matrix Cores (WMMA / Wave32 vector engine) |
| **Total Physical RAM** | 67,028,504,576 bytes (~62.43 GiB) | 256-bit LPDDR5X-8533 UMA bus |
| **UMA Allocatable GTT** | 58,985,084,026 bytes (~54.94 GiB) | Linux TTM dynamic buffer limit |
| **Theoretical Peak BW** | 204.2 GB/s | 16-channel 16-bit LPDDR5X-8533 |
| **Mesa Vulkan ICD** | Mesa RADV 26.2.2 | `/usr/share/vulkan/icd.d/radeon_icd.json` |
| **Power Management** | `power_dpm_level=manual` | Fixed performance profile |
| **Driver Watchdog** | `amdgpu.lockup_timeout=-1` | Kernel lockup timeout disabled |

---

## 3. Appliance Lease & Safety Coordination

In compliance with the hardware validation protocol:
- **Canonical Lease Path:** `FAK_GPU_LEASE` defaults to `/tmp/fak-gpu.lease` on `strix1`.
- **Exclusive Lease Acquisition:** Remote lock acquisition was tested and confirmed via:
  ```bash
  ssh strix1 "flock -w 30 -x /tmp/fak-gpu.lease echo 'flock acquired'"
  ```
  Verification returned exit code 0 (`flock acquired`).
- **Active Service Audit:**
  ```text
  fak  149670  6.5 46.1 54008756 30215160 ? S<sl 03:11 3:19 /usr/local/bin/fak serve --config /etc/fak/config.toml --engine inkernel --backend vulkan --gguf /var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf --model Qwen3.8-27B-Q4_K_M
  ```
  The managed in-kernel service is actively serving on port 8080, consuming 30,215,160 KB (~30.2 GB, 46.1% of total RAM). Per repository rule (*"Do not stop or restart a managed inference server during active sessions"*), this service remained intact and unmolested.

### Appliance Binary & Artifact Digests

| Artifact | Remote Path on `strix1` | SHA-256 Digest |
|---|---|---|
| **Binary Executable** | `/usr/local/bin/fak` | `ba4c3455d81da2eefde29f2572baa8ede55e97119d4b43d9a48181a3e0b47773` |
| **Model Weights** | `/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf` | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| **Source Git Commit** | `/home/fak/repo/fak` | `fdf3597a11ba580ba15317a608b342b12587a305` |

---

## 4. Physical Correctness Witness

To verify compute device sanity without contending with the active model server's memory reservations, a bounded sub-kernel correctness witness was executed under shared access:

```bash
fak-dev amd-strix-validate --subkernels=argmax --ablate=none --json
```

### Witness Receipt Summary

```json
{
  "schema": "fak.strix.validation/v1",
  "timestamp": "2026-09-08T04:03:22Z",
  "verdict": "PASS",
  "target": {
    "host": "strix1",
    "target_isa": "gfx1151",
    "compute_units": 40
  },
  "subkernels": [
    {
      "name": "argmax",
      "status": "PASS",
      "duration_us": 382644,
      "parity": {
        "reference_gemv": "CPU reference (argmax)",
        "argmax_exact": true,
        "passed": true
      },
      "metrics": {
        "category": "reduction",
        "wall_ms": 382
      }
    }
  ],
  "digest": "sha256:af367511893625c789114c4fb87f5ac67173861771db08c840529fd3067b1ec5",
  "verified": true
}
```

The `argmax` reduction kernel executed on the physical Radeon 8060S GPU in 382 ms and achieved bit-exact numerical parity against the CPU reference oracle (`argmax_exact: true`).

---

## 5. Frozen Workload Specification (#11576)

The evaluation matrix is defined under the frozen parameters of #11576:
- **Model:** `Qwen/Qwen3.8-27B-Instruct`
- **Quantization:** `Q4_K_M` (17.1 GB weights footprint on disk/memory)
- **Context Length:** 32,768 tokens
- **Output Length:** 64 tokens per subagent
- **Concurrency Grid:** $B \in \{1, 2, 4, 8\}$
- **Evaluation Scenarios:**
  1. `cold`: Process-cold prompt ingestion, no prior cache.
  2. `warm_same_prefix`: Identical prefix prefill already resident in KV cache.
  3. `shared_prefix_forked`: Shared system instructions + prompt prefix shared across $B$ subagents with distinct query leaves.
- **Evaluation Arms:**
  - Arm A: `fak-native` with cache disabled (`cache=off`).
  - Arm B: `fak-native` with context MMU shared prefix fork (`cache=on`).
  - Arm C: Pinned local reference comparator from #12098 (matched llama.cpp Vulkan build).

---

## 6. Dependency Audit & Blocking Analysis

Execution of the full comparative matrix requires the completion of several interdependent leaves:

| Dependency Issue | Title | Current Status | Blocker Description |
|---|---|:---:|---|
| **#12097** | `feat(qwen38campaign): attach physical fanout to the real fak-native product runner` | **OPEN** | Physical mode in `internal/qwen38campaign/subagent_fanout.go` returns `physical execution is unavailable; this harness models GPU metrics, use --simulated=true`. Real runner dispatch interface is not yet attached. |
| **#12098** | `bench(qwen38): run same-token fak-native vs pinned llama.cpp A/B on local Strix Halo` | **OPEN** | Local reference binary (pinned `llama-server` / `llama-cli` Vulkan build) is not installed on `strix1`. |
| **#12096** | `bench(modelbench): emit live canonical Qwen3.8 Vulkan receipts from raw-decode` | **OPEN** | Canonical physical receipt pipeline is awaiting completion. |
| **#12034** | `source/binary execution binding` | **OPEN** | Cryptographic source/binary binding validation gate in review. |

Because dependencies #12097 and #12098 remain open, generating synthetic numbers or reusing simulated receipts would violate the repository's anti-conflation and falsification rules.

---

## 7. Comparative Performance Matrix

Under the frozen workload of #11576 on appliance `strix1`:

| Concurrency (B) | Scenario | Arm A: fak-native (Cache Off) | Arm B: fak-native (Cache On / Fork) | Arm C: Pinned Local Reference (#12098) | Cell Verdict | Reason / Blocker |
|:---:|---|:---:|:---:|:---:|:---:|---|
| **1** | `cold` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **1** | `warm_same_prefix` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **1** | `shared_prefix_forked` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **2** | `cold` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **2** | `warm_same_prefix` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **2** | `shared_prefix_forked` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **4** | `cold` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **4** | `warm_same_prefix` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **4** | `shared_prefix_forked` | `PENDING` | `PENDING` | `NOT_COMPARABLE` | `NOT_COMPARABLE` | Pending physical runner (#12097) & reference binary (#12098) |
| **8** | `cold` | `REFUSED` | `REFUSED` | `NOT_COMPARABLE` | `CAPACITY_REFUSAL` | 30.2 GB held by PID 149670; 8×32k context exceeds remaining 24.7 GiB GTT |
| **8** | `warm_same_prefix` | `REFUSED` | `REFUSED` | `NOT_COMPARABLE` | `CAPACITY_REFUSAL` | Requires service handover maintenance window |
| **8** | `shared_prefix_forked` | `REFUSED` | `REFUSED` | `NOT_COMPARABLE` | `CAPACITY_REFUSAL` | Requires service handover maintenance window |

---

## 8. Memory Capacity & Allocation Analysis

On a 64 GB Strix Halo system:
- **Total Physical RAM:** 62.43 GiB
- **Max Allocatable GTT (`ttm.pages_limit`):** 54.94 GiB
- **Active Service Reservation (PID 149670):** ~30.2 GiB
- **Available Headroom:** ~24.7 GiB

### Qwen3.8-27B Footprint per Subagent Concurrency

For a 27B model at 32k context with FP16 KV cache:
- **Model Weights (Q4_K_M):** ~17.1 GiB resident
- **KV Cache per 32k stream:** ~4.0 GiB ($2 \times 32 \times 32768 \times 64 \times 2$ bytes / GQA factor)
- At $B=1$: 17.1 GiB weights + 4.0 GiB KV = 21.1 GiB (fits within 24.7 GiB headroom).
- At $B=2$: 17.1 GiB weights + 8.0 GiB KV = 25.1 GiB (at boundary of headroom).
- At $B=4$: 17.1 GiB weights + 16.0 GiB KV = 33.1 GiB (exceeds headroom; requires shared prefix or exclusive appliance access).
- At $B=8$: 17.1 GiB weights + 32.0 GiB KV = 49.1 GiB (requires stopping active service PID 149670).

Without terminating the active service, admitting an uncoordinated $B=8$ full-context fanout would trigger Linux OOM / TTM thrashing. Emitting an explicit `CAPACITY_REFUSAL` preserves appliance stability.

---

## 9. Next Checkable Step

To complete the physical measurement and transition cells from `NOT_COMPARABLE` to `MEASURED`:

1. **Prerequisite Landing:**
   - Land #12097 to wire the real physical runner into `fak bench subagent`.
   - Land #12098 to compile and verify the pinned llama.cpp comparator on `strix1`.
2. **First Checkable Step:**
   Execute the physical $B=1$ control run under exclusive appliance lock:
   ```bash
   ssh strix1 "flock -w 30 -x /tmp/fak-gpu.lease /usr/local/bin/fak bench subagent --scenario=cold --concurrency=1 --simulated=false --json"
   ssh strix1 "flock -w 30 -x /tmp/fak-gpu.lease /usr/local/bin/fak bench subagent --scenario=shared_prefix_forked --concurrency=1 --simulated=false --json"
   ```
3. **Verification Criterion:**
   Prove that both receipts bind identical source revision, artifact SHA-256, and token IDs, and attest observed physical prefix reuse before admitting concurrency $B > 1$.

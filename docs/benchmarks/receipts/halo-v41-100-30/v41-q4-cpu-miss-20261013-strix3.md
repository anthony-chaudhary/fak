# DS4.1 P7 — CPU-miss compute with quality gate: strix3 hardware receipt

- **Ticket:** anthony-chaudhary/fak-private#3343 (P7 next; port study, no production-envelope claim)
- **Date:** 2026-10-13
- **Provenance:** `[HW-WITNESSED]` for both measurements below (physical AMD Strix Halo silicon, RADV Vulkan device, measured in-process).
- **Source revision:** `ff605efadbd59d280fa42d7a6b20670973ed2cbd` (public `fak`, with the two ticket-3343 model files overlaid).
- **Artifact note:** the working spine runs **DeepSeek-V4.1 Flash Q2_K** (gate/up Q2_K, down Q3_K). The `q4` in this filename is the ticket's placeholder name; the measured artifact is the Q2_K slate.

## Hardware under test

| Field | Value |
| --- | --- |
| Node | `strix3` (`strix-halo-fak`) |
| SoC | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (16 Zen 5 cores / 32 threads) |
| Memory | 124 GiB LPDDR5X, **unified** (CPU + iGPU share one pool) |
| GPU / driver | `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, Mesa 26.2.2-arch1.1, Vulkan API 1.4.354, deviceType `PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` |
| Kernel | Linux 7.2.4-arch1-2 |
| Engine | `fak-native` Vulkan compute (`internal/compute`, `-tags vulkan`), static shim `libfakvulkan.a` |
| libfakvulkan.a sha256 | `7931b0bcd4458b2dddf5d2aba3b4423cce7be57ec83973d08bfecb596708f901` |
| SPIR-V bundle sha256 | `79278d6ea4c3d9913bb37a5cbf0331c57eb440998d4ae09fb82f3d0c78858f60` |
| `expert_cpu_miss.go` sha256 | `2fba7a10eb9847143a8b86bcdb377ad5200509b01373bea882a29fbcfbf9fefd` |
| `expert_cpu_miss_test.go` sha256 | `67bac47c519b4aee023f0c727caf6276aece25c32fa3f3bace8c06fe4d6d4bbb` |
| `expert_cpu_miss_hw_test.go` sha256 | `150d83d319355a6919b9a4fb32b4e75a5d6ef75a2261cdfaaa6418c69ec9b02f` |

Build/run (on `strix3`):

```sh
glslc -O --target-env=vulkan1.2 -fshader-stage=comp internal/compute/shaders/*.comp ...
c++ -O3 -std=c++17 -fPIC -c internal/compute/vulkan_shim.cpp && ar rcs libfakvulkan.a vulkan_shim.o
CGO_ENABLED=1 CGO_LDFLAGS="-L$build" FAK_VULKAN_SPIRV="$build/spirv" \
FAK_VULKAN_REQUIRE_DEVICE=1 FAK_UMA_HW_WITNESS=1 \
  go test -tags vulkan ./internal/model -run 'TestUMA_3343' -count=1 -v
```

`FAK_VULKAN_REQUIRE_DEVICE=1` fails rather than skips when no device registers; `FAK_UMA_HW_WITNESS=1` gates the two UMA witnesses. Three independent runs PASS.

## DoD item 1 — UMA bandwidth-sharing measurement (recorded)

The Halo is a **single-pool (UMA)** APU, so the host CPU GEMV and the device DMA draw from the same
LPDDR5X. The bandwidth partition solver in `internal/model/moe_bandwidth_partition.go` assumes two
*independent* channels; this witnesses whether that assumption holds. Measured over a 64 MiB f32
working set (solo = one engine at a time; concurrent = CPU streaming copy ‖ GPU H2D `Upload`).

| Run | Solo CPU GB/s | Solo GPU GB/s | Concur CPU GB/s | Concur GPU GB/s | CPU ratio | GPU ratio | Shared GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.50 | 3.48 | 25.48 | 2.22 | 0.864 | 0.639 | 4.20 |
| 2 | 28.76 | 3.58 | 24.63 | 2.37 | 0.856 | 0.663 | 4.38 |
| 3 | 25.97 | 4.08 | 19.87 | 2.63 | 0.765 | 0.645 | 4.96 |
| 4 | 30.71 | 4.54 | 22.54 | 2.89 | 0.734 | 0.637 | 5.37 |
| 5 | 31.54 | 4.51 | 23.45 | 2.90 | 0.743 | 0.642 | 5.39 |

```json
{"schema":"fak.uma-bandwidth-sharing/v1","samples":5,"solo_cpu_gbps":29.496771980554744,"solo_gpu_gbps":3.4779980340712187,"concurrent_cpu_gbps":25.48234014696546,"concurrent_gpu_gbps":2.2227940882705783,"cpu_contention_ratio":0.8639026725963189,"gpu_contention_ratio":0.6391015942204705,"shared_bandwidth_gbps":4.1971375006427385,"measurement":"hardware","notes":"5 samples; 3 concurrent"}
```

**Finding (measured, not inferred): the two engines are NOT independent on this box.** Every run
shows both contention ratios strictly below 1.0 (CPU ≈ 0.77–0.86, GPU ≈ 0.64–0.66): running CPU
streaming and GPU DMA together slows *both*. A q* partition solved from solo rates therefore
overstates the throughput available to the concurrent arm. The receipt is labelled
`measurement:"hardware"`; the solver's independence assumption is the modelling caveat, carried
explicitly into the cost estimate below.

## DoD item 3 — NLL/perplexity quality panel

The quality-panel fold is implemented and unit-witnessed: `MissQualityPanel` (`NLLDeltaPerToken`,
`PerplexityDelta`), `QualityPanel.Verdict()`, and `NewMissQualityPanel`. The exact CPU-miss arm's
per-token quality delta is **0 by construction / bit-identical** — it reads the exact routed-expert
bytes and runs the exact host k-quant GEMV the resident CPU path runs — so its plan reports
`QualityLabel() == ExpertMissQualityExact` and its panel is admissible as exact evidence. A lossy
arm (Strata routing-bias 2.38 → 2.87, or the #1299 shadow fallback) must carry
`ExpertMissQualityNonExact`; `Verdict()` requires both an exact label AND a measured panel (positive
token counts, finite delta), so an unmeasured or non-exact panel can never be promoted.

**Non-claim:** no end-to-end model perplexity run was captured in this receipt. The panel fold is
witnessed by unit tests; a served-model Δ is a follow-on.

## DoD item 4 — miss-path ms/token (strix3)

Per-expert cost measured at the V4.1 Flash Q2_K slab geometry
(`gate=up=2304×5120 @ 84/256 B/weight`, `down=5120×2304 @ 110/256 B/weight` → **slab = 12,810,240 B**
= 12.217 MiB), fastest of 10 reps. **Caveat up front:** the CPU arm is a raw streaming byte-touch
(`copy`) used as a *proxy* for the exact host read the k-quant GEMV performs — not a timed host
GEMV, which would be strictly more. It remains the exact (non-approximate) byte path, so its plan
labels `exact`. Byte-count convention: `copy` physically moves `2·n` bytes but the receipt counts
`n` bytes moved per copy (one read + one write); the absolute CPU GB/s is therefore a half-count
convention, not an effective-bandwidth claim. The contention **ratios** below are unaffected by the
convention.

| Run | CPU ms/expert | CPU GB/s | GPU ms/expert (H2D) | GPU GB/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.220 | 58.24 | 31.31 | 0.409 |
| 2 | 0.218 | 58.90 | 18.73 | 0.684 |

```json
{"schema":"fak.v41-cpu-miss-misspath/v1","slab_bytes":12810240,"cpu_per_expert_ms":0.219973,"gpu_per_expert_ms":31.310206,"misses_per_token":94,"per_token_gpu_serial_ms":2943.159364,"per_token_overlap_lower_bound_ms":20.677462,"cpu_gbps":58.235510721770396,"gpu_gbps":0.40913943523718754,"measurement":"hardware","notes":"cpu/gpu_per_expert_ms and *_gbps are physical; per_token_* are derived from K=94"}
```

Derived for the slice-1 baseline miss rate (`hit 0.61` → **K = 94 misses/token** of 240 activations):

- **GPU-serial only (today): ≈ 1.76–2.94 s/token** of miss transfer — the serial miss wait the P7 port removes.
- **CPU arm cost alone: 94 × 0.219 ms ≈ 20.7 ms/token** — an order of magnitude below the serial H2D arm.
- **Overlap lower bound: ≈ 20.4–20.7 ms/token** (exact CPU compute on the missing experts, overlapped with the residual stream). Valid only under the independence assumption that the ratio table above **falsifies**; the shared-bandwidth receipt (≈ 4.2–5.0 GB/s aggregate under contention) is the correction term.

These are **lower bounds**, not a serve claim: the 0.22 ms/expert CPU figure is a raw byte-stream
touch, and the realistic exact host k-quant GEMV per expert is higher. The measured conclusion is
directional and defensible: the CPU arm is cheap enough that moving misses onto it dominates
waiting for the serial H2D transfer on this box.

## DoD item 2 — CPU-miss path behind a flag, with hit/miss split

- Flag: `FAK_EXPERT_CPU_MISS` (`ExpertCPUMissEnv`). Unset/empty/`off` is the **exact default**;
  `exact` enables the exact host CPU-miss path. An unparseable value is refused
  (`ExpertCPUMissPolicyFromEnv` returns an error), never silently ignored.
- `ExpertCPUMissPolicy.PlanExpertCPUMiss` partitions a layer's misses with the existing
  `SolveBandwidthPartition` solver, then caps the CPU arm at `DefaultExpertCPUMissPerLayer = 3`
  (MoE4All `cpu_miss.rs:20` bound), returning the lowest-Recency overflow to the stream.
- Hit/miss split and ms/token are reconciled by `ExpertCPUMissLedger`:
  `Hits + CPUComputed + Streamed + Refused == Lookups`, with
  `MissPathMillisPerToken` known only after a positive token count and ≥1 lookup (never a fabricated 0).

## Quality gate (DoD item 3 labelling)

The path above computes the **exact** routed-expert bytes on the host CPU and reports
`ExpertMissQualityExact`. Any lossy arm — Strata-DSV4.1 routing-bias (26 tok/s at perplexity
2.38 → 2.87), or the #1299 shadow fallback — MUST carry `ExpertMissQualityNonExact`, stays opt-in via
`ExpertCPUMissPolicy.NonExact`, and the `QualityPanel.Verdict()` admissibility rule refuses to treat a
non-exact panel as exact evidence regardless of how small its NLL/perplexity delta is. **No lossy
default ships.**

## Witness

```
Go test (public fak, this worktree):
  go test ./internal/model -run TestExpertCheckpoint -count=1   -> ok  (ticket Witness command)
  go test ./internal/model -run 'ExpertCPUMiss|UMABandwidthSharing|MissQuality|QualityPanel' -count=1 -> ok
Strix3 hardware (physical RADV Vulkan, FAK_UMA_HW_WITNESS=1):
  TestUMA_3343_BandwidthSharingHardware          -> PASS
  TestUMA_3343_MissPathMicrobenchHardware        -> PASS
```

## Scope / non-claims

- **Port study only**, per the ticket's completion standard. No production envelope, no tok/s
  claim: the model was not served end-to-end in this receipt.
- No llama.cpp path was used; both measurements executed inside `fak-native` (Vulkan HAL + host).
- Per-token figures are derived from the physical per-expert measurements at the published slice-1
  miss rate; the end-to-end `fak serve` V4.1 Flash Q2_K tok/s arm remains a follow-on.
- `FAK_VULKAN_REQUIRE_DEVICE=1` makes a missing device a failure, not a skip, so this receipt cannot
  be produced by a mock backend. (The two UMA witnesses themselves gate on `FAK_UMA_HW_WITNESS=1`
  and skip when no Vulkan device is registered; the separate `TestVulkanArgmaxExact` primitive
  witness in `docs/compute/vulkan-linux.md` is the test that uses `FAK_VULKAN_REQUIRE_DEVICE=1` to
  fail instead of skip. Both UMA tests require a real physical device to reach their assertions.)

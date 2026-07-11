---
title: "Decode occupancy/traffic witness landed; A100 ncu measurement dispatched (bridge-blocked); 4 of 8 shortlist borrows analytically confirmed"
description: >
  The KernelWiki study's Blackwell decode-kernel shortlist (Cluster G, 8 candidates) was left
  unfiled behind one gate: "land a decode-shaped occupancy/traffic witness in internal/compute
  first, then promote the ones that show a real gap." That witness is now landed
  (internal/compute/decode_occupancy.go, 7 tests green, on origin/main) together with its A100
  Nsight-Compute corroboration harness (tools/dgx_decode_occupancy_ncu.sh). The witness's
  file/no-file verdict is EXACT (grid-block counts vs SM counts, operand-byte counts) and needs
  no device: 4 of the 8 candidates show a real modeled decode gap, 1 shows none (grid-bound, not
  per-SM-bound), and 3 are Blackwell/Hopper-only mechanisms this A100 witness cannot measure. The
  device ncu run to corroborate the achieved-occupancy / DRAM-% numbers was dispatched but the
  Slack control-bridge readback was transport-wedged this session (context deadline exceeded on
  every conversations.history tail); no device numbers were fabricated.
metadata:
  type: project
---

# Decode occupancy/traffic witness + A100 measurement (2026-07-11)

## Context

Follow-on to `docs/notes/2026-07-10-kernelwiki-study.md` — specifically its **"Noted, not filed —
Blackwell decode-kernel shortlist (Cluster G, 8)"** section, which deliberately filed nothing and
named one prerequisite for the whole shortlist:

> **File-after-measurement**: land a decode-shaped occupancy/traffic witness in `internal/compute`
> first, then promote the ones that show a real gap.

This pass lands that witness and its device corroboration harness, and records the verdict.

## What landed (on origin/main)

| Artifact | Commit | What it is |
|---|---|---|
| `internal/compute/decode_occupancy.go` | `84a8034ac` | the decode-shaped occupancy + HBM-traffic witness (exact counts, no timer; mirrors `fusion_traffic.go`/`prefill.go` discipline) |
| `internal/compute/decode_occupancy_test.go` | `84a8034ac` | 7 tests: arch table, underfill, register-independence, wave-quant tail, traffic, gap verdicts |
| `tools/dgx_decode_occupancy_ncu.sh` | `13c95e268` | the Nsight-Compute (`ncu`) run that corroborates the witness on a real A100 |

The witness adds the first NVIDIA per-compute-capability occupancy table in the package (`NVArch`,
sm_80/sm_90/sm_100; `rocm_arch.go` was AMD-only), models the three real decode launches
(`k_flash_attention<<<nH,128,(hd+128)*4>>>`, `k_q8_gemm<<<dim3(out,1),256>>>`,
`k_awq_gemv<<<out,256>>>`), and reports per-candidate file/no-file verdicts via `DecodeGapReport`.

## The decisive, exact fact (why the verdict needs no device)

At a decode step `k_flash_attention` launches **one block per query head** — the whole grid is `nH`
blocks. SmolLM2-135M has `nH=9`; an A100 has **108 SMs**. So **99 SMs sit idle every decode step**,
and this is *arithmetic* (blocks vs SMs), not a measurement — and it is **independent of the
compiler's register count** (`TestUnderfillRegisterIndependent`). Decode attention is also
memory-bound at ~**0.5 FLOP/byte** (exact operand count), dominated (~99.9%) by the use-once K/V
stream. Those two exact facts decide the shortlist.

## Verdict — 4 file, 1 no-gap, 3 defer

`DecodeGapReport(sm_80, 108 SMs, SmolLM2-135M)`:

| # | Candidate | Seam | A100-measurable | Modeled gap | Verdict |
|---|---|---|---|---|---|
| 1 | persistent-kernel-work-stealing-tail-fix | `cuda_kernels.cu:1084` / flash grid=nH | yes | **IdleSMs=99** | **FILE** |
| 2 | l1-cache-hints-decode (`__ldcs`) | `cuda_kernels.cu:726,742` | yes | **memory-bound, KV-stream 99.9%** | **FILE** |
| 3 | clc-decode-tile-scheduling | `cuda_kernels.cu:455` | yes | **~21% wave-quant tail** on the GEMV | **FILE** |
| 4 | moe-launch-fusion-ladder | `fusion_traffic.go:141` | yes | **per-expert grid underfill** | **FILE** |
| 5 | tmem-accumulator-migration | `cuda_kernels.cu:719` | yes | none — grid-bound, not per-SM-bound | **no-gap (do not file)** |
| 6 | clc-try-cancel-speculative | `discard_admit.go:44` | no (Hopper+ CLC / host-admit) | — | defer |
| 7 | nvfp4-two-level-block-scale | `gguf_dequant.go:422` | no (sm_100-only format) | — | defer |
| 8 | pdl-moe-kernel-overlap | `cuda_kernels.cu:454` | no (sm_90+ PDL, inter-kernel) | — | defer |

The **tmem** result is the witness earning its keep: migrating the `acc[8]` register array to
Blackwell tensor memory raises *per-SM* occupancy, but decode is *grid-bound* (only nH≪108 blocks
exist), so it buys no device-level speedup — a speculative ticket the measurement gate correctly
stops.

## A100 measurement — dispatched, bridge-blocked this session

The `ncu` corroboration (`tools/dgx_decode_occupancy_ncu.sh`) collects, per kernel: achieved
occupancy (`sm__warps_active…pct`), registers/thread (`launch__registers_per_thread`), grid size
(`launch__grid_size`), SM throughput, and DRAM % (`gpu__dram_throughput…pct`) → `ncu.csv`. It was
dispatched over the Slack control-bridge to a live A100 session, but **every readback wedged**
(`READBACK_WEDGED: context deadline exceeded` on `conversations.history`, and `sentinel_missing` on
busy sessions) — a transport-layer failure from this host, not a box or kernel failure. **No device
numbers were run, estimated, or fabricated.** This is the honest gate: the A100 achieved-occupancy /
DRAM-% corroboration is *not yet* obtained.

**To obtain it when the bridge is healthy (one command):**

```sh
dgxbridge -probe bg tools/dgx_decode_occupancy_ncu.sh decode-occ
dgxbridge -probe -timeout 2h wait decode-occ
dgxbridge -probe pull /tmp/fakncu/ncu.csv ./ncu.csv
```

Then reconcile `ncu.csv` against the witness: `launch__grid_size` must read 9 for flash (matching
`GridBlocks`); `sm__throughput` low + `gpu__dram_throughput` high confirms the underfill +
memory-bound verdicts; `launch__registers_per_thread` plugs into `FlashDecodeLaunch(g, regs)` to
confirm the per-SM limiter. Introducing `ncu` on the boxes is itself the unbuilt ncu-evidence arm
(#4101).

## Promotion status

The four **FILE** candidates have an *exact analytic* gap (block/byte counts), so the file/no-file
decision is made; the pending item is the device-% corroboration above. Filing the four tickets
under the shortlist's umbrella (KernelWiki epic in `2026-07-10-kernelwiki-study.md`) is the residual
step — held for the go-ahead on whether to file now (analytic-witnessed, device-pending) or after the
ncu numbers land.

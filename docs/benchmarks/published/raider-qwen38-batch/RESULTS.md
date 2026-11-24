---
title: "Campaign record: Raider Qwen3.8-27B llama.cpp concurrency × long-context sweep (internal RTX 5090 only)"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak raider-qwen38-batch (2026-10-27): Raider Qwen3.8-27B llama.cpp concurrency × long-context sweep (internal RTX 5090 only)."
---

# RESULTS — Raider Qwen3.8-27B llama.cpp concurrency × long-context sweep (internal RTX 5090 only)

**[HW-WITNESSED]** — physical raider-fak, boot `d1026348-79cd-485e-a9b5-c8f5da69e578`,
internal GPU UUID `GPU-2a2a837f-c82d-0f22-4538-e3a063903380` (RTX 5090 Laptop, 24463 MiB).
eGPUs physically disconnected for this pass; `--phys-set internal` on every op.

Engine: OSS llama.cpp b11870 CUDA `llama-server`
(`sha256 8453c97a0243d3feceea0340110b846c9516cd82f34ef16f553d668476080310`,
mtime 2026-10-25T19:09Z, unchanged through the window).
Model: `/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf`
(`sha256 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`).
Client: pinned loadgen `d449a5a061f002a394303bfa479824e7e302134b423fb9556674ec17ac79c694`.
Workload: one shared deterministic prompt per cell (warm-prefix serving regime),
decode-dominated, `max-tokens 64`, prompt target `ctx − 512`.
Server argv per cell: `-m MODEL -a Qwen3.8-27B -c <ctx×np> -ngl 99 -np <np> -b 512 -ub 128 -cb --port 18080`
(`-c` is the TOTAL context budget in this build; per-slot ctx = `ctx×np ÷ np = ctx`).

Lease trail (class `owned-rtx5090-raider`, holder `pi.hw-bench-campaign-owned-nvidia`):
stage under maintain `hwl-fc73e5c3e6348ae2` (epoch 64, released); all probes/cells under
measure `hwl-af4d3d76e0153638` (epoch 65, released after cells); service restore under
maintain `hwl-cf7588664be90449` (epoch 66, released). Final census: `leases=0`,
service active, boot unchanged.

Campaign window this pass: 2026-10-28T09:55Z–10:10Z (earlier stage + first probe attempt
08:14Z–08:17Z, see Anomalies).

## Throughput vs concurrency (per cell, all verdicts `ok`, 0 errors, cleanup confirmed)

### Cell A — ctx 2048, np 16, rp 32 (`raw/batch-A-ctx2048.json`)

| C | tps | p50 ms | p95 ms | errors | completion_tokens | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 22.20 | 2839 | 2902 | 0 | 2048 | 92.24 |
| 2 | 36.24 | 3420 | 4904 | 0 | 2048 | 56.51 |
| 4 | 51.04 | 4696 | 6445 | 0 | 2048 | 40.13 |
| 8 | 103.81 | 4807 | 5753 | 0 | 2048 | 19.73 |
| 16 | 135.81 | 6986 | 7809 | 0 | 2048 | 15.08 |
| 32 | **144.60** | 8103 | 14162 | 0 | 2048 | 14.16 |

### Cell B — ctx 4096, np 8, rp 16 (`raw/batch-B-ctx4096.json`)

| C | tps | p50 ms | p95 ms | errors | completion_tokens | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20.89 | 2854 | 6252 | 0 | 1024 | 49.03 |
| 4 | 33.41 | 4898 | 16622 | 0 | 1024 | 30.65 |
| 8 | 70.60 | 5557 | 10869 | 0 | 1024 | 14.50 |
| 16 | **116.61** | 4666 | 8780 | 0 | 1024 | 8.78 |

### Cell C — ctx 8192, np 4, rp 16 (`raw/batch-C-ctx8192.json`)

| C | tps | p50 ms | p95 ms | errors | completion_tokens | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 19.16 | 2894 | 10027 | 0 | 1024 | 53.43 |
| 4 | 23.59 | 5021 | 29489 | 0 | 1024 | 43.41 |
| 16 | **51.95** | 10279 | 19709 | 0 | 1024 | 19.71 |

### Cell D — ctx 16384, np 2, rp 4 (`raw/batch-D-ctx16384.json`)

| C | tps | p50 ms | p95 ms | errors | completion_tokens | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.64 | 2911 | 17823 | 0 | 256 | 26.57 |
| 4 | **10.95** | 20159 | 23380 | 0 | 256 | 23.38 |

### Cell E — ctx 32768, np 1, rp 4 (`raw/batch-E-ctx32768.json`)

| C | tps | p50 ms | p95 ms | errors | completion_tokens | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **5.61** | 3048 | 36505 | 0 | 256 | 45.64 |

Per-cell shape: the first request of each level pays the cold-prefix prefill
(p95 spike), later requests hit the warm prefix cache (p50 stays near the
single-stream decode latency). At saturation (A@C32, B/C@C16) p95 degrades
sharply while aggregate tps keeps climbing.

## Context curve at matched concurrency (C=1, all cells)

| ctx | np | tps @ C=1 | peak tps (cell max C) |
| --- | --- | --- | --- |
| 2048 | 16 | 22.20 | 144.60 @ C=32 |
| 4096 | 8 | 20.89 | 116.61 @ C=16 |
| 8192 | 4 | 19.16 | 51.95 @ C=16 |
| 16384 | 2 | 9.64 | 10.95 @ C=4 |
| 32768 | 1 | 5.61 | 5.61 @ C=1 |

Single-stream throughput holds ~20–22 tps up to ctx 8192 (warm-prefix decode is
context-insensitive there), then falls off as KV pressure grows: 9.6 tps at
16384, 5.6 tps at 32768. Aggregate peak is 144.6 tps (ctx 2048, C=32).

## KV-fit boundaries (accepted np per ctx + evidence)

Accepted np: **2048→16, 4096→8, 8192→4, 16384→2, 32768→1** — i.e.
`np(ctx) = floor(32768 / ctx)`, a constant ~32–33K-token total KV budget.

Evidence (measured directly under the measure lease, boot d1026348, before cells):
- Baseline: weights + 4096 KV tokens + 1 slot ≈ 16.2 GiB VRAM after first request.
- Scaling: 32768 total KV tokens (np16 × 2048/slot) instead of 4096 added
  ≈ 4.1 GiB ⇒ ≈ 0.144 MiB/token (incl. extra per-slot compute buffers).
- Headroom: 24463 MiB card − 16.2 GiB baseline ≈ 7.8 GiB ⇒ ≈ 54K tokens, but the
  ≥1 GiB headroom preference plus buffer scaling caps the usable budget at
  ~32–33K tokens.
- Every probe at the accepted np returned verdict `ok` with `mem_mib_max`
  ≤ 20330 MiB (≤ 83 % of VRAM): 20328 / 19084 / 18462 / 18152 / 18006 MiB for
  ctx 2048/4096/8192/16384/32768. Halving np beyond the accepted value was
  never required.
- Note: this build does not emit the legacy `kv self … MB` / `will fit` log
  lines, so probe `kv.fits_budget` is `null`; acceptance followed the
  null-with-verdict-ok rule plus the direct VRAM measurements above.

## Telemetry aggregates (per receipt, target GPU only)

power_w [min,max,mean] · temp_c_max · mem_mib_max · sm_mhz_mean:

| receipt | power W | temp °C | mem MiB | SM MHz |
| --- | --- | --- | --- | --- |
| probe-2048-np16-rerun | 39.9 / 94.4 / 85.0 | 42 | 20328 | 833 |
| probe-4096-np8 | 36.6 / 94.9 / 85.4 | 47 | 19084 | 879 |
| probe-8192-np4 | 39.6 / 95.1 / 87.8 | 49 | 18462 | 921 |
| probe-16384-np2 | 34.9 / 97.6 / 89.9 | 54 | 18152 | 971 |
| probe-32768-np1 | 40.5 / 94.9 / 92.0 | 56 | 18006 | 1058 |
| batch-A-ctx2048 | 19.3 / 98.2 / 89.4 | 63 | 20330 | 775 |
| batch-B-ctx4096 | 22.4 / 96.5 / 88.0 | 62 | 19086 | 824 |
| batch-C-ctx8192 | 39.8 / 96.9 / 91.7 | 61 | 18462 | 828 |
| batch-D-ctx16384 | 40.6 / 94.8 / 91.9 | 59 | 18152 | 966 |
| batch-E-ctx32768 | 39.9 / 95.1 / 91.7 | 59 | 18006 | 1000 |

No thermal throttling observed (max 63 °C; SM clocks 775–1058 MHz mean, well
inside boost range). Power plateaus near 95–98 W under sustained decode.

## Anomalies

1. **Probe `artifact_mismatch` at 08:16Z (`raw/probe-2048-np16.json`) — root
   cause identified, not transient.** The coordinator's initial read was a
   transient I/O/timing blip; the evidence contradicts that. The committed
   tool pinned llama SHA `8453c97ac3…` (a transcription error), while the
   stable on-node binary hashes to `8453c97a02…` (mtime 2026-10-25T19:09Z,
   unchanged through the window — re-verified by `sha256sum` on the node
   2026-10-28T~08:40Z). With the stale committed pin, every probe/cell op was
   deterministically refused `artifact_mismatch` at the digest gate (the
   receipt's empty `llama_sha256` reflects the refusal path, not a read
   failure). Resolved by the re-pin + b11870 CLI argv fix landed in
   `b6c8d9768adb30c1edc8b545987a35cbf3e75f21`; the 09:55Z rerun probe at identical parameters
   returned verdict `ok`. Both receipts are retained.
2. **Node reboot inside the campaign window**: boot `d003c8a2…` →
   `d1026348…` between 03:15Z and 08:14Z (witnessed by the coordinator census).
   All campaign ops ran under the single boot `d1026348…`; no op straddled the
   reboot. Pinned digests were re-verified live after the reboot.
3. **Pinned b11870 build CLI drift**: `-alias`/`-port` are rejected by this
   build (`-a`/`--port`), and `-c` is the *total* context budget divided evenly
   across slots (witnessed: `-c 2048 -np 16` ⇒ `n_ctx_slot = 256`, long prompts
   refused `exceed_context_size_error`; `-c 32768 -np 16` ⇒ `n_ctx_slot = 2048`).
   Payload argv corrected accordingly (`-a`, `--port`, `-c ctx×np`); without
   this every op would have been `startup_failed`.
4. **Publish-lane contention** (host-side, not node-side): dualsync admission
   queue held the trunk publish for >30 min mid-campaign; campaign ops were
   therefore executed from the working tree carrying the uncommitted fixes
   (documented dev lane), with `payload_sha256 c274c9d5…` recorded in every ok
   receipt. The fixes are committed in
   `b6c8d9768adb30c1edc8b545987a35cbf3e75f21`, whose source reproduces the
   exact script bytes those receipts ran.

## Comparability

- **Warm-prefix shared-prompt regime**, not cold prefill: one deterministic
  prompt per cell, repeated across all requests; the first request pays full
  prefill, the rest hit the prefix cache. Never mix with cold-prefill numbers.
- **eGPUs physically absent**: internal-only configuration. The
  `internal-a` / `internal-b` / `internal-ab` cells are **OPEN** pending
  operator reconnect of the two RTX 5070 Ti eGPUs; no claim is made for them.
- **Different workload/client/concurrency axes** than the 2026-10-27 h2h
  numbers (that baseline: cold serial 128-output, client c9870f151d, different
  engine pair). Strict ratios only within this campaign's matched cells
  (same ctx, np, prompt, client, boot).
- **Loadgen re-pin provenance**: the prior pin `34599f91…` proved
  irreproducible from its documented inputs on 2026-10-28 (toolchain-state
  delta); the campaign uses the re-pinned verified build `d449a5a0…` of the
  same source revision `736dc9bfba` (PLAN.md note; commit 43b726912c).
- Prompt generator undershoots its token target slightly
  (char-length heuristic): actual prompt tokens 1348 / 3086 / 6554 / 13499 /
  27394 for the 1536 / 3584 / 7680 / 15872 / 32256 targets — deterministic and
  identical across probe and cell for each ctx (matching `prompt_sha256`).

## Open items

- `internal-a` / `internal-b` / `internal-ab` cells: blocked on physical eGPU
  reconnect (operator decision), then re-run under fresh measure leases.
- Probe `kv` fields are null on this build (legacy log format absent); a
  follow-up could parse the new build's allocation lines if exact
  `kv_self_mb` accounting is wanted.

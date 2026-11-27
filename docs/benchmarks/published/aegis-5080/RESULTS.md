---
title: "Campaign record: RTX 5080 host RTX 5080 Qwen3.8-27B llama.cpp overnight sweep"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-5080 (2026-10-30): RTX 5080 host RTX 5080 Qwen3.8-27B llama.cpp overnight sweep."
---

# RESULTS — RTX 5080 host RTX 5080 Qwen3.8-27B llama.cpp overnight sweep (2026-10-30)

**[HW-WITNESSED]** — physical `rtx5080-host` (RTX 5080 desktop desktop), boot
`31fc7046-7671-4813-8d33-7856b915dd9d` (unchanged through the window), GPU UUID
`GPU-3e9fda0a-061a-6eb7-26c3-541bf8489090` (RTX 5080, 16303 MiB, sm_120).

## Envelope

| item | value |
| --- | --- |
| engine | OSS llama.cpp b11870 CUDA `llama-server`, sha256 `8453c97a0243d3feceea0340110b846c9516cd82f34ef16f553d668476080310` |
| model | `/var/lib/fak/models/Qwen3.8-27B-UD-Q2_K_XL.gguf` 9828981664 B, sha256 `fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0` |
| client | pinned loadgen sha256 `d449a5a061f002a394303bfa479824e7e302134b423fb9556674ec17ac79c694` |
| driver / power | NVIDIA 615.78.08; **native 360 W only** (no caps; readback 360.0 W in [340,380] band on every op) |
| host | Arch Linux 7.2.11-arch1-1, 30 GB RAM, 1.9 TB NVMe, 1 GbE enp5s0 |
| producer | `fak-run fak-raider-bench` at committed HEAD (host-row commits 937fdec2c8 + 9d0d595250); payload_sha256 `e690fed59605334407ef1e3923afc6634805bcc345326c3bf9aea08fe136250a` on all ok ops |
| lease trail | measure `hwl-ab3dc5d0417c6a5e` (owned-rtx5080, epoch 125, holder `pi.rtx5080-host-5080-night-bench`) for all probe/cell ops; maintain `hwl-43b9e563350c6f4e` (epoch 124) for the service-stop window; both released at close |
| workload | warm-prefix shared deterministic prompt per cell, decode-dominated, max-tokens 64, prompt target ctx−512; server argv `-m MODEL -a Qwen3.8-27B -c <ctx×np> -ngl 99 -np <np> -b 512 -ub 128 -cb --port 18080`; 300 s inter-cell cooldown |

## Headline numbers

### Qwen decode tok/s per cell (all verdicts `ok`, 0 errors, cleanup confirmed)

| cell | ctx | np | C | tps @ C | p50 ms @ C | peak tps (cell max C) |
| --- | --- | --- | --- | --- | --- | --- |
| A | 2048 | 16 | 1 | **63.91** | 977 | 279.19 @ C32 |
| A | 2048 | 16 | 2 | 103.89 | 1179 | |
| A | 2048 | 16 | 4 | 146.52 | 1596 | |
| A | 2048 | 16 | 8 | 176.38 | 2435 | |
| A | 2048 | 16 | 16 | 152.17 | 4004 | |
| A | 2048 | 16 | 32 | **279.19** | 3612 | |
| B | 4096 | 8 | 1 | **58.73** | 985 | 192.97 @ C16 |
| B | 4096 | 8 | 4 | 83.74 | 1732 | |
| B | 4096 | 8 | 8 | 133.55 | 2961 | |
| B | 4096 | 8 | 16 | **192.97** | 2911 | |
| C | 8192 | 4 | 1 | **52.30** | 1003 | 141.95 @ C16 |
| C | 8192 | 4 | 4 | 55.58 | 1796 | |
| C | 8192 | 4 | 16 | **141.95** | 3937 | |
| D | 16384 | 2 | 1 | **22.17** | 1041 | 24.49 @ C4 |
| D | 16384 | 2 | 4 | **24.49** | 9315 | |
| E | 32768 | 1 | 1 | **12.60** | 1121 | 12.60 @ C1 |

Single-stream (C1) context curve: 63.9 → 58.7 → 52.3 → 22.2 → 12.6 tps at
ctx 2048/4096/8192/16384/32768. Aggregate peak **279.19 tps** (ctx 2048, C32).
Probe cross-check: standalone C1 @ ctx 2048 = 69.67 tps (receipt
`probe-ctx2048-np1.json`, inter-token p50 14.286 ms) — consistent with cell A
once the first request pays cold prefill.

### Tier bandwidth (measured vs estimated, idle box)

| tier | measured | method / label | replaces ESTIMATED fit-spine rate |
| --- | --- | --- | --- |
| NVMe seq-read | **4.0 GB/s** (3.987/4.013/3.997 over 3 cold passes; warm page-cache 38.4 GB/s contrast) | dd bs=1M over the 9.83 GB gguf, drop_caches before each pass, HW-WITNESSED | SSD expert-stream estimate |
| DRAM | **WEAK**: python triad 0.42 GB/s (loop-bound); tmpfs single-stream read proxy 18.9 GiB/s | no C compiler on rtx5080-host (gcc/cc/clang absent) ⇒ documented WEAK fallback | DRAM spill estimate |
| GPU baseline | 16.16 W idle / 360.00 W limit / 180 MHz idle / 33 °C | nvidia-smi, HW-WITNESSED | envelope pin |

Receipt: `receipts/tier-bandwidth.json`.

### DS4.1 Flash first attempt — outcome class: **GATE-NOT-REACHED (two typed load failures)**

Full receipt: [`ds41-attempt-1/ATTEMPT-RECEIPT.md`](ds41-attempt-1/ATTEMPT-RECEIPT.md).

- **Transfer closed.** v2's `FAILED-chunk-fail-shard1-chunk10` @ 08:35:05Z is fully
  explained by a code defect: `count=$((len/MB))` integer division truncated the final
  partial chunk (123316096 B → 117 MiB, 632704 B short) in both the stream and the
  remote reference hash while the local gate hashed the full range. Offline replay of
  **both** recorded digests against today's pinned source reproduces them bit-exactly
  (`receipts/t2-rewitness/v2-replay*.txt`) — no transient source/link anomaly. v3
  (manifest-gated, ceil-count, byte-exact size gates) finished `DONE-ALL-VERIFIED`
  @ 14:10:57Z; all 7 shards whole-file sha256 vs pin; pinned set passes 7/7 `sha256sum -c`.
- **Attempt executed, zero tokens.** Pinned 7/7 set: typed `model-metadata-missing`
  (`array key not found in model: deepseek41.attention.kv_source_layers` — export lacks
  13 `deepseek41.*` KVs b11870 requires). After an out-of-band 13-KV metadata splice from
  a GSQ-lineage reference shard1 (audited clean; original preserved as `.orig-pin`):
  typed `model-load-error` (`check_tensor_dims: tensor 'per_layer_token_embd.weight'
  not found`). Structural parse of both shard1 headers proves the tensor is absent from
  the export lineage itself — the blocker sits between the c4a08554 Q2_K export and
  b11870's deepseek41 loader, not on this host. Both classes re-witnessed deterministically
  (< 1 s each); stop rule never triggered. No ms/token telemetry exists for this envelope;
  next checkable step: corrected export (or qualified alternate engine version), then
  re-enter at step 7 with the same argv/gate/stop rule.

## KV-fit boundaries (accepted np per ctx + evidence)

Accepted np: **2048→16, 4096→8, 8192→4, 16384→2, 32768→1** — same schedule as the
raider precedent (~32K total KV tokens), chosen by probe-measured VRAM, not copied:

- Weights+ctx2048+1slot loaded = 9264 MiB; post-request 9468 MiB (≈204 MiB/slot
  incl. compute buffer at ctx 2048).
- Probe mem_mib_max at accepted np: 13716 / 12472 / 11850 / 11540 / 11394 MiB for
  ctx 2048/4096/8192/16384/32768 — all ≤ 84 % of 16303 MiB, ≥ 2.5 GiB headroom.
- Every cell ran at its probed np; no `np_fallback` retry was needed, no OOM.
- This build does not emit legacy `kv self … MB` / `will fit` lines, so probe
  `kv.fits_budget` is null (same as raider precedent); acceptance followed the
  null-with-verdict-ok rule plus direct VRAM measurements.

## Telemetry aggregates (per receipt, target GPU only)

power_w [min,max,mean] · temp_c_max · mem_mib_max · sm_mhz_mean (slowdown_samples 0
on every op; thermal_or_hw_slowdown_observed false):

| receipt | power W | temp °C | mem MiB | SM MHz |
| --- | --- | --- | --- | --- |
| probe-2048-np1 | 15.8 / 294.6 / 122.6 | 52 | 9468 | 1882 |
| probe-2048-np16 | 17.1 / 300.4 / 125.4 | 58 | 13716 | 1880 |
| probe-4096-np8 | 16.3 / 296.9 / 123.2 | 54 | 12472 | 1880 |
| probe-8192-np4 | 16.6 / 297.7 / 123.9 | 54 | 11850 | 1880 |
| probe-16384-np2 | 16.2 / 298.1 / 123.7 | 56 | 11540 | 1880 |
| probe-32768-np1 | 17.2 / 299.9 / 124.9 | 57 | 11394 | 1880 |
| batch-A-ctx2048 | 16.6 / 319.8 / 265.1 | 69 | 13736 | 2650 |
| batch-B-ctx4096 | 16.6 / 312.6 / 254.6 | 66 | 12488 | 2591 |
| batch-C-ctx8192 | 16.4 / 299.1 / 261.8 | 66 | 11864 | 2596 |
| batch-D-ctx16384 | 16.6 / 298.1 / 247.1 | 64 | 11552 | 2483 |
| batch-E-ctx32768 | 16.3 / 306.6 / 250.5 | 62 | 11404 | 2481 |

No throttling: sustained decode plateaus near 250–265 W mean / ≤ 320 W peak under
the 360 W native limit; max temp 69 °C; SM means 2480–2650 MHz.

## Anomalies and deviations

1. **Probe attempt 1 refused `startup_failed` (11:38Z, retained receipt replaced by
   the ok rerun)**: payload pins `LD_LIBRARY_PATH=<build-cuda>/bin`, but the CUDA
   runtime libs (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`) were
   staged under `<build-cuda>/lib-ext/` on rtx5080-host ⇒ server exited at load. Fixed by
   node-side symlinks `bin/<lib> -> ../lib-ext/<lib>` (pinned binaries and tool
   unchanged; ldd clean afterwards). Labeled `node-staging-symlink-fix-20261030`.
2. **Service stop was manual**: `fak.service` (Qwen2.5-Coder-3B native serve,
   auto-started 09:39Z post-dispatch, 3216 MiB VRAM) would have tripped the
   `service_active` gate; `fak-sync nvlease service stop` pins its SSH target to
   raider, so the stop ran as `sudo -n systemctl stop fak.service` under maintain
   lease `hwl-43b9e563350c6f4e`. Labeled `service_stop=manual-under-maintain-lease`.
3. **DS4.1 transfer v2 typed failure** (see headline): root-caused to v2's
   integer-division chunk truncation (bit-exact replay of both recorded digests);
   v3 completed the transfer. Subsequent out-of-band metadata splice of the live
   shard1 was peer-initiated outside this lane; audited, documented, and preserved
   (`.orig-pin` + patched OID in the receipt).
4. **Prompt generator token counts differ from the raider run**: probe target 128
   yielded 156 actual tokens here (char-length heuristic; deterministic per target,
   matching `prompt_sha256` across probe and cells at each ctx). Cell prompt targets
   are ctx−512 exactly as on raider; actuals are recorded in receipts.

## Comparability

- **Warm-prefix shared-prompt regime**, decode-dominated, max-tokens 64 — same
  workload/client/concurrency axes as the raider-qwen38-batch-20261027 campaign, so
  within-campaign ratios are strict; cross-campaign ratios against raider are NOT
  apples-to-apples: different quant (UD-Q2_K_XL 9.83 GB vs Q4_K_M 17.1 GB), different
  card (desktop 5080 GDDR7 @ native 360 W vs laptop 5090 @ session 150 W cap),
  different boot. Directionally: lighter weights + higher memory bandwidth + uncapped
  power ⇒ 63.9 tps C1@2048 vs raider 22.2 tps C1@2048 (150 W cap), 279.2 tps peak
  vs 144.6 tps.
- **First 5080 numbers on this envelope**: zero prior baseline; absolute deltas only.
- DRAM tier rate is WEAK-labeled (no C toolchain on rtx5080-host); NVMe rate is
  HW-WITNESSED cold-cache.
- All receipts are self-contained: tool commit ref + payload_sha256, model/engine/
  loadgen digests, boot-id, GPU UUID, UTC timestamps, admitted lease records.

## Open items

- DS4.1 first attempt: executed and typed-blocked at model load (export lacks
  `per_layer_token_embd.weight` + 13 `deepseek41.*` KVs vs b11870); needs corrected
  export or qualified alternate engine version — see
  [`ds41-attempt-1/ATTEMPT-RECEIPT.md`](ds41-attempt-1/ATTEMPT-RECEIPT.md).
- DRAM STREAM with a real C toolchain (install gcc on rtx5080-host, re-run tier receipt).
- `fak.service` state at close: see CAMPAIGN-LOG final entry.

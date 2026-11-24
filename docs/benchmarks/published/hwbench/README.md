---
title: "Campaign record: hwbench-20261028 campaign receipts (pi/hwbench-campaign-20261028)"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak hwbench (2026-10-28): hwbench-20261028 campaign receipts (pi/hwbench-campaign-20261028)."
---

# hwbench-20261028 campaign receipts (pi/hwbench-campaign-20261028)

Run window 2026-10-28T03:24–03:35 UTC. All receipts HW-WITNESSED unless noted.
Raider boot id: `d003c8a2-1ac5-4ccf-838a-ec38aec1c968` (service active since 2026-10-27T23:22:34Z).
No guard refusals occurred; no `refusals.txt`. No host power/driver/clock mutation anywhere.

| File | Branch | Verdict |
| --- | --- | --- |
| `lease-census.txt` | 1 lease census | ok — all classes free at start; no ACTIVE leases held by others |
| `absence-egpu-a.json`, `absence-a/` | 2 eGPU typed absence (raider) | `gpu_absent` / TARGET_GPU_ABSENT — matches expected disconnected state |
| `absence-egpu-b.json`, `absence-b/` | 2 eGPU typed absence (raider) | `gpu_absent` / TARGET_GPU_ABSENT — matches expected disconnected state |
| `reachability.txt` | 3 raider reachability | ok — pinned-key SSH accepted; hostname raider-fak; boot id as above; internal RTX 5090 Laptop GPU present |
| `leases.txt` | 4+7 lease lifecycle | measure lease `hwl-c4e620a79809ea63` (epoch 59) granted 03:26:31Z, released 03:28:40Z, verified free |
| `fak-h2h-internal.json` | 5 FAK h2h arm | closed receipt, execution=observed, verdict=`refused`, error_class=`benchmark_incomplete`, cleanup=confirmed; **0/9 rows valid** (6× http_response, 2× http_response decode, 1× usage_missing) |
| `fak-arm-journal-excerpt.txt` | 5 FAK arm root cause (journal) | repeated `cudaMalloc(178257920) failed: out of memory`; KV capacity precheck margin ≈ −200…−290 MB; `stale CUDA status: out of memory`; also `session-registry.json.lock: read-only file system` |
| `llama-arm-skip.json` | 6 llama arm | typed SKIP — service non-idle per criterion (campaign's own FAK requests <5 min old) + degraded CUDA state uncertainty; llama arm already measured 2026-10-27 (decode 24.29 tok/s n=3; cold-512 prefill 922.70 tok/s n=3); service left active/running, original unit unchanged |
| `plan.json` | post-campaign dogfood of the landed `hwbench plan` verb | exit 0; auto-selected raider REACHABLE (boot id matches this campaign), 7 standard cells by priority, no host/workload flags |
| `anthony-probe.txt` | 8 Anthony laptop probe | REACHABLE — key-only SSH via Tailscale IPv4 <redacted-ip> (HostKeyAlias=anthony), key `id_ed25519_prod_to_laptop`; live census: 4070 Laptop gen4 x8 + **both eGPUs back on Anthony at gen4 x4** (state change vs 2026-10-27: egpu-b was gen1 x4, egpu-a absent). Read-only; no mutation |

## Key observations

- FAK arm gap persists and is now typed: GPU OOM on prefill scratch (≈170 MiB allocations fail with ~18–34 MB budget) → HTTP 502 on all six cold rows; one decode completed without authoritative usage. Same failure signature as the 2026-10-27 capture (results.json), i.e. not a flake.
- Peer activity observed mid-run (not acted on): holder `egpu-raider-e2e-20261027` granted measure leases on both eGPU classes at 03:27:23/03:27:27Z (epochs 60/61) — consistent with the eGPUs having moved to Anthony.
- Token-time metrics: none new this session (FAK rows invalid; llama arm skipped). Last HW-WITNESSED token-time numbers remain the 2026-10-27 captures (Fak decode 13.44 tok/s n=2; llama decode 24.29 tok/s n=3; llama cold-512 prefill 922.70 tok/s n=3).

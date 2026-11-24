---
title: "Campaign record: Current GPU baseline collection: Raider and eGPUs"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak gpu-baseline (2026-10-25): Current GPU baseline collection: Raider and eGPUs."
---

# Current GPU baseline collection: Raider and eGPUs

2026-10-25 Pacific. **Physical collection pending: both Raider eGPUs are unhealthy.**
No new token-time samples or native/OSS speed ratios are claimed here.
The [preflight](preflight.json) records the earlier access-loss snapshot and device ownership;
the [campaign](campaign.json) is the ordered collection packet. The user requested
new baselines for current Fak versus OSS across eGPU configurations and the
Raider's different internal GPU.

## Next most useful measurement

Run current Fak-native CUDA versus llama.cpp on the Raider's **internal RTX 5090
Laptop**, using its exact canonical Qwen3.8-27B Q4_K_M file. Start inside the
historically functional **2048-token** native envelope. The previous native
bring-up proves functionality; it does not contain matched performance data.
The existing [Raider OSS corpus](../raider-5090/oss-baselines/README.md) supplies
historical denominators, but a new same-window OSS arm is required for a ratio.

Actual token counts govern admission. The old h2h 2048/6144 synthetic prompt sizes
can exceed those numbers after template/tokenization. Do not send the extended
workload to the existing 2048-token native service. Test larger contexts in a
separate arm and retain refusals, OOMs and timeouts.

## Ordered configuration coverage

| Priority | Host / GPU set | Exact-file comparison | Collection state |
|---|---|---|---|
| 1 | Raider internal 5090, 24 GB | Current Fak vs llama.cpp, canonical 27B Q4_K_M, short requests | Waiting for native service recovery |
| 2 | Raider internal 5090 | Current Fak vs llama.cpp, official Coder-3B Q4_K_M | Waiting for native service recovery and file verification |
| 3–4 | Raider external 5070 Ti A, then B, 16 GB each | Same Coder-3B file and workload as priority 2 | Waiting for migration acceptance and measure leases |
| 5 | Raider internal 5090 | Canonical 27B at longer contexts and repeated prefixes | Native context/headroom unqualified |
| 6 | Raider external A+B | Canonical 27B, explicit layer/split policy | Native multi-GPU support unverified |
| 7 | Raider internal+A | Same canonical file; mixed internal/external split | Depends on simpler cells |
| 8 | Raider internal+A+B | Same file; three separate memory pools | Depends on simpler cells |

For vLLM/SGLang, first bind exact INT4/NVFP4 checkpoint and installed build on the
new Linux layout, then compare single-device/pipeline/tensor configurations where
they fit. Their historical Anthony failures are negative results for that old
environment, not findings about the Raider. Concurrency and sustained qualification
follow the single-client baseline; the h2h collector does not supply simultaneous
clients. No full Cartesian expansion is proposed.

## What changed since the old baseline

The maintenance owner observed both 5070 Ti UUIDs on the Raider at 02:33 UTC on
October 26, corresponding to October 25 Pacific. Both eGPUs remain under that
owner's maintenance leases. Enumeration does not prove computation, inference,
reconnect persistence or measured link bandwidth. The source snapshot and its
digest are bound in the preflight; re-read the owner's published acceptance and
current leases before using either device.

Older eGPU measurements used Anthony/Windows/WSL, an internal 4070 and USB4
controllers on that laptop. They remain a separate historical envelope. Raider
adds Linux, a different CPU/RAM configuration, an internal 5090 and different
external port topology. Re-measure the negotiated links, copy routes, sharing,
power and thermals. Never carry a cross-host result forward as a GPU-only effect.

The prior Raider OSS runs used a measured **95 W** internal-GPU envelope. Observe
the current state without changing power limits or Dynamic Boost. A raised-power
comparison is a distinct experiment.

## Reproducible collection

Use the current committed h2h client, matched model IDs that each endpoint actually
serves, unique run IDs and one report per ledger/run. A first short pilot uses the
flags in `campaign.json` (`--cold 128,512 --reps 3 --decode 128 --turns 2 --system
256 --turn-tokens 64 --answer 16 --timeout 60s --gpu-busy off`). Bind `--host`,
`--arm`, `--model`, `--ledger` and `--run-id` explicitly. Device pinning is an
engine-launch setting; the h2h host label does not select hardware.

`--gpu-busy off` disables the AMD-specific sampler. Capture NVIDIA compute-process
and utilization/power samples separately before, during and after each arm; an
exclusive device lease alone does not establish a quiet CPU/controller/host.
Run engines sequentially in an ABBA window with profilers off. Save correctness
responses before timing, preserve startup and failed trials, record actual input
and output lengths, and compare only matched identities. Profile attribution is a
separate later arm. This is an internal shaped workload, not an industry score.

The [historical inventory](historical-inventory.json) derives request-level counts,
per-run cells and artifact hashes from checked-in ledgers; [its readback](INVENTORY.md)
states coverage and defects. It does not turn historical samples into fresh runs.

After access recovers: verify the host and executing binary, obtain an exclusive
internal-GPU measure lease, run a nonempty functional request through the ordinary
native entrypoint, then collect priority 1. Preserve and restore the existing
serving configuration within the lease. Never overlap two near-full canonical
model processes. An unstarted or failed native arm keeps the ratio null.

## Recovery-boot update

The operator recovered normal startup using an EC reset and confirmed both eGPUs remained connected. [Preserved incident logs](incident-logs/DIAGNOSIS.md) show both external GPUs falling off PCIe during recovery. Both remain authorized/enumerated without working NVIDIA driver binding; the internal5090 responds to management queries. The persistence daemon timed out twice, and Fak's hard service dependency prevented serving startup. The earlier terminal freeze's cause remains unproven. No timing run or recovery service/driver change was performed.

The next control is a normal shutdown and boot with both eGPUs disconnected, followed by verification of internal-GPU serving before any benchmark. Enclosure/port isolation remains with the maintenance owner. The eight-cell queue below remains prepared; its original access preflight is historical evidence, not current health.


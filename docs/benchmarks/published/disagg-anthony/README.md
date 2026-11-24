---
title: "Campaign record: Anthony dual-eGPU prefill/decode proof"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak disagg-anthony (2026-10-25): Anthony dual-eGPU prefill/decode proof."
---

# Anthony dual-eGPU prefill/decode proof

Status: **bidirectional physical P/D correctness witnessed and independently accepted**, 2026-10-26 05:41–05:45 UTC. All four CUDA role processes exited0; A→B and B→A each matched all eight continuation tokens. No production-readiness or speedup claim follows.

## Observed result

| Direction | Producer / consumer Linux PIDs | Snapshot / logits / framed wire bytes | Token parity | Role exits |
| --- | --- | --- | --- | --- |
| A→B | 448 /396 | 334,594 /607,744 /943,451 | 8/8 | 0/0 |
| B→A | 568 /516 | 334,594 /607,744 /943,451 | 8/8 | 0/0 |

All four receipts contain tokens `[1,374,264,16301,304,279,42783,75]`, config SHA-256 `d60173751c6f1dad7518c7a66e2130030e4b2ce191ecbfa8133d3d116baa06b3`, CUDA/sm_120/device-memory and Q8-on-load identities. Independent review read every role receipt, process/lease exits and per-UUID telemetry.

Each eGPU reached3,784MiB; utilization peaks A/B were91%/89% for A→B and94%/89% for B→A. These aggregate samples correlate the declared single-UUID process selection with physical activity; they do not provide runtime UUID or PID-to-device attestation. Invalid telemetry cells remain in the CSV and are excluded from numeric summaries. The source command explicitly labels its narrower evidence scope.

Initial and final inventories show all three GPUs at0MiB. Final WSL process query returned only its header. Both canonical measure leases were explicitly released successfully, then the host window was handed back for baseline work. Both finite telemetry shell sessions ended with outer exit1; the exact remote timeout exit is unknown. Role exits are separately observed0/0 in each direction.

Read [execution.json](execution.json), the four role stdout files, two telemetry CSVs, [launch-plan.json](launch-plan.json), [admissions.json](admissions.json) and [raw-manifest.json](raw-manifest.json). Text copies normalize line endings; the manifest preserves original byte counts/hashes and records the transformation. [REDACTION.json](REDACTION.json) records mesh-address and user-directory redaction in committed copies; these commands require the enrolled host values before replay. Original bytes remain in the owning local scratch bundle.

## Fixed experiment

- Executable: public cmd/pdbench at 2cc7d389469e13470aed47239e96bd82827aaaf0; SHA-256 2987f38001bf677fc2c43011a2d3a92f1c422a32406b5f7be6c71ecf257e6a5f.
- Build: Linux amd64, Go1.26.10, CUDA13.4, sm_120 SASS, no PTX. GPU-free build; externally bound to the exact Git source archive, without embedded VCS metadata.
- Host: Anthony Windows laptop, Ubuntu-24.04 WSL2, NVIDIA621.01, CUDA13.0 runtime. ldd -r passed before execution.
- Model: Qwen2.5-Coder3B Instruct Q4_K_M GGUF, 2,104,932,800 bytes, SHA-256 724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7; CUDA uses existing Q8-on-load.
- eGPU A: GPU-a9127ec5-7901-e9bf-42b8-81b798a32010; B: GPU-248188e5-f66f-da22-8911-5c71d271b37b. Each RTX5070Ti16GiB, USB4. Internal4070 is excluded from role selection.
- Exact prompt IDs 1,42,17; eight greedy continuation tokens; separate producer and consumer OS processes; real loopback TCP.
- Consumer restores complete prefix snapshot and final prefill logits, then decodes without calling Prefill. Producer also computes a local reference continuation solely for correctness comparison.
- Hard330-second external process timeout; internal five-minute socket/stage deadline. Model load/hash are excluded from the internal timeout.
- Canonical eGPU lease authority remains the control desktop. Physical host movement does not create a second lease authority.

## Acceptance

Both A→B and B→A must exit0/0, report cuda/device-memory backend identities and distinct selected UUIDs, identical model/config identities, equal nonzero framed wire byte counts, and exact eight-token parity. Retain per-UUID memory/utilization, process/socket observations, argv, admission/lease receipts and final cleanup.

The output UUID is a CUDA_VISIBLE_DEVICES declaration. Correlate controlled process lifetimes with actual UUID telemetry; report unavailable NVML process attribution explicitly. Keep diagnostic times separate from serving TTFT/TPOT or speedup.

## Preserved failures

The first consumer deadline expired before the producer finished cold loading. Consumer exit1 at05:02:39Z; producer connection-refused exit1 at05:02:48Z. Empty outputs prove no continuation; no physical parity claim follows.

Subsequent unexecuted proposals returned actual Fak DEFAULT_DENY: gw3 outside-workspace, gw4/gw5 unknown local write footprint. Source-supported explicit local stdout/stderr targets resolved admission in newly scoped ab04 commands; all four roles were admitted with read_only=false. No denied command was executed, and no guard setting changed.

## Next steps after correctness

1. Complete process/UUID binding and repeat bidirectional continuation across longer prompts and session cleanup failures. Observe snapshot bytes, restore/decode duration and failures without claiming production readiness.
2. Extend the ordinary public serving entrypoint with complete-state P/D handoff. Require caller readback, real transport, consumer no-prefill, cancellation cleanup and first/restored-token parity. The standalone proof command currently has no production serving integration.
3. Compare equal two-device budgets: colocated execution, independent replicas and P/D, with identical model/quality, prompt/arrival trace, cache state and latency limits. Begin with at least20 finite requests per cell; retain failures and report TTFT, TPOT p50/p95, deadline-goodput and transport/GPU utilization. No speedup or p99 claim is supported by this eight-token diagnostic.


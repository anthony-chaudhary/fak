---
title: "Campaign record: Raider Arch eGPU incident recovery, 2026-10-26"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak egpu-recovery (2026-10-26): Raider Arch eGPU incident recovery, 2026-10-26."
---

# Raider Arch eGPU incident recovery, 2026-10-26

## Outcome and physical scope

Recover persistent Linux incident logs with both external enclosures physically
disconnected. Target Arch Linux on the Raider; Windows operation already works.
The operator withdrew the proposed reconnection step and requires offline work.
Do not describe the cards as repaired or qualified. No PCI configuration write,
Gen3 cap, driver/module replacement, unbind, rescan, reboot, or enclosure change
was performed in this investigation.

## Captured evidence

The native bounded, pinned-host-key reader collected these immutable raw JSON
receipts. Each contains exact probe argv, timestamps, exit status, byte counts,
SHA-256, and retained streams. Whole-file hashes are in `manifest.json`.

| File | Collection UTC | Observation |
| --- | --- | --- |
| `disconnected-control.json` | 21:43:23–21:43:30 | Current boot `2a2c614f-4573-46fd-b05b-d03909ecea28`; internal5090 UUID present and bound; external UUIDs absent; ordinary Fak and bolt services active. Management evidence only. |
| `incident-previous.json` | 22:33:48–22:34:00 | Selected boot `8686dff1873246698c1f1c89f4bf27ba`; default prior-boot Xid capture retains both external Xid79/154 at18:07:40UTC. Terminal excerpt shows an orderly shutdown at21:31:15UTC. |
| `incident-cuda-loss.json` | 22:36:25–22:36:36 | Selected boot `5c922548f21a48718134f56d17c50c31`; both enclosures enumerate; NVIDIA reports unavailable PCI port handles at20.590/47.122seconds; UCSI errors and repeated USB configuration errors occur. Journal ends near03:40:31UTC while the last retained Fak metrics request succeeds. |

The selected CUDA-loss boot's retained transport-filter output is3538bytes,
not byte-truncated, and contains no AER replay-timeout event. This is a filtered
observation, not proof that no unlogged link failure occurred. The terminal
excerpt contains no completed GPU workload or definitive crash cause. A missing
network response and abrupt journal end do not distinguish power loss, network
failure, firmware lock, or an unlogged kernel hang.

Historical hot-remove investigations separately found NVIDIA DRM teardown
stalls. Do not combine an intentional unplug's Xid, a later boot-time link loss,
and a CUDA-adjacent loss of SSH into one proven causal chain. Earlier cached
receipts in `_scratch/guard-remote-rollout/` retain those distinctions.

## Why the process became confused

1. Earlier bring-up notes were read as current inventory. The catalog legitimately
   records the last return to Windows; it does not override the requested Arch
   target. An identity mismatch must prompt placement reconciliation, not a reset.
2. The diagnostic read whole-current-boot faults without an experiment boundary.
   An earlier fault or intentional unplug could contaminate a later control.
3. Every verdict used the same reset advice, including clean management reads,
   missing evidence and deliberately disconnected targets.
4. Management visibility, completed collection and actual CUDA readiness are
   different outcomes. A zero-exit diagnostic has always kept qualification false.
5. Some non-PTY shell results in this session lost output despite exit0. PTY
   readback recovered the artifacts. Empty tool output is not an empty journal.

The process repair adds explicit attachment intent, a validated current-boot
experiment interval, selected incident-boot collection, and state-derived next
actions. It preserves raw history, host/UUID fences and unknown evidence; it
never promotes software tests or NVML visibility into physical qualification.

## Gen3 disposition

[NVIDIA issue974](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/974)
contains a same-driver615.78.08 report of Gen4 replay errors and successful Gen3
tests on different hardware. [Issue979](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/979)
contains reports involving pre-bind bridge caps and autonomous-speed controls;
it also distinguishes detection fixes from compute failures. These are candidate
experiments, not a universal driver specification or proof of a Raider remedy.
The collected Raider transport evidence does not establish that specific replay
failure, and disconnected devices expose no live endpoint/parent registers.

The [Blackwell eGPU manager's requirements](https://github.com/DamianKA1993/blackwell-egpu-manager#prerequisites--system-requirements)
exclude hosts with an internal NVIDIA dGPU. The Raider has one. Do not install
that manager unchanged, copy its global module actions, or infer safety from a
report on a different host.

Keep Gen3 as a conditional, reversible, exact-pair experiment. A usable packet
must pin UUID, enclosure, current endpoint and immediate parent, current boot,
before/after registers, maintenance ownership, workload bounds and rollback.
Avoid shared module unloads or global GSP/IOMMU/ASPM changes. Under the current
no-connection constraint, physical link and CUDA acceptance remain unrun.

## Verification

Run `go test ./cmd/fak-sync -run 'TestNVLeaseDiagnose|TestNvlease|TestNVLease' -count=1`.
Regressions cover host mismatch, expected disconnection, contradictory attachment,
fresh versus historical faults, invalid context, failed journal access, truncated
intervals and state-dependent advice. A distinct reviewer checked source and
found two masking defects; both were corrected and reviewed again.

The live read-only collection validates invocation and retained evidence. It
does not test a driver or Gen3 change and does not qualify either external GPU.

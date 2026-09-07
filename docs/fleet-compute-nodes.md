---
title: "Choose sanctioned compute for hardware-gated work"
description: "Operator route for selecting local proof, cloud GPU, private lab GPU, DC/CPU, or nightly evidence collection by workload and witness requirement."
---

# Choose sanctioned compute for hardware-gated work

**Audience:** operators deciding where a fak task must run to produce valid hardware or
network evidence. The workstation is the **control point** for planning, dispatch, and
ledgering; the selected node is the **compute boundary** for the workload and witness.

**Current default:** use the local 60-second proof when the claim is about the kernel gate
and does not require a device. **Next action:** match the required evidence to one row
below and run that row's preflight before dispatching the workload.

## Choose by workload and evidence

| Required evidence | Sanctioned target | Current support boundary | Preflight and next action |
|---|---|---|---|
| **Kernel policy and offline behavior; no device claim** | Local control point | The canonical proof needs no key, model, or GPU. It proves the policy and offline agent path, not CUDA, accelerator throughput, or a datacenter network. | Run the [60-second offline proof](repro-packet.md#the-60-second-offline-proof). Stop here only when its evidence matches the claim. |
| **AMD Strix Halo APU sub-kernel, Vulkan compute, or differential ablation witness** | AMD Strix Halo appliance (`strix1` / `strix-halo-fak.local`) | AMD Ryzen AI MAX+ 395 (16 Zen 5 cores, 32 threads) + Radeon 8060S (40 CUs, gfx1151) with 64GB LPDDR5X UMA. Validated natively via Mesa RADV Vulkan compute. | Run `fak-dev amd-strix-probe` to verify LAN presence and hardware facts; run `fak-dev amd-strix-validate` or `fak validate --strix --subkernels=all --ablate=all` for physical sub-kernel verification and differential ablation. |
| **Live CUDA serve or an operator-controlled cloud GPU burst** | GCP GPU node | The public cloud route supports quota discovery, a dry-run command preview, and bounded instance creation. Availability, quota, accelerator shape, region, and price remain observations from the selected project at dispatch time. | Run `python tools/gcp_gpu_probe.py --all-tiers`. A candidate is eligible only when the probe reports quota for its accelerator/region and does not report an auth or API blocker. Then run `python tools/gcp_bench.py --dry-run` and use its printed command, or run `GCP_PROJECT=<id> ./scripts/gcp-qwen-serve.sh --apply`. A create failure is a failed availability check, not a witness. |
| **Device-GEMM, kernel, or multi-GPU witness on private lab hardware** | Private lab GPU server | Access, node identity, commands, credentials, and transcripts are private. The public repo may receive only a scrubbed result and its scoped hardware witness. | Open the [private control-channel route](private-comms-channel.md) and follow its authorized private runbook; return the scrubbed witness through the [lab development loop](fak/lab-dev-loop.md). |
| **Datacenter-network access or heavy CPU evidence** | Private lab CPU/DC target | The target is CPU-only for accelerator claims. Network reach, credentials, node selection, and transcripts remain private operator data. | Use the same [private control-channel route](private-comms-channel.md), select the CPU/DC workload in its maintained node map, and return only scrubbed evidence. |
| **Recurring device evidence rather than an interactive result** | Nightrun device-witness pipeline | The pipeline collects scheduled hardware evidence. Its runbook chooses the configured cloud or private worker; that physical target retains its own access boundary. The public ledger proves only the task, build, scrubbed machine class, and scope recorded in each row, not live capacity. | Follow the [nightrun runbook](nightrun/README.md) and verify the resulting row in `docs/nightrun/collected.jsonl`; use that row, rather than the worker's location, as the public readback. |

The evidence requirement selects the target. “GPU” alone is insufficient: a live serving
check, a single-device kernel witness, a multi-GPU result, and a recurring nightly ledger
are different jobs. When both a public-cloud and private-lab target meet the same device requirement, choose
public cloud for a self-service single-node run when its probe reports eligibility; choose
private lab for multi-GPU, device-specific lab hardware, DC reach, or a witness whose
maintained runbook is private. If neither condition decides the target, use nightrun for a
recurring job and ask the authorized lab operator to select the maintained interactive
route for a one-off job.

## Public choice versus private control

This page is the public **selection route**. It names workload classes, public preflights,
and the evidence that must come back. It does not carry the private node map, bridge
commands, credentials, channel identifiers, recovery steps, or raw transcripts. Those
belong to the authorized companion runbook reached through
[the private-control public stub](private-comms-channel.md).

A missing credential or bridge session changes the next actor, not the compute
requirement. An unauthorized reader can select the workload class but cannot verify a
private target's identity or availability; that is an intentional support boundary, not a
publicly reproducible check. Hand the authorized operator the selected row and public
preflight; the operator completes and records private discovery in the private runbook.
Do not reconstruct private commands from dated public notes, and do not report local
hardware absence as the result of a task whose claim requires remote hardware.

## Witness contract

A hardware claim is supported only by a run on a target that provides the required device
or network boundary. For CUDA acceptance, for example,
`FAK_CUDA_ARCH=<matching-arch> bash tools/cuda_acceptance.sh` (or `make cuda-accept`)
must execute on the selected GPU node. Its exit 3 on a non-GPU host is an honest skip,
not a passing device witness.

Every returned result should identify the tested commit or module revision, workload,
machine class and accelerator architecture where applicable, command or runbook step,
result, and artifact location. Scrub private identifiers before committing public
evidence. The [GPU-server private boundary](gpu-server-private-boundary.md) is authoritative
for what may cross repositories.

## Mode, generation, support, and lifecycle

| Context | Current contract |
|---|---|
| Mode | Public selection and evidence routing. Private control is a separate authorized mode; model-serving traffic is separate from both. |
| Generation | This is the current (`gen/now`) selection route. Linked runbooks own executable commands and supersede dated notes. |
| Support | Public cloud tools are supported to the behavior their current preflight reports. Private targets are supported only for authorized operators through the private runbook. Nightrun support is scoped to recorded ledger rows. |
| Lifecycle | The maintainer of this public table records selection status here; the maintainer of each linked runbook records executable details there. Add or promote a row only with a working preflight plus a returned witness. Mark the row held when access or capacity cannot be established. Remove or replace it when its maintained runbook names a successor. |

Capacity is checked at dispatch time; a dated quota or stockout observation is not a
permanent support promise. If a preflight finds no eligible cloud target, preserve its
output and choose another row that can produce the same required evidence, or hand the
exact public preflight and selected requirement to the authorized operator.

## Enforcement and related routes

`fak hwgate-lint` detects a final answer that treats missing local hardware as terminal
and redirects it to this selection route. `fak manage` can enforce the same rule at Stop,
and `fak guard-stops` reports the resulting redirects. These gates select a route; they do
not fabricate a hardware witness.

- [Private control channel](private-comms-channel.md) — authorized discovery and readback.
- [Lab development loop](fak/lab-dev-loop.md) — return scrubbed hardware evidence.
- [Nightrun](nightrun/README.md) — schedule and read recurring evidence.
- [Backend selection](supported/backends.md) — choose model execution separately from the
  node that hosts it.

### Anthony laptop (registered baseline)

`anthony-laptop` is registered for offline, CPU, single-GPU CUDA, dataset, and single-user serving benchmarks. It is explicitly excluded from public serving, unattended overnight work, multi-GPU work, and datacenter-network measurements. `fak bench-loop fleet status --json` reports the registration and its latest comparable baseline.

The authenticated route and raw evidence live behind the sanctioned private bridge; public coordinates are intentionally absent. The scrubbed three-trial receipt, rerun class, power/thermal preconditions, cleanup rules, and known measurement limitations are in [`docs/_witnesses/issue-8486-anthony-laptop/`](_witnesses/issue-8486-anthony-laptop/README.md). Authorized operators must use the current adjacent `fak-private` runbook rather than copying a route from public documentation.

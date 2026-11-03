---
title: "Choose sanctioned compute for hardware-gated work"
description: "Operator route for selecting local proof, enrolled LAN GPU, DC/CPU, or nightly evidence collection by workload and witness requirement."
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
| **Local silicon accelerator verification or micro-dose probe (<50ms)** | Local physical silicon (Metal / CUDA / Vulkan) | Apple Silicon Metal (M-series), local NVIDIA CUDA, or local Vulkan. Direct in-process CGo execution. | Run `fak hil` (<50ms physical micro-dose suite) or `fak hil --probe` to verify physical execution capabilities and enforce active hardware bias before executing larger workloads. |
| **Historical Strix Halo appliance evidence** | Retired appliance records | Historical receipts remain available; retired appliances are not testing capacity. | Select an eligible enrolled local or LAN target for new evidence. Do not wake or dispatch to retired appliances. |
| **Live CUDA serve or device witness** | Eligible enrolled local host or LAN CUDA node | GCP test targets are retired. Historical cloud receipts and generic provider support do not establish current testing capacity. | Verify current enrollment, trust, lease, reachability, runtime, and required device capability using the maintained operator inventory before dispatch. Record the actual target and scoped result. Do not reauthenticate or provision cloud capacity for testing. |
| **Device-GEMM, kernel, or multi-GPU witness on private lab hardware** | Eligible enrolled owned LAN GPU node | Current trust, lease, device capability, and access must be established; retired external lab records provide no admission. Node identity, commands, credentials, and transcripts remain private. | Use the maintained private inventory to admit the owned LAN target, run the scoped witness, and return only a scrubbed result through the [lab development loop](fak/lab-dev-loop.md). |
| **Datacenter-network access or heavy CPU evidence** | Eligible enrolled owned LAN CPU/network node | The target must expose the required CPU/network boundary; CPU-only execution cannot establish accelerator claims. Retired external lab records provide no admission. | Establish current trust, lease, reachability, and workload capability in the maintained private inventory; return only scrubbed evidence. If the required boundary is absent, retain an unverified gap. |
| **Recurring device evidence rather than an interactive result** | Nightrun device-witness pipeline | The pipeline collects scheduled hardware evidence. Its runbook selects an eligible enrolled local or LAN worker; that physical target retains its own access boundary. The public ledger proves only the task, build, scrubbed machine class, and scope recorded in each row, not live capacity. | Follow the [nightrun runbook](nightrun/README.md) and verify the resulting row in `docs/nightrun/collected.jsonl`; use that row, rather than the worker's location, as the public readback. |

The evidence requirement selects the target. “GPU” alone is insufficient: a live serving
check, a single-device kernel witness, a multi-GPU result, and a recurring nightly ledger
are different jobs. Select an eligible enrolled local or LAN target whose current
capability matches the required evidence. A retired DGX or GCP record is historical,
not an active route. If no target passes admission, record the precise unverified gap;
a cloud login, provisioning plan, or operator handoff is not a passing test.

## Public choice versus private control

This page is the public **selection route**. It names workload classes, public preflights,
and the evidence that must come back. It does not carry the private node map, bridge
commands, credentials, channel identifiers, recovery steps, or raw transcripts. Those
belong to the authorized companion runbook reached through
[the private-control public stub](private-comms-channel.md).

A missing credential or admission leaves the hardware requirement unverified. An unauthorized reader can select the workload class but cannot verify a
private target's identity or availability; that is an intentional support boundary, not a
publicly reproducible check. Record the selected workload and unverified gap in the maintained private runbook;
an operator handoff alone does not establish availability or satisfy the witness.
Do not reconstruct private commands from dated public notes, and do not report local
hardware absence as the result of a task whose claim requires remote hardware.

## Witness contract

A hardware claim is supported only by a run on a target that provides the required device
or network boundary. For CUDA acceptance, for example,
`FAK_CUDA_ARCH=<matching-arch> bash tools/cuda_acceptance.sh` (or `make cuda-accept`)
must execute on the selected GPU node. Its exit 3 on a non-GPU host is an honest skip,
not a passing device witness.

### Real hardware comparisons vs. software simulations

Software simulations (analytical rooflines, execution trace simulations, mock fixtures) are strictly
**early indicators and search-space bounds**—never comparative claims, achieved speedups, or victory proofs.
Any reported head-to-head comparison must be measured on real physical silicon under matched operating
envelopes. Any comparison citing a simulated arm is gated as `EARLY_INDICATOR_ONLY` via
`fak hil --audit-comparison` and is structurally forbidden from claiming victory, advertising an
achieved speedup, or closing performance issues.

Before running comparative benchmarks or sub-kernel evaluations:
- **Local silicon** (Metal / CUDA / Vulkan) must be probed via `fak hil` (<50ms physical micro-dose suite)
  or `fak hil --probe` to verify physical capabilities and establish an active hardware bias.
- **LAN nodes** must first pass current enrollment, trust, lease, reachability, runtime,
  and device-capability checks through the maintained private inventory. Run only the
  scoped physical witness supported by that admitted node; retired appliance recipes
  and historical discovery results provide no testing capacity.

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
| Support | Testing uses eligible enrolled local or owned LAN targets admitted through the maintained private inventory. GCP and retired external lab targets are excluded. Nightrun support is scoped to recorded ledger rows. |
| Lifecycle | The maintainer of this public table records selection status here; the maintainer of each linked runbook records executable details there. Add or promote a row only with a working preflight plus a returned witness. Mark the row held when access or capacity cannot be established. Remove or replace it when its maintained runbook names a successor. |

Capacity is checked at dispatch time; dated inventory and historical receipts provide
no current admission. If no eligible local or LAN target can produce the required
evidence, record the exact unverified gap. Do not substitute cloud provisioning,
reauthentication, or an operator handoff for the missing physical witness.

## Enforcement and related routes

`fak hwgate-lint` detects a final answer that treats missing local hardware as terminal
and redirects it to this selection route. `fak manage` can enforce the same rule at Stop,
and `fak guard-stops` reports the resulting redirects. These gates select a route; they do
not fabricate a hardware witness.

### 100× micro-dose HIL testing contract (`fak hil`)

Alongside `fak hwgate-lint`'s routing gate, `fak hil` enforces the physical execution invariant
through **100× more frequent micro-doses (<50ms physical silicon probes)**:
- **Sub-50ms Silicon Probes:** Runs micro-doses exercising context allocation, compute (GEMV/GEMM),
  bandwidth, and numerical parity on actual silicon (Metal on Darwin, CUDA on Linux/Windows,
  Vulkan on an admitted supported device).
- **Real HW Comparison Auditing:** `fak hil --audit-comparison` audits comparative claims against
  baselines, failing with exit code 1 if either candidate or baseline was simulated, and forbidding
  issue closure or speedup claims from simulation leakage.
- **Active Hardware Bias:** Enforces an active bias toward physical silicon whenever an accelerator
  is present on the host or reachable on the LAN/fleet, preventing silent fallbacks to CPU mocks.
- **Detailed Architecture & Roadmap:** See [`docs/benchmarks/HARDWARE-IN-THE-LOOP-ROADMAP.md`](benchmarks/HARDWARE-IN-THE-LOOP-ROADMAP.md).

- [Private control channel](private-comms-channel.md) — authorized discovery and readback.
- [Lab development loop](fak/lab-dev-loop.md) — return scrubbed hardware evidence.
- [Nightrun](nightrun/README.md) — schedule and read recurring evidence.
- [Backend selection](supported/backends.md) — choose model execution separately from the
  node that hosts it.
- [HIL testing roadmap](benchmarks/HARDWARE-IN-THE-LOOP-ROADMAP.md) — hardware-in-the-loop roadmap and verification contract.

### Anthony laptop (registered baseline)

`anthony-laptop` is registered for offline, CPU, single-GPU CUDA, dataset, and single-user serving benchmarks. It is explicitly excluded from public serving, unattended overnight work, multi-GPU work, and datacenter-network measurements. `fak bench-loop fleet status --json` reports the registration and its latest comparable baseline.

The authenticated route and raw evidence live behind the sanctioned private bridge; public coordinates are intentionally absent. The scrubbed three-trial receipt, rerun class, power/thermal preconditions, cleanup rules, and known measurement limitations are in [`docs/_witnesses/issue-8486-anthony-laptop/`](_witnesses/issue-8486-anthony-laptop/README.md). Authorized operators must use the current adjacent `fak-private` runbook rather than copying a route from public documentation.

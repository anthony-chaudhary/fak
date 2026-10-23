---
title: "NVIDIA RTX 5090 FE & Gigabyte X570 BaM P2PDMA Planning and Diagnostics"
description: "Target hardware planning, observed inventory, and explicit simulation boundaries for the unqualified NVIDIA RTX 5090 BaM NVMe P2PDMA path."
---

# NVIDIA RTX 5090 FE & Gigabyte X570 BaM P2PDMA Planning and Diagnostics

> **Scope & status: unqualified hardware target, not a deployment guarantee.**
> The RTX 5090 FE / Ryzen 9 5950X / X570 workstation is a representative target for the BaM-style storage model. The current `cuda-gpudirect` commands do not implement a physical NVMe-to-GPU transfer or an end-to-end Qwen inference benchmark. Installing this hardware or changing BIOS/driver settings does not turn the simulation into a physical backend.
>
> Hardware `inspect` reports available inventory only. Unprobed settings and route properties remain **UNKNOWN/UNVERIFIED**. Explicit simulation exercises in-memory descriptors, queues, and state coordination; it does not qualify hardware performance or physical zero-copy operation.

Related model and evidence limits: [`QWEN38-NVIDIA-5090-GPUDIRECT-RESULTS.md`](../benchmarks/QWEN38-NVIDIA-5090-GPUDIRECT-RESULTS.md)
Simulation receipt schema: `fak.modelengine.qwen38-cudadirect-swap/1`
Tracking issue: [#11326](https://github.com/anthony-chaudhary/fak/issues/11326)

---

## 1. Representative Workstation Profile

These are **model inputs**, not a discovered or validated workstation configuration:

| Component | Representative target | Intended role |
| :--- | :--- | :--- |
| GPU | NVIDIA GeForce RTX 5090 FE, 32 GB GDDR7, `sm_120` | Tier 0 weights, active KV pages, modeled BaM queues |
| CPU | AMD Ryzen 9 5950X | Host processing and PCIe root-complex target |
| Motherboard | Gigabyte X570 Aorus Elite WiFi | Candidate PCIe interconnect layout |
| Host memory | 128 GB DDR4 | Tier 1 model capacity, including embedding/PLE placeholders |
| NVMe SSD | Illustrative 2 TB NVMe SSD | Tier 2 storage target |

Neither a device name nor its PCI address proves VRAM capacity, BAR1 size, ReBAR, ACS behavior, link bandwidth, motherboard slot identity, or supported peer DMA. Record those properties from appropriate hardware/driver evidence before using them as admission inputs.

## 2. Motherboard Layout and Slot Planning

The representative layout uses the primary `PCIEX16` GPU slot and `M2A_CPU` NVMe slot rather than a chipset-attached `M2B_SB` slot. Confirm wiring against the manual for the exact board revision and inspect the actual PCIe ancestry. Linux sysfs names and BDF numbering alone do not establish a named motherboard slot or a usable P2P path.

A shorter candidate path may avoid shared chipset resources, but this guide provides no measured latency or bandwidth for either placement. In particular, the earlier **680 ns**, **7.1 GB/s**, and chipset slowdown figures were illustrative assumptions, not physical observations. An apparent common root complex does not prove driver-supported DMA or the absence of ACS redirection.

## 3. UEFI / BIOS Planning

Above 4G Decoding, Resizable BAR, negotiated PCIe link settings, ten-bit tag support, and boot mode are properties to inspect with the exact board, GPU, and driver documentation. The current `cuda-gpudirect audit` does not read enough firmware/register state to certify these settings.

Do not infer a full 32 GiB BAR1 aperture from a GPU model string, and do not treat an unknown value as a disabled or enabled setting. Record observed values and their source separately from the representative model.

IOMMU and ACS changes can weaken device isolation. This guide does not prescribe disabling them to obtain a favorable audit result. Any authorized configuration experiment needs a recorded baseline, an exact supported change, and a rollback plan; a modeled receipt cannot validate that change.

## 4. Linux Kernel and Driver Planning

Record the exact OS/kernel, GPU driver, device identifiers, negotiated links, and supported peer-memory interface before attempting hardware integration. Consult the matching official kernel and NVIDIA driver documentation for supported parameters and device combinations.

Earlier versions of this guide presented `pcie_p2pdma=1`, `NVreg_EnableResizableBar=1`, `NVreg_EnableGpuFirmware=0`, and `RMDisableP2PDMA=0` as a verified setup recipe. This path has no physical witness establishing that recipe. Do not copy those options or rebuild a boot configuration on the basis of this guide. Driver availability is not the only missing piece: the current CLI also lacks a physical transfer/execution backend.

---

## 5. Diagnostics and Explicit Simulation

### Step 5.1: Inspect available hardware inventory

```bash
go run ./cmd/fak-dev cuda-gpudirect inspect --json
```

The hardware path reads available inventory without substituting a representative RTX 5090 or NVMe SSD when devices are absent. A successful enumeration exits zero while its verdict remains **UNVERIFIED**: enumeration is not a physical execution witness.

With `--sysfs` omitted or empty, discovery follows the platform: Linux reads `/sys`; Windows queries CIM display identity/driver data only, without an NVMe probe or inferred VRAM/BAR1/ReBAR. Windows-reported adapter RAM is not treated as authoritative GPU VRAM. Query/parse errors are surfaced, and other platforms report unsupported discovery. An explicit nonempty `--sysfs PATH`, including `/sys`, selects the Linux sysfs parser on every platform. It does not enable hardware execution or simulation. `--simulate` instead selects representative fixtures before any hardware probe.

Interpret the output conservatively:
- `provenance: OBSERVED` describes inventory observations, not measured P2PDMA or model execution.
- `is_physical_silicon: false` means no physical execution witness was produced, even if real devices were enumerated.
- `unknown_fields` identifies unprobed capabilities. Missing host specifications, VRAM/BAR1 sizes, ReBAR, slot mappings, ACS/IOMMU settings, and route performance must not become healthy defaults.
- A probe error returns nonzero with diagnostic context; any partial inventory is not a complete discovery result.

### Step 5.2: Read the hardware audit limits

```bash
go run ./cmd/fak-dev cuda-gpudirect audit --json
```

The current audit reports unsupported checks as **UNKNOWN**, `healthy: false`, and verdict **UNVERIFIED**, with a nonzero exit. This is an evidence gap, not proof that the workstation is defective. There is no fabricated all-PASS output to match and no automatic BIOS/driver remediation established by this audit.

The following are exit codes of the **`fak-dev` executable**. The examples use `go run`, which returns its own wrapper status and prints `exit status N` for a child failure; do not expect the shell to receive the child's exact code through `go run`.

Executable exit meanings are deliberate:
- `inspect`: **0** for completed enumeration, **1** for a probe error; neither means qualified DMA.
- `audit`: **1** for the current unknown/unsupported checks. `audit --simulate` also exits **1**, with MODELED provenance/verdict, UNKNOWN checks, and `healthy: false`.
- `bench`, `qwen38`, or `qwen38-bench` without `--simulate`: **2** (unsupported physical mode/usage). Invalid or surplus arguments also return **2** before probing.

### Step 5.3: Exercise the in-memory BaM queue model

```bash
go run ./cmd/fak-dev cuda-gpudirect bench --simulate --json
```

`bench` without `--simulate` refuses rather than silently running a model or claiming a hardware benchmark. The explicit simulation uses representative fixtures and in-memory queue operations, even on a GPU-equipped host. It reports **MODELED**, `is_physical_silicon: false`, and `zero_copy_verified: false`.

Legacy numeric throughput/IOPS fields are zero placeholders for **unmeasured**, not physical zero-performance results. A descriptor's zero staging count, a simulated doorbell count, or elapsed Go execution time does not measure PCIe DMA, GPU queue residency, or host bounce copies.

### Step 5.4: Exercise the Qwen state-coordination simulation

```bash
go run ./cmd/fak-dev cuda-gpudirect qwen38 --simulate --json
# Equivalent explicit simulation entry points:
go run ./cmd/fak-dev cuda-gpudirect qwen38-bench --simulate --json
go run ./cmd/fak-dev cuda-gpudirect qwen38-sim --json
```

`qwen38` and `qwen38-bench` require `--simulate`; `qwen38-sim` is the shorthand. These commands do not load model weights or run physical token generation. A successful receipt is **MODELED**, not PASS or PHYSICAL_WITNESSED, and its legacy performance fields remain unmeasured placeholders. A constructor or state-restoration error returns nonzero without a success receipt; do not promote it into one.

Capture any run in a fresh, exclusively owned evidence directory. Preserve stderr, exit status, source identity, and the original receipt together. Never overwrite a historical receipt or relabel modeled output as hardware evidence.

---

## 6. Troubleshooting and Next Physical Witness

| Observation | Meaning | Next check |
| :--- | :--- | :--- |
| Empty device list | No matching inventory was discovered | Check the requested sysfs root and the host's actual device visibility; do not inject fixtures into hardware output. |
| `probe_error` / nonzero probe exit | Enumeration was incomplete or unavailable | Inspect the reported error and preserve partial inventory separately from a complete result. |
| `UNKNOWN` / `UNVERIFIED`, `healthy: false` | Required capability evidence is missing | Gather actual firmware, driver, register, and route evidence; do not assume a failed device or change isolation settings blindly. |
| `bench` / `qwen38` refuses without `--simulate` | The physical execution backend is not implemented | Use explicit simulation only for model/interface work; implement and separately witness the hardware path. |
| Modeled throughput fields are zero | Physical rates were not measured | Do not interpret these placeholders as a benchmark. |
| Qwen simulation returns an error | The in-memory path did not complete | Preserve the exact failure and repair it before claiming model/interface success. |

The next qualification step is an implemented physical backend with traceable on-device transfers, data-integrity checks, measured timings, and an end-to-end fak-native workload. See the [promotion requirements](../benchmarks/QWEN38-NVIDIA-5090-GPUDIRECT-RESULTS.md#8-promotion-gate--the-missing-physical-witness). Hardware inventory and a successful simulation remain separate evidence classes.

---
title: "NVIDIA RTX 5090 FE & Gigabyte X570 BaM P2PDMA Deployment Guide"
description: "Step-by-step hardware layout, UEFI/BIOS configuration, Linux kernel tuning, and validation guide for zero-copy BaM NVMe P2PDMA on NVIDIA RTX 5090 FE and AMD Ryzen 9 5950X."
---

# NVIDIA RTX 5090 FE & Gigabyte X570 BaM P2PDMA Deployment Guide

> **Scope & Purpose:**  
> This guide provides the complete deployment runbook to configure and verify **zero-copy Big Accelerator Memory (BaM) Peer-to-Peer DMA (P2PDMA)** on a workstation pairing an **NVIDIA GeForce RTX 5090 FE (32GB GDDR7, sm_120)** with an **AMD Ryzen 9 5950X** on a **Gigabyte X570 Aorus Elite WiFi** motherboard.  
> Following this guide ensures direct PCIe root complex switching (<700ns latency), full 32GB Resizable BAR aperture mapping, zero host DRAM bounce copies (`StagingCopyCount() == 0`), and seamless operation of `fak`'s 3-Tier Hierarchical Memory Manager for Qwen 3.8 models.

Related benchmark evidence: [`docs/benchmarks/QWEN38-NVIDIA-5090-GPUDIRECT-RESULTS.md`](../benchmarks/QWEN38-NVIDIA-5090-GPUDIRECT-RESULTS.md)  
Receipt schema: `fak.modelengine.qwen38-cudadirect-swap/1`  
Tracking issue: [#11326](https://github.com/anthony-chaudhary/fak/issues/11326)

---

## 1. Workstation Hardware Specification

The validated target profile consists of:

| Component | Hardware Specification | Role in Architecture |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5090 Founders Edition (32GB GDDR7, sm_120, PCIe 4.0 x16) | **Tier 0 VRAM:** Model weights, active KV pages, BaM queues |
| **CPU** | AMD Ryzen 9 5950X (16 cores / 32 threads, Zen 3, PCIe 4.0 Root Complex) | Root Complex PCIe arbiter (10-bit Tag completer) |
| **Motherboard** | Gigabyte X570 Aorus Elite WiFi (AMD X570 chipset, rev 1.0) | PCIe interconnect fabric & direct CPU lane switching |
| **Host Memory**| 128 GB DDR4 (4 × 32 GB dual-channel DDR4-3200/3600 CL16/18) | **Tier 1 Host DRAM:** 2.37 GB UVA embeddings + 51 GB PLE table |
| **NVMe SSD** | Samsung SSD 990 PRO 2TB (M.2 NVMe PCIe 4.0 x4, NVMe 2.0) | **Tier 2 Storage:** Paged KV slabs & serialized GDN states |

---

## 2. Motherboard Physical Layout & Slot Selection

Physical slot selection is critical on the Gigabyte X570 platform. Installing the SSD in the incorrect M.2 slot will degrade P2PDMA bandwidth by up to 60% and introduce severe Access Control Services (ACS) latency stalls.

```
       +-------------------------------------------------------+
       |             REAR I/O & MOTHERBOARD VRM                |
       |                                                       |
       |                  [ AM4 CPU SOCKET ]                   |
       |                   AMD Ryzen 9 5950X                   |
       |                                                       |
       |  ===================================================  |
       |  || [M2A_CPU SLOT] (Top M.2 under thermal heatsink)|| <--- MUST USE THIS SLOT!
       |  || Direct 4x PCIe 4.0 lanes to CPU Root Complex  ||      (Samsung 990 PRO 2TB)
       |  ===================================================  |
       |                                                       |
       |  +-------------------------------------------------+  |
       |  | [PCIEX16 SLOT] (Steel-reinforced primary slot)   | <--- NVIDIA RTX 5090 FE
       |  | Direct 16x PCIe 4.0 lanes to CPU Root Complex   |      (32GB GDDR7 BAR1)
       |  +-------------------------------------------------+  |
       |                                                       |
       |  [ PCIEX1 Slot ]                                      |
       |  [ PCIEX4 Slot ]                                      |
       |                                                       |
       |  +---------------------------+  +------------------+  |
       |  | [M2B_SB SLOT]             |  | AMD X570 CHIPSET |  |
       |  | DO NOT USE! (Chipset Hop) |  | (Active Fan)     |  |
       |  +---------------------------+  +------------------+  |
       +-------------------------------------------------------+
```

### Critical Slot Rules:
1. **NVMe Storage SSD: Install into `M2A_CPU` (Top Slot)**
   - Located directly between the CPU socket and the primary `PCIEX16` slot, beneath the integrated metal M.2 Thermal Guard heatsink.
   - Wired **directly** to the Ryzen 5950X's internal I/O Die (IOD) PCIe controller.
   - Enables direct peer-to-peer DMA switching inside the CPU root complex at **<700 ns** latency without touching host RAM.
2. **DO NOT USE `M2B_SB` (Lower Slot)**
   - The lower slot is routed through the AMD X570 chipset (southbridge) over an interconnect downlink.
   - P2P traffic traversing the chipset suffers:
     - 3–5× higher latency (~2,400 ns vs ~680 ns).
     - Access Control Services (ACS) packet reflection stalls (`ACSStallRisk == true`).
     - Bandwidth throttling down to <3.0 GB/s due to shared chipset bus contention.
3. **GPU Installation:**
   - Install the RTX 5090 FE into the primary steel-reinforced **`PCIEX16`** slot closest to the CPU.

---

## 3. UEFI / BIOS Configuration (Gigabyte X570)

Power on the machine and press `Delete` repeatedly to enter the Gigabyte UEFI BIOS. Press `F2` to switch to **Advanced Mode**.

Configure the following settings:

### A. PCIe & Memory Aperture Settings (`Settings -> IO Ports`)
- **Above 4G Decoding:** Set to **`Enabled`**
  - *Purpose:* Allocates 64-bit memory address space for PCIe devices above the 4 GB physical boundary. Required for ReBAR.
- **Re-Size BAR Support:** Set to **`Auto`** (or **`Enabled`**)
  - *Purpose:* Advertises the full 32 GB VRAM capacity as a single BAR1 aperture (`BAR1SizeBytes == 34,359,738,368 bytes`) rather than clamping to the legacy 256 MB window.
- **PCIe Slot Configuration (PCIEX16):** Set to **`Gen4`**
  - *Purpose:* Prevents auto-negotiation down to PCIe Gen3 speeds, ensuring the full 16 GT/s line rate (~31.5 GB/s bidirectional).
- **M2A_CPU Link Speed:** Set to **`Gen4`**
  - *Purpose:* Locks the Samsung 990 Pro slot to PCIe 4.0 x4 (16 GT/s, ~7.88 GB/s theoretical ceiling).

### B. AMD CBS / NBIO Options (`Settings -> AMD CBS`)
- **PCIe Ten Bit Tag Support:** Set to **`Enabled`** (`AMD CBS -> NBIO Common Options`)
  - *Purpose:* Extends PCIe tag completer capability from 8-bit (256 tags) to 10-bit (up to 768 non-posted outbound transactions). Essential for high-concurrency BaM NVMe queue submissions.
- **IOMMU:** Set to **`Enabled`** (`AMD CBS -> NBIO Common Options`)
  - *Purpose:* Arms AMD-Vi hardware virtualization for Linux `iommu=pt` pass-through mapping.
- **ACS Enable:** Set to **`Auto`** or **`Disabled`** on Root Ports
  - *Purpose:* Avoids forced redirection of peer-to-peer PCIe packets up to the OS kernel.

### C. Boot & Compatibility (`Boot`)
- **CSM Support (Compatibility Support Module):** Set to **`Disabled`**
  - *Crucial:* Resizable BAR cannot operate in legacy CSM / BIOS mode; pure UEFI boot is strictly required.

Press **`F10`** to Save & Exit.

---

## 4. Linux Kernel & Driver Configuration

Boot into your Linux distribution (Ubuntu 24.04 LTS / Debian 12 / Rocky Linux 9 with kernel 6.6+ recommended).

### A. Kernel Boot Parameters (`/etc/default/grub`)
Edit `/etc/default/grub` using `sudo nano /etc/default/grub`:

```ini
# Add iommu=pt, amd_iommu=on, and pcie_p2pdma=1
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt amd_iommu=on pcie_p2pdma=1"
```

- **`iommu=pt`**: Sets IOMMU to pass-through mode. Direct hardware accesses bypass virtualization page tables, reducing P2P latency to sub-700ns while allowing DMA-BUF peer mapping.
- **`amd_iommu=on`**: Ensures AMD Zen 3 IOMMU drivers initialize properly.
- **`pcie_p2pdma=1`**: Activates kernel-level PCIe peer-to-peer DMA support across memory tiers.

Update the GRUB bootloader:
```bash
sudo update-grub
# On Fedora / RHEL:
# sudo grub2-mkconfig -o /boot/grub2/grub.cfg
```

### B. NVIDIA Kernel Module Options (`/etc/modprobe.d/nvidia.conf`)
Create or edit `/etc/modprobe.d/nvidia.conf`:

```ini
# Enable large Resizable BAR aperture in the NVIDIA kernel driver
options nvidia NVreg_EnableResizableBar=1

# Open GPU kernel module firmware configuration
options nvidia NVreg_EnableGpuFirmware=0

# Ensure P2PDMA transactions are permitted on consumer Blackwell hardware
options nvidia NVreg_RegistryDwords="RMDisableP2PDMA=0"
```

Rebuild the initramfs image and reboot:
```bash
sudo update-initramfs -u
sudo reboot
```

---

## 5. Verification & Diagnostics with `fak-dev cuda-gpudirect`

`fak` provides dedicated diagnostic and verification tools within `cmd/fak-dev` to audit hardware topology, verify BIOS flags, benchmark P2PDMA speeds, and validate zero-copy memory invariance.

### Step 5.1: Inspect PCIe Topology & Interconnect Routing
Run the topology prober:
```bash
go run ./cmd/fak-dev cuda-gpudirect inspect
```

**Expected Healthy Output:**
```text
CUDA GPU Direct Topology & Interconnect Prober:
  Host CPU:            AMD Ryzen 9 5950X (Zen 3, 16c/32t)
  Motherboard:         Gigabyte X570 Aorus Elite WiFi
  Host Memory:         128GB DDR4
  CUDA Devices (1 discovered):
    GPU 0: NVIDIA GeForce RTX 5090 FE [sm_120] BDF=0000:09:00.0 VRAM=32.0 GiB BAR1=32.0 GiB (Large BAR (ReBAR enabled))
           Upstream Root Port: 0000:00:01.1 | NUMA Node: 0
  NVMe Storage Devices (1 discovered):
    NVMe 0: Samsung SSD 990 PRO 2TB BDF=0000:01:00.0 Slot=M2A_CPU (Direct CPU lanes) PeakRead=7.45 GB/s
            Upstream Root Port: 0000:00:01.2 | NUMA Node: 0
  PCIe P2P Interconnect Routes (1 evaluated):
    Route 0: DIRECT_CPU_ROOT_COMPLEX | Optimal: true | Latency: ~680 ns | Max BW: 7.1 GB/s
             ACS Status: No ACS stall risk | ReBAR compliant
             - Direct CPU Root Complex P2P route verified (optimal latency ~680ns, bandwidth ~7.1 GB/s)
  Summary ACS Status: Direct P2P Permitted (No ACS redirection stall risk detected on primary route)
```

Verify that:
- `BAR1=32.0 GiB` reports **Large BAR (ReBAR enabled)**.
- Slot reports **`M2A_CPU (Direct CPU lanes)`**.
- Route reports **`DIRECT_CPU_ROOT_COMPLEX`**, `Optimal: true`, `Latency: ~680 ns`, and `No ACS stall risk`.

---

### Step 5.2: Run the Hardware & Driver Audit
Run the automated configuration auditor:
```bash
go run ./cmd/fak-dev cuda-gpudirect audit
```

**Expected Healthy Output:**
```text
CUDA GPU Direct & BaM P2PDMA Hardware Audit:
  Healthy:             true (Verdict: PASS)
  Host CPU:            AMD Ryzen 9 5950X (Zen 3, 16c/32t)
  Motherboard:         Gigabyte X570 Aorus Elite WiFi
  Host Memory:         128GB DDR4
  System Settings & Verification Checks:
    [PASS] Above 4G Decoding          : Required 64-bit PCIe MMIO aperture enabled above 4GB physical memory boundary.
    [PASS] Resizable BAR (ReBAR)      : Full 32GB BAR1 aperture active (BAR1: 32 GiB, VRAM: 32 GiB).
    [PASS] PCIe Ten Bit Tag           : PCIe 10-bit Tag completer enabled on Zen 3 root complex for up to 768 non-posted outbound transactions.
    [PASS] IOMMU / ACS                : Direct CPU Root Complex interconnect route: P2P DMA traffic permitted without ACS redirection stalls.
    [PASS] NVreg_EnableResizableBar=1 : NVIDIA kernel module parameter NVreg_EnableResizableBar=1 active; large BAR mapping established.
```

If any check returns `WARN` or `FAIL`, follow the printed remediation instructions.

---

### Step 5.3: Run the BaM NVMe P2PDMA Microbenchmark
Benchmark direct bus-master DMA transfers between the Samsung 990 Pro and RTX 5090 FE BAR1 memory:
```bash
go run ./cmd/fak-dev cuda-gpudirect bench
```

**Expected Output:**
```text
CUDA BaM NVMe P2PDMA Zero-Copy Microbenchmark Results:
  Device:               NVIDIA GeForce RTX 5090 FE (sm_120)
  Storage Device:       Samsung SSD 990 PRO 2TB (M2A_CPU)
  Route Interconnect:   DIRECT_CPU_ROOT_COMPLEX (PCIe Root Complex)
  Queue Architecture:   CUDA BaM VRAM Submission/Completion Queues (ASPLOS 2023)
  Commands Executed:    32 (16 Reads, 16 Writes)
  Total Bytes Moved:    2097152 (2.00 MB)
  Throughput:           7.12 GB/s
  IOPS:                 118720 IOPS (64 KiB blocks)
  Staging Copies:       0 (zero host DRAM bounce copies)
  Doorbell MMIO Rings:  1
  Zero-Copy Invariant:  VERIFIED (staging_copy_count == 0)
```

The microbenchmark verifies that:
- `Staging Copies: 0` (strictly zero host DRAM bounce buffers).
- `Zero-Copy Invariant: VERIFIED`.
- Sustained throughput exceeds **7.1 GB/s**.

---

### Step 5.4: Execute the Qwen 3.8 Model Coordination Benchmark
Simulate end-to-end model execution for Qwen3.8-27B and Flash Next:
```bash
go run ./cmd/fak-dev cuda-gpudirect qwen38
```

To output the canonical machine-readable receipt in JSON format:
```bash
go run ./cmd/fak-dev cuda-gpudirect qwen38 --json
```

---

## 6. Troubleshooting & Common Pitfalls

| Symptom / Error | Root Cause | Remediation Action |
| :--- | :--- | :--- |
| `BAR1 clamped to 256 MB` / `small BAR bottleneck` | CSM enabled or Re-Size BAR disabled in BIOS | Enter BIOS -> `Boot -> CSM Support -> Disabled`, then `Settings -> IO Ports -> Re-Size BAR Support -> Auto`. |
| `IOMMU / ACS: WARN` / Route is `CHIPSET_DOWNLINK` | NVMe drive installed in `M2B_SB` (chipset slot) | Power down workstation and relocate Samsung 990 Pro SSD into **`M2A_CPU`** (top slot under CPU heatsink). |
| `ACS STALL RISK DETECTED` in topology prober | PCIe Access Control Services redirecting packets | Verify `iommu=pt` in GRUB commandline and ensure drive is in `M2A_CPU`. |
| `NVreg_EnableResizableBar=1: WARN` | Missing NVIDIA module option | Add `options nvidia NVreg_EnableResizableBar=1` to `/etc/modprobe.d/nvidia.conf` and reload driver. |
| Inter-token latency spikes during token generation | Host memory bus contention from DRAM staging | Verify `fak-dev cuda-gpudirect bench` reports `Staging Copies: 0` and zero host bounce copies. |

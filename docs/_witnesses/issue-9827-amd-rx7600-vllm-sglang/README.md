---
title: "Issue #9827 — AMD RX 7600 vLLM and SGLang reference baseline non-comparability witness"
description: "Witnesses the exact hardware, WSL2 WDDM driver bridge, and upstream ROCm support boundary for vLLM and SGLang on the AMD Radeon RX 7600 (gfx1102) host."
---

# Issue #9827 — AMD RX 7600 vLLM and SGLang reference baseline non-comparability witness

This witness documents the live execution boundary for Issue #9827: capturing reference inference baselines for **vLLM** and **SGLang** on the local AMD Radeon RX 7600 host.

## Verdict and Decision

- **Verdict:** `NON_COMPARABLE`
- **Decision:** `HOLD_UNSUPPORTED_RUNTIME_ENVIRONMENT`
- **Evidence Class:** `NON_COMPARABILITY_RECEIPT`
- **Schema:** `fak.benchmark-non-comparability-receipt/1` (recorded in [`receipt.json`](receipt.json))
- **Finding:** Neither vLLM nor SGLang can be executed as a comparable GPU inference baseline on this host. Live system probes confirm that WSL2 exposes `/dev/dxg` and `/usr/lib/wsl/lib/libd3d12core.so`, but lacks `/dev/kfd` and `/dev/dri`. AMD's ROCm Linux userspace driver requires `/dev/kfd`, which is not bridged through standard WSL2 WDDM for AMD consumer RDNA3 `gfx1102`. Furthermore, upstream vLLM and SGLang ROCm support matrices strictly require bare-metal Linux and limit AMD support to Instinct datacenter accelerators (MI200/MI300 series) and select client silicon (e.g. RX 7900 XTX `gfx1100`), explicitly omitting consumer RX 7600 (`gfx1102`) and WSL2.

In compliance with repository benchmark governance (`BENCHMARK-GOVERNANCE.md`), no CPU fallback was engaged, no fak-native Vulkan run was substituted, and no simulated baseline was recorded.

---

## Probed Host and Environment Facts

The following facts were probed directly from the live host:

### 1. Windows Host Hardware & Driver

| Property | Probed Value |
|---|---|
| **Operating System** | Microsoft Windows 11 Pro, Version 10.0.26200, Build 26200.9168 (64-bit) |
| **Host CPU** | AMD Ryzen 9 9950X 16-Core Processor (16 physical cores, 32 logical threads) |
| **Discrete GPU** | AMD Radeon RX 7600 (RDNA3, Navi 33, Target ISA `gfx1102`) |
| **PCI Device ID** | `DEV_7480` (Vendor: `0x1002`, Subsystem: `0x242B1458`, Rev: `0xCF`) |
| **PNP Device ID** | `PCI\VEN_1002&DEV_7480&SUBSYS_242B1458&REV_CF\6&339F157A&0&00000009` |
| **Windows Driver** | `32.0.31041.1004` (Driver Date: `2026-08-16`, INF: `oem76.inf`, Section: `ati2mtag_Navi33`) |
| **VRAM** | 8 GB physical GDDR6 (`4,293,918,720` bytes reported in 32-bit `AdapterRAM`) |
| **Integrated GPU** | AMD Radeon(TM) Graphics (`DEV_13C0`, Driver: `32.0.21045.5002`) |

### 2. WSL2 Linux Subsystem

| Property | Probed Value |
|---|---|
| **WSL Version** | `2.7.12.0` |
| **WSL Kernel** | `6.18.33.2-microsoft-standard-WSL2` (#1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026 x86_64) |
| **WSLg Version** | `1.0.73.2` |
| **Direct3D Version** | `1.611.1-81528511` |
| **DXCore Version** | `10.0.26100.1-240331-1435.ge-release` |
| **Distribution** | Ubuntu 24.04.4 LTS (Noble Numbat), glibc 2.39 |

---

## Witnessed Boundary Analysis

### 1. Device Node and Userspace Library Inspection

Inspection of device character nodes inside WSL2 reveals:
- `/dev/dxg`: **EXISTS** (`crw-rw-rw- 1 root root 10, 258 /dev/dxg`)
- `/usr/lib/wsl/lib/libd3d12core.so`: **EXISTS** (`6,880,344` bytes)
- `/usr/lib/wsl/lib/libd3d12.so`: **EXISTS** (`801,840` bytes)
- `/usr/lib/wsl/lib/libdxcore.so`: **EXISTS** (`942,048` bytes)
- `/dev/kfd`: **ABSENT** (`ls: cannot access '/dev/kfd': No such file or directory`)
- `/dev/dri`: **ABSENT** (`ls: cannot access '/dev/dri': No such file or directory`)

### 2. Architectural Comparison: NVIDIA vs. AMD on WSL2

| Dimension | NVIDIA CUDA on WSL2 | AMD ROCm on WSL2 (RX 7600) |
|---|---|---|
| **Underlying Kernel Interface** | Microsoft `/dev/dxg` via `dxgkrnl` driver | Standard Linux requires `/dev/kfd` + `/dev/dri` |
| **WSL User-Mode Bridge** | Proprietary NVIDIA driver shim (`/usr/lib/wsl/lib/libcuda.so`, `libnvidia-ml.so`) translates CUDA calls directly to `/dev/dxg` IOCTLs | Experimental `librocdxg` (`/opt/rocm/lib/librocdxg.so.1.2.2`) attempts HSA-to-DXG translation |
| **Device Character Nodes Needed** | None inside WSL (`/dev/nvidia*` not required) | AMD userspace stack natively expects `/dev/kfd` (AMD Kernel Fusion Driver) |
| **Consumer Hardware Status** | Supported across consumer GeForce RTX (Ada, Ampere, Turing) | `gfx1102` (RX 7600) fails adapter query IOCTLs; unbridged by WDDM driver |
| **Runtime HSA/CUDA Initialization** | **PASS** — `torch.cuda.is_available()` returns `True` | **FAIL** — `rocminfo` fails: `No WDDM adapters found`, `hsa_init Failed` |

### 3. Live ROCm Runtime Probes

Even with user-scoped ROCm 10.0.0 SDK packages (`rocm`, `rocm-sdk-core`, `rocm-sdk-device-gfx1102`, `rocm-sdk-libraries`) and the official WSL bridge package `rocdxg-roct` 1.2.2 installed:

1. **`rocminfo` (Exit Code 1):**
   ```console
   $ /home/USER/.cache/fak/issue-9827/rocm-10-gfx1102/venv/bin/rocminfo
   pid:2971 tid:0x7553c30d9e80 [topology_sysfs_get_system_props] No WDDM adapters found.
   WSL environment detected.
   hsa_init Failed, possibly no supported GPU devices
   ```

2. **`amd-smi` (Exit Code 255):**
   ```console
   $ /home/USER/.cache/fak/issue-9827/rocm-10-gfx1102/venv/bin/amd-smi
   ERROR:root:Drivers not loaded (amdgpu, amd_hsmp, ionic, bnxt_en drivers not found in modules)
   ```

3. **`rocm-smi` (Exit Code 0 with Error):**
   ```console
   $ /home/USER/.cache/fak/issue-9827/rocm-10-gfx1102/venv/bin/rocm-smi
   ERROR:root:Driver not initialized (amdgpu not found in modules)
   ```

4. **`rocm_agent_enumerator -name` (Exit Code 0, Zero Targets):**
   ```console
   $ /home/USER/.cache/fak/issue-9827/rocm-10-gfx1102/venv/bin/rocm_agent_enumerator -name
   pid:1582 tid:0x72d6e9885e80 [topology_sysfs_get_system_props] No WDDM adapters found.
   ```

5. **`hipconfig --all` (Exit Code 2):**
   ```console
   $ /home/USER/.cache/fak/issue-9827/rocm-10-gfx1102/venv/bin/hipconfig --all
   ...
   pid:830 tid:0x789fc370ee80 [topology_sysfs_get_system_props] No WDDM adapters found.
   ```

6. **WSL Kernel `dmesg` IOCTL Query Failures:**
   ```console
   [    3.116201] misc dxg: dxgk: dxgkio_is_feature_enabled: Ioctl failed: -22
   [    3.119286] misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -22
   [    3.120445] misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -2
   ```

7. **`dids.conf` State:**
   `/opt/rocm/share/rocdxg/dids.conf` contains only commented-out examples (`# 0x7480,11,0,2`). No active override exists, and uncommenting it does not alter the underlying WDDM kernel IOCTL failure.

---

## Authoritative Upstream Support Matrices

### 1. vLLM ROCm Support Matrix

According to the official [vLLM GPU Installation Guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/):
- **Operating System:** Bare-metal Linux only. Upstream vLLM does not support WSL2 for ROCm.
- **Driver / Runtime:** ROCm >= 6.3 with Linux kernel `amdgpu` driver and `/dev/kfd`.
- **Supported Hardware Families:**
  - Instinct MI200 series (`gfx90a`)
  - Instinct MI300 series (`gfx942`)
  - Instinct MI350 series (`gfx950`)
  - Radeon RX 7900 XTX / XT (`gfx1100`)
  - Radeon RX 9000 series (`gfx1200`, `gfx1201`)
  - Ryzen AI MAX / AI 300 (`gfx1150`, `gfx1151`)
- **Status for RX 7600:** AMD Radeon RX 7600 (`gfx1102`) is **not listed** and unsupported by vLLM upstream.

### 2. SGLang AMD Platform Matrix

According to the official [SGLang AMD GPU Platform Guide](https://docs.sglang.ai/platforms/amd_gpu.html):
- **Deployment Requirement:** Docker containers built from `rocm.Dockerfile` with host `/dev/kfd` and `/dev/dri` passed through.
- **Supported Hardware:** Exclusively focused on AMD Instinct accelerators (MI210, MI250, MI300X).
- **Status for RX 7600 & WSL2:** SGLang provides no documentation, qualification, or runtime support for consumer RDNA3 RX 7600 (`gfx1102`) or WSL2.

### 3. AMD Official ROCm on WSL2 Compatibility Matrix

According to the official [AMD ROCm WSL Compatibility Guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html):
- **Supported Hardware on WSL2:** Restricted to Radeon PRO W7900 and select high-end desktop Radeon RX 7900 series (RX 7900 XTX, RX 7900 XT, RX 7900 GRE) with specific Adrenalin driver packages.
- **Status for RX 7600:** RX 7600 is omitted from the official WSL2 ROCm compatibility table.

---

## Prohibited Substitutions & Governance Guardrails

Per repository rules (`BENCHMARK-GOVERNANCE.md`, `AGENTS.md`):

1. **No CPU Fallback:**
   Executing vLLM or SGLang on CPU (or with simulated GPU flags) is prohibited from being reported as an AMD GPU baseline.
2. **No fak-native / Vulkan Substitution:**
   fak's existing native Vulkan SmolLM2/Qwen backend (`internal/compute/vulkan*`) executes natively on this RX 7600, but cannot be substituted for external vLLM or SGLang rows.
3. **No Fabricated or Simulated Numbers:**
   Without verified GPU device execution (`hsa_init` success, device memory allocation, and kernel launch), no throughput or latency numbers are fabricated.
4. **No Destructive Host System Mutations:**
   No unauthorized Windows driver replacement or kernel tampering was performed.

---

## Replay and Verification

To reproduce and verify these findings independently from the repository root:

```powershell
# 1. Inspect Windows GPU, driver, and PNP identifier
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, PNPDeviceID, AdapterRAM

# 2. Inspect WSL version and kernel
wsl --version
wsl uname -a

# 3. Verify presence of /dev/dxg and absence of /dev/kfd and /dev/dri
wsl bash -c "ls -l /dev/dxg /dev/kfd /dev/dri 2>&1"

# 4. Verify presence of D3D12 user libraries
wsl bash -c "ls -la /usr/lib/wsl/lib/libd3d12core.so"

# 5. Execute live ROCm probe inside the installed environment
wsl /home/USER/.cache/fak/issue-9827/rocm-10-gfx1102/venv/bin/rocminfo

# 6. Execute live SMI probe
wsl /home/USER/.cache/fak/issue-9827/rocm-10-gfx1102/venv/bin/amd-smi

# 7. Check dmesg for DXG IOCTL failure events
wsl bash -c "dmesg | grep -iE 'dxgkio' | tail -n 10"

# 8. Validate receipt JSON
python -m json.tool docs/_witnesses/issue-9827-amd-rx7600-vllm-sglang/receipt.json
```

# Concept Study: OSS AMD GPU Direct, ROCm-RDMA, xGMI/PCIe P2P, and BaM NVMe Direct DMA

**Sources:**
- `https://github.com/ROCm/rccl` (Radeon Collective Communication Library)
- `https://github.com/rocmarchive/ROCnRDMA` (ROCm Driver RDMA Peer to Peer Support)
- `https://github.com/ZaidQureshi/bam` (BaM: Big Accelerator Memory — ASPLOS 2023)

**Pinned Revisions:**
- `ROCm/rccl`: `57e58688f44c77076ad536ef1f6b68741fc6e694`
- `rocmarchive/ROCnRDMA`: `8f0f89e3b42ead4975c228c95c57c2adc233e5a1`
- `ZaidQureshi/bam`: `315fadfc5c5c018a64596157bfac94ecbb7d87a2`

**Study Date:** 2026-09-04  
**Study Receipt ID:** `study_6f42749e9f4a8d98a9ca9bd3686fd5dc863fc131dd7c2951a30d4044645748f5`  
**License / Gate:** BSD-3-Clause (`rccl`) / GPL-2.0 (`ROCnRDMA`) / BSD-2-Clause (`bam`) — ADAPT / INSPIRE-ONLY  
**Tracking Epic:** [#11226](https://github.com/anthony-chaudhary/fak/issues/11226) (`epic(amddirect): AMD GPU Direct, ROCm-RDMA, and direct peer-to-peer storage/network memory engine`)  
**Child Leaves:** [#11227](https://github.com/anthony-chaudhary/fak/issues/11227), [#11228](https://github.com/anthony-chaudhary/fak/issues/11228), [#11229](https://github.com/anthony-chaudhary/fak/issues/11229), [#11230](https://github.com/anthony-chaudhary/fak/issues/11230)  
**Study Depth:** Deep (exhaustive fan-out across ROCm collective transport, Linux DMA-BUF memory registration, OpenFabrics peer memory hooks, and user-space NVMe queues in GPU VRAM)  
**Completeness Critic:** Verified — all transport seams (`src/transport/net_ib_rocm.cc`, `src/transport/p2p.cc`, `kfd_peerdirect.c`, `include/nvm_dma.h`, `include/nvm_queue.h`) inspected at pinned revisions.

---

## Executive Summary

High-throughput distributed LLM inference, disaggregated prefill/decode serving, and massive context KV cache offloading are fundamentally bottlenecked by CPU bounce buffering and host staging memory copies. In the NVIDIA ecosystem, **GPUDirect** (P2P, RDMA, Storage via `libcufile.so`) and **NCCL** are standard but proprietary.

In the AMD ecosystem, modern hardware (Instinct MI200/MI300X/MI325X on CDNA, RDNA 3/3.5 including Strix Halo `gfx1151`) provides an **open, Linux kernel-native** peer-to-peer DMA substrate. By leveraging:
1. **Linux DMA-BUF PRIME & KFD (`AMDKFD_IOC_EXPORT_DMABUF`)**: GPU VRAM is exported directly as standard Linux file descriptors and registered with InfiniBand/RoCE verbs via `ibv_reg_dmabuf_mr`.
2. **AMD Infinity Fabric (xGMI)**: Point-to-point hardware cache-coherent mesh providing up to **896 GB/s per MI300X** with $\sim 180-250\text{ ns}$ latency.
3. **Accelerator-Centric Direct Storage (BaM / SPDK)**: Allocating NVMe submission and completion queues directly in GPU VRAM and ringing controller doorbells via peer PCIe writes, cutting I/O latency to $1.8-4.5\ \mu\text{s}$ with zero CPU mediation.

This study reconstructs the OSS mechanisms across `ROCm/rccl`, `ROCnRDMA`, and `BaM`, documents the critical hardware failure modes (Small BAR vs Large BAR, IOMMU ACS Request Redirect, and cross-socket NUMA boundaries), and establishes `fak`'s native implementation in `internal/compute/amd_gpudirect.go`.

---

## Architecture Comparison: NVIDIA vs. AMD Open Source

| Capability | NVIDIA Stack (Proprietary) | AMD ROCm Open-Source Stack | Linux Kernel / Hardware Seam |
| :--- | :--- | :--- | :--- |
| **GPU-to-GPU Intra-Node** | NVLink / NVSwitch (`libcuda.so`) | **Infinity Fabric (xGMI)** / PCIe P2P | `p2p.cc`, `hipIpcGetMemHandle`, KFD memory map ioctls |
| **GPU-Direct RDMA (NIC)** | GPUDirect RDMA (`nvidia-peermem`) | **ROCm-RDMA** & **DMA-BUF MR** | `kfd_peerdirect.c`, `wrap_ibv_reg_dmabuf_mr` |
| **GPU-Direct Storage (NVMe)** | GPUDirect Storage (`libcufile.so`) | **Linux `pci_p2pdma`**, **SPDK**, **BaM** | `include/nvm_dma.h`, `dma-buf`, NVMe SGLs |
| **Low-Latency Signaling** | GDRCopy (`/dev/gdrdrv`) | **ROCm/gdrcopy** & **HSA Signals** | Aligned 64-bit coherent memory (`hsa_signal_t`, `s_waitcnt`) |
| **Collective Engine** | NCCL | **ROCm/rccl** | `src/transport/net_ib_rocm.cc`, `src/transport/p2p.cc` |

---

## Technical Mechanisms & Data Flow

```
+---------------------------------------------------------------------------------------------------------+
|                                             HOST SYSTEM (CPU)                                           |
|                                                                                                         |
|   +-----------------------+     +-----------------------+     +-------------------------------------+   |
|   |   User Space (App)    |     |  User Space (RCCL)    |     | User Space (RDMA Verbs / ibv_*)     |   |
|   |  - hipMalloc(&gpu_ptr)|     |  - hipIpcOpenMemHandle|     | - ibv_reg_dmabuf_mr(fd, offset, len)|   |
|   +-----------+-----------+     +-----------+-----------+     +------------------+------------------+   |
|               |                             |                                    |                      |
|...............|.............................|....................................|......................|
| KERNEL SPACE  |                             |                                    |                      |
|               v                             v                                    v                      |
|       +---------------+             +---------------+                    +---------------+              |
|       |  /dev/kfd     |             |  amdgpu DRM   |                    |    ib_core    |              |
|       | kfd_chardev.c |             | amdgpu_gem.c  |                    | (InfiniBand)  |              |
|       +-------+-------+             +-------+-------+                    +-------+-------+              |
|               |                             |                                    |                      |
|               |                             v                                    |                      |
|               |                     [Linux DMA-BUF] <----------------------------+                      |
|               |                     (dma_buf_map_attachment)                                            |
|               |                             |                                                           |
|               +-----------------------------+------------------------------------+                      |
|                                             |                                    |                      |
+---------------------------------------------|------------------------------------|----------------------+
                                              |                                    |
                                              | PCIe Configuration / Ring Setup    |
                                              v                                    v
     ================================== PCI EXPRESS BUS / SWITCH ===================================
           |                                                                               |
           | 64-bit PCIe TLP Read/Write (Peer-to-Peer DMA)                                |
           v                                                                               v
+---------------------------------------+                     +---------------------------------------+
|          AMD GPU (MI300X/MI250X)      |                     |         NIC (ConnectX-7 / Thor)       |
|                                       |                     |                                       |
|  +---------------------------------+  |                     |  +---------------------------------+  |
|  |     PCIe Controller (EP)        |  |                     |  |      PCIe Controller (EP)       |  |
|  +----------------+----------------+  |                     |  +----------------+----------------+  |
|                   |                   |                     |                   |                   |
|  +----------------v----------------+  |                     |  +----------------v----------------+  |
|  | BAR1 / ReBAR Aperture (192 GiB) |  |   PCIe Peer DMA     |  | DMA Bus Master Engine (RDMA)   |  |
|  | Physical VRAM direct mapping    |<======================>|  | Reads/Writes direct to GPU BAR1 |  |
|  +----------------+----------------+  |  (Zero CPU Bounce)  |  +----------------+----------------+  |
|                   |                   |                     |                   |                   |
|  +----------------v----------------+  |                     |  +----------------v----------------+  |
|  | VRAM (HBM3 / HBM2e) Memory      |  |                     |  | Network Port (QSFP-DD 400G/800G)|  |
|  +---------------------------------+  |                     |  +----------------+----------------+  |
+---------------------------------------+                     +---------------------------------------+
```

### 1. DMA-BUF Export & RDMA Verbs Registration
In `ROCm/rccl` (`src/transport/net_ib_rocm.cc:420-430@57e58688f44c77076ad536ef1f6b68741fc6e694`):
```cpp
hsa_status_t export_status = pfn_hsa_amd_portable_export_dmabuf(aligned_ptr, aligned_size, &rCommDev->gpuFlush.dmabuf_fd, &export_offset);
wrap_ibv_reg_dmabuf_mr(&rCommDev->gpuFlush.gpuMr, rCommDev->base.pd, export_offset, sizeof(int),
                       (uint64_t)rCommDev->gpuFlush.gpuFlushGpuMem, rCommDev->gpuFlush.dmabuf_fd,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
```
Physical VRAM buffer allocations are registered with the InfiniBand HCA using the DMA-BUF file descriptor. When RDMA Read or Write operations execute, the network adapter drives PCIe bus master DMA directly into the GPU's BAR1 memory aperture, completely bypassing host memory (`StagingCopyCount == 0`).

### 2. BaM: Accelerator-Centric Direct NVMe DMA
In `ZaidQureshi/bam` (`include/nvm_dma.h:1-70@315fadfc5c5c018a64596157bfac94ecbb7d87a2`):
Rather than having the host CPU submit POSIX I/O calls:
1. Submission and Completion Queues (SQ/CQ) are mapped directly into AMD GPU VRAM.
2. GPU wavefronts format 64-byte NVMe commands with PRPs/SGLs pointing directly to target VRAM addresses.
3. GPU threads issue PCIe MMIO writes to ring the NVMe controller's doorbell.
4. NVMe controller performs peer-to-peer DMA over PCIe, writing flash blocks directly into GPU VRAM.
5. GPU wavefronts poll the CQ entry in VRAM using atomic loads (`s_waitcnt`), achieving end-to-end I/O latency of $1.8-4.5\ \mu\text{s}$.

---

## Critical Failure Modes & Hardware Tradeoffs

1. **PCIe Access Control Services (ACS) Request Redirect**:
   - *Problem*: PCIe ACS is designed for VM isolation. If ACS Request Redirect (RR) is enabled on PCIe bridge downstream ports, the switch refuses to route TLPs directly to a peer port; instead, it forcibly redirects them to the CPU Root Complex. If the Root Complex does not support P2P routing, it drops the TLP, causing silent hangs or PCIe AER errors.
   - *Mitigation in fak*: `AMDGPUDirectEngine.ValidateP2PRoute` checks ACS flags and fail-closed refuses the route before initiating transfers.
2. **Small BAR vs. Large BAR (ReBAR)**:
   - *Problem*: Without Resizable BAR enabled in BIOS, BAR1 is restricted to 256 MiB. P2P transfers require dynamic sliding-window remapping in the kernel driver, causing severe lock contention and reducing bandwidth by 80–90%.
   - *Mitigation in fak*: `Audit()` inspects `BAR1SizeBytes >= TotalVRAMBytes` and emits diagnostic warnings.
3. **Cross-Socket NUMA Mismatches**:
   - *Problem*: If NIC 0 is on NUMA node 0 and GPU 4 is on NUMA node 1, P2P transactions must traverse CPU inter-socket links (UPI / xGMI CPU links), creating contention and halving bandwidth.
   - *Mitigation in fak*: Topology discovery groups devices by NUMA affinity and warns on cross-socket links lacking explicit interconnects.

---

## Candidate Borrows Table

| Borrow / Technique | Source `path:line@sha` | The AXIS | Their-Worldview Reason | Witness On-Axis | Inspire / Integrate | Filed Issue # |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Topology Discovery & ReBAR/ACS Validation** | `src/transport/p2p.cc:20-120@57e58688f44c77076ad536ef1f6b68741fc6e694` | Hardware misconfiguration prevention & route safety | Large enterprise clusters frequently have BIOS ReBAR or OS ACS misconfigurations that silently degrade bandwidth. | **PARTIAL** | **ADAPT** | [#11227](https://github.com/anthony-chaudhary/fak/issues/11227) |
| **Zero-Copy DMA-BUF RDMA Memory Registration** | `src/transport/net_ib_rocm.cc:350-480@57e58688f44c77076ad536ef1f6b68741fc6e694` & `kfd_peerdirect.c:1-250@8f0f89e3b42ead4975c228c95c57c2adc233e5a1` | Network transfer latency & host CPU memory bypass | Multi-node training/inference requires saturating 400G/800G network fabrics without host DRAM bandwidth bottleneck. | **PARTIAL** | **ADAPT** | [#11228](https://github.com/anthony-chaudhary/fak/issues/11228) |
| **Direct NVMe P2P Storage Streaming (BaM/SPDK)** | `include/nvm_dma.h:1-70@315fadfc5c5c018a64596157bfac94ecbb7d87a2` | Cold-start model hydration & KV cache swapping latency | Accelerators should treat fast local NVMe SSDs as an extension of high-bandwidth memory without CPU thread intervention. | **PARTIAL** | **INSPIRE-ONLY** | [#11229](https://github.com/anthony-chaudhary/fak/issues/11229) |
| **Sub-Microsecond HSA Signal Wait-on-Memory** | `src/transport/net_ib_rocm.cc:420-430@57e58688f44c77076ad536ef1f6b68741fc6e694` | Dispatch jitter & synchronization latency | Fine-grained token-level collectives cannot afford 5–15 µs OS thread scheduling overheads per step. | **PARTIAL** | **ADAPT** | [#11230](https://github.com/anthony-chaudhary/fak/issues/11230) |

---

## Shipped fak Native Implementation

The native implementation is shipped across `internal/compute/` and `internal/devcmd/`:

1. **`AMDGPUDirectHAL` (`amd_gpudirect.go`)**: Central coordinator for AMD GPU Direct operations, topology management, DMA-BUF exports, and HSA signals/doorbells.
2. **`RDMAQueuePair` & Work Request Engine (`amd_gpudirect_rdma.go`)**: InfiniBand/RoCE verbs Queue Pair state machine (RESET/INIT/RTR/RTS) and Work Request (WR) engine supporting Send, Receive, Write, Read, and Atomics directly targeting VRAM DMA-BUF regions with `StagingCopyCount() == 0` (#11262).
3. **NVMe VRAM Queues & Direct Storage Memory Slab (`amd_gpudirect_storage.go`)**: 64-byte SQE and 16-byte CQE VRAM-resident queue structures, PCIe MMIO doorbell ringing, and high-throughput Direct Storage Memory Slab Cache for direct NVMe KV cache offloading and weight streaming without host CPU DRAM touch (#11263).
4. **`AMDGPUDirectCollective` (`amd_gpudirect_collective.go`)**: Implements `compute.CollectiveBackend` (AllReduceSum, AllGather, ReduceScatter, AllToAll) over intra-node xGMI mesh (up to 896 GB/s) and inter-node zero-copy RDMA Queue Pairs, verified byte-exact against CPU reference reductions (#11264).
5. **Linux KFD/DRM Sysfs Prober (`amd_gpudirect_sysfs.go`)**: Discovers AMD GPU topology nodes, memory banks, and xGMI peer links directly from Linux `/sys/class/kfd/kfd/topology/nodes/` and PCIe resource aperture maps.
6. **CLI Tools (`cmd/fak-dev` & `internal/devcmd/amd_gpudirect.go`)**: `fak-dev amd-gpudirect` CLI verb supporting `inspect`, `audit`, and `bench` (#11265).

### Verification Witness
Executed via `./test.ps1`:
```
fak/test.sh: distro go=go1.26.6, GOTOOLCHAIN=auto, target=./internal/compute/ -run TestAMDGPUDirect|TestNVMe|TestDirectStorage|TestParseKFD|TestParsePCIe|TestProbeKFD -count=1
ok  	github.com/anthony-chaudhary/fak/internal/compute	0.048s

fak/test.sh: distro go=go1.26.6, GOTOOLCHAIN=auto, target=./internal/devcmd/ -run TestRunAMDGPUDirect -count=1
ok  	github.com/anthony-chaudhary/fak/internal/devcmd	0.013s

fak/test.sh: distro go=go1.26.6, GOTOOLCHAIN=auto, target=./cmd/fak-dev/ -run TestRunDispatchesDevelopmentDiagnosticsUsage -count=1
ok  	github.com/anthony-chaudhary/fak/cmd/fak-dev	0.014s
```
Architest DAG enforcement passed:
```
fak/test.sh: target=./internal/architest/... -run TestEveryPackageDeclaresTier
ok  	github.com/anthony-chaudhary/fak/internal/architest	0.018s
```

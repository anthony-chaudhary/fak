# D41-HW-20261028-0310 — four named SM120 component witnesses on aegis-fak (2026-10-31)

Request: `D41-HW-20261028-0310` ([#12640](https://github.com/anthony-chaudhary/fak/issues/12640),
refresh comment 6545784762). This run returns the requested inventory/admission result, a
matching native build at the current source pin, and per-witness physical execution evidence
with checked teardown. It is **not** a six-expert model arm, not FlashMLA, and not a
performance receipt. Machine-readable pins: [manifest.json](manifest.json); raw streams in
[raw/](raw/).

## Inventory and admission (step 1 of the request)

| Field | raider-fak | aegis-fak (admitted) |
| --- | --- | --- |
| SKU / GPU | Gigabyte-class laptop, RTX 5090 Laptop 24 GB | MSI Aegis ZS2 desktop, RTX 5080 16 GB |
| UUID / CC | GPU-2a2a837f-c82d-0f22-4538-e3a063903380 / sm_120 | GPU-3e9fda0a-061a-6eb7-26c3-541bf8489090 / sm_120 |
| OS / kernel | Arch Linux 7.2.11-arch1-1 x86_64 | Arch Linux 7.2.11-arch1-1 x86_64 |
| Driver / toolkit | 615.78.08 / no nvcc on PATH | 615.78.08 / CUDA 13.4 V13.4.92 at /opt/cuda |
| Go / host compiler | go1.27.3 / gcc 16.2.1 | go1.27.3 / gcc 16.2.1 |
| VRAM total / free | 24463 MiB / **1204 MiB** | 16303 MiB / **12692 MiB** |
| Existing consumer | `/usr/local/bin/fak` peer process holding 22862 MiB | `/usr/local/bin/fak` serving service, 3216 MiB |
| RAM / disk | 30 GiB (8 available) / 51 GiB free of 953 G (95%) | 30 GiB (24 available) / 1.4 TiB free of 1.9 T |
| State | UP, degraded: open Xid 31 MMU recurrence; main-tip boot regression (#14118) | ACTIVE serving node, healthy |

Admission: **aegis-fak admitted** via share lease `hwl-9d0e2ccd4da8bbba`
(`fak-sync nvlease acquire --class owned-rtx5080-aegis`, kind=share, epoch 119; released
after teardown, verdict ok). **raider-fak BLOCKED for this campaign**: actual free VRAM
1.2 GiB leaves no headroom above the ≥1 GiB steady-state floor after any allocation, the
Xid 31 MMU fault recurrence wedges CUDA contexts until process restart, and the peer
process is foreign (not killable under the stop rules). The control workstation's own GPUs
(RTX 4070 Laptop sm_89 internal; dual RTX 5070 Ti eGPUs dark after the 2026-10-30 USB4
power loss, recovery goal separate) are not sm_120 or unavailable. Concurrency stayed 1;
the effective cap was min(24,000,000,000 bytes, actual free) = actual free throughout.

## Build (step 2)

Isolated clean checkout of public fak at `42d81609391aaf274783e70f3b107507b8180a8c`
(verified ancestor of origin/main; 0 dirty paths; source tar SHA256
`64c78b387d48c91022dfc2bdd1d92caf3a0ac880950d5302f64e3f188b03966e`). Built with the exact
published profile (`--arch sm_120 --v41-defaults=false --q2k-mmq-prefill
--q3k-mmq-prefill --q8-mmq-prefill --v41-compressor --v41-compressor-pending
--q4k-qkv-qualification --q4k-qkv-async --v41-engram-iq4nl-qualification --nccl=false
--smoke=false`, verified OS-native nvcc/gcc/g++ supplied): outcome success in 64.6 s,
0 warnings/spills, SASS sm_120 only, `libfakcuda.a` SHA256
`f7e715384ec95161d0ff707f0c15910d151f9faf3e095c5f3e83718d273afd8f`. Test executable linked
with the published nine-tag set; builder-resolved cgo environment propagated
(`CGO_CFLAGS=-I/opt/cuda/include`, `CGO_LDFLAGS=-L/opt/cuda/lib64 -Wl,-rpath,/opt/cuda/lib64`;
NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS/GOFLAGS cleared); executable SHA256
`c6a0997a22e63c60f9a8e1236dd559a4bcd969a826fd0c35c64b16a1d921491a`.

## Per-witness results (each in its own admitted process, JSON evidence in raw/)

| # | Witness | Result | Evidence |
| --- | --- | --- | --- |
| 1 | `TestCUDAV41CompressorRingMatchesPool` | **PASS** (run=1 pass=1 fail=0 skip=0, rc=0) | raw/w1.json `ac6897…` |
| 2 | `TestCUDAV41CompressorRingRefusesNonFinite` | **PASS** (run=1 pass=1 fail=0 skip=0, rc=0) | raw/w2.json `ccef66…` |
| 3 | `TestCUDAV41EngramIQ4NLRows` | **PASS** (run=1 pass=1 fail=0 skip=0, rc=0) | raw/w3.json `705242…` |
| 4 | `TestCUDAQ4KAsyncOwnedPhysical` (FAK_CUDA_Q4K_ASYNC_ACCEPT=1, FAK_CUDA_Q4K_DECODE=mmvq before startup) | **FAIL** (run=1 fail=1 skip=0, rc=1, deterministic across 2 runs) | raw/w4.json `abf774…` |

Witness 4 failure detail: `Q4 output[1,1]=383.97656 reference=384` — absolute delta
0.0234375 against the witness's fixed 0.02 bound. Diagnosis: **bound selection, not a port
defect or device anomaly.** The Q4_K MMVQ arm's established gate is relative
(`maxAbs <= 1e-2 * max|y|`; the sibling test states "the q8_1 activation error scales with
|y|, so an absolute bound does not fit"). Observed relative error 6.1e-5 matches the arm's
witnessed behavior, and on this same device/build the established gates pass:
`TestCUDAQ4KDecodeArmMMVQMatchesRef` PASS (cosine ≥ 0.99999990, argmax exact, tier sm_120)
and `TestCUDAQ4KDecodeArmMMVQMatchesInt8Model` PASS (port matches CPU int8 replay). The
fixture's expected magnitudes reach 768, where ~6e-5 relative is ~0.047 absolute — beyond
any sub-0.047 fixed bound. Expectation: identical FAIL on any sm_120 chip at this source
pin. No local relaxation of the witness was performed; it stands as authored.

## Teardown (checked)

No owned GPU processes remain; post-run memory 3234 MiB used / 12692 MiB free — identical to
the pre-run baseline; only the pre-existing peer serving process remains. Lease released
(verdict ok). Runner script retained in raw/run-witness.sh for re-execution.

## Remaining bounded work (unchanged by this run)

- Six-expert model arms (G0/G4/G+, optional U0/U4, protected-fabric F): still require their
  own admission, model acquisition and loader prerequisites; nothing here touches them.
- Direct FlashMLA: stays on the #13993 route with actual SM100a/SM103a hardware.
- Witness 4 disposition: either relax its bound to the arm's documented relative contract or
  tighten the kernel; that is a source change at public fak, then a re-run of the full
  four-witness profile at the new pin.

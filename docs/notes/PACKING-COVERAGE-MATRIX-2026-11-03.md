# Loader packing coverage matrix — baseline (2026-11-03)

Companion to [`docs/data/packing-coverage.json`](../data/packing-coverage.json)
(schema `fak.packingmatrix.v1`). This note explains the table and fixes it as the
**baseline metric** for the tensor-layout-mismatch prevention campaign
(layout-guard lane, ticket `lg-packing-matrix`).

## The issue class

High-performance quantized kernels expect weights (and their scale side-channels)
pre-packed or swizzled in specific tile layouts at load time. Porting the compute
kernel without the exact loader packing logic forces runtime transposes/repacks on
the inference hot path, which erase the kernel speedup. The matrix records, for
every shipped quantized kernel family x device, whether fak's loader/upload path
delivers the kernel's expected layout **at load time** or pays for it **per call**.

## Mode semantics (the three-valued verdict)

| mode | meaning | campaign action |
|---|---|---|
| `native` | The kernel's exact byte layout is produced at load/upload time: a byte-identical copy of the checkpoint payload, or a one-shot load-time pack/transpose/narrow that is cached and never re-executes on the hot path. | Verified safe; keep the file:line evidence alive against refactors. |
| `runtime-repack` | A transpose/swizzle/repack of weight or scale bytes runs on the per-forward/per-token hot path. | Defect cell: name the fix (amortize into a resident buffer packed once at upload) and open the follow-on ticket. |
| `unknown` | Not determinable from committed code. Every unknown cell must name the missing probe. | Probe first; never guess. |

One-shot H2D conversions (device-side F32->Q8_0 narrowing, F32->F16 narrow,
host-side AWQ transpose, WKV scale-tile packing) count as `native`: they run once
per weight at load, are cached, and do not touch the decode/prefill hot path. The
campaign's enemy is repack-per-call, not conversion-at-load.

## The table (summary)

Device axis: `vulkan / cuda / cpu` per the ticket. 30 cells; full evidence in the JSON.

| family | cpu | cuda | vulkan |
|---|---|---|---|
| q4_k | native | native | native |
| q5_k | native | native | native |
| q6_k | native | native | native |
| q3_k | native | native | native |
| q2_k | native | native* | native |
| q8_0 | native | native | native |
| iq_raw (IQ1_M..IQ4_NL) | native | native | native |
| nvfp4 | native (oracle only) | **runtime-repack** | excluded (no path) |
| fp8_block32_e8m0 | native (scalar oracle) | native | excluded (no path) |
| mxfp4_e8m0 | native (scalar oracle) | native | excluded (no path) |
| awq_int4_group | native (load-time transpose) | excluded (host-resident) | excluded (host-resident) |
| gptq_int4_group | native | excluded (host-resident) | excluded (host-resident) |
| f16_hgemm (non-quant seam) | excluded (F32 SGEMM) | native (ColMajor branch unused) | excluded (F32-only upload) |

\* adjacent observation, not a layout defect: CUDA Q2_K widens to transient FP32
for Tensor Core GEMM at P>=128 (`k_q2k_dequant_transient`,
`internal/compute/cuda/quant_matmul.cuh:1197`) — a format widening with preserved
element order. Flagged so a bandwidth audit does not misread it as a stride
mismatch.

### The one defect cell: NVFP4 x CUDA (W4A4 route)

The cuBLASLt W4A4 path (`MatMulNVFP4`) re-swizzles **both** the weight scales and
the activation scales into per-call scratch buffers on every matmul call
(`internal/compute/cuda/weight_formats.cuh:370-373`; per-call scratch allocation at
`internal/compute/cuda.go:1053-1059`). Root cause chain: the ModelOpt loader
deliberately normalizes any TensorRT-produced scale swizzle back to the LINEAR
layout at load time (`internal/model/modelopt_nvfp4.go:24-28`), so the device path
rebuilds the TRT-LLM 128x4 tiled layout per call. The W4A16 fallback kernel
consumes linear scales directly (`internal/compute/cuda/quant_matmul.cuh:854-877`)
— the defect is confined to the Lt route. Fix shape: swizzle the immutable weight
scales once at upload and keep them resident; only activation scales need per-call
work. That follow-on is the first work item this baseline authorizes.

### Latent risk flagged, not counted: ColMajor F16 H2D repack

`uploadF16` carries a ColMajor transpose-repack branch at H2D
(`internal/compute/cuda_upload.go:411`), landed with #484 (`f6eb11d4fc`). Rg
witness at audit head: zero production callers set `Layout=ColMajor` — tests only.
The cell stays `native`; the branch is recorded so a future ColMajor adoption lands
with a load-time pack instead of discovering the H2D branch under load.

## How to read this as the campaign baseline metric

1. **Denominator**: every (family, device) pair with a shipped kernel path is a
   cell; pairs without a path are listed in `excluded_pairs` with the reason, so
   adding a new device path later shows up as a NEW cell, not a silent gap.
2. **Metric**: fraction of cells `native`. Baseline at `9bd0f1516a`:
   **29/30 = 96.7%**, with the single `runtime-repack` cell above.
3. **Ratchet**: any refactor that moves a load-time pack onto the hot path flips a
   cell to `runtime-repack` and regresses the ratio — the same drift-gate idea the
   host-overhead stack uses (`glue-census`, `h2d-inventory`; see
   [FAK-HOST-OVERHEAD-PREVENTION-INDEX-2026-11-03.md](FAK-HOST-OVERHEAD-PREVENTION-INDEX-2026-11-03.md)).
   Re-audit method is the ticket's: rg the Upload dispatches + loaders, diff the
   cells.
4. **Evidence discipline**: SW-VERIFIED only — every cell cites file:line at the
   audit head plus landed SHAs where the seam was introduced (77bf256c92 #4843 raw
   k-quant H2D, f6eb11d4fc #484 HGEMM, 3d34127991 NVFP4 W4A4, 4a47ea7cb4 #12640
   deepgemm WKV, 8ce840c298 #5285 bench-only Q4_K repack A/B). No hardware numbers
   are claimed anywhere in this baseline; the missing physical witness for the
   NVFP4 defect (swizzle cost measured in a perf receipt) is named in
   `known_gaps`, not guessed.

## Scope notes and open follow-ons

- **Metal and ROCm are out of scope for v1** (ticket fixed the axis to
  vulkan/cuda/cpu). Missing probes: the Metal projection-upload seam
  (`internal/compute/metal_q4k_projection_upload_on.go`) and the ROCm backend
  Upload dispatch. Extend the device axis in a follow-on ticket.
- **No MBL-\* lesson IDs exist in the public repo** at audit head (rg witness);
  PHL-051/PHL-127 are cited as sibling-campaign context only. If the MBL ledger
  lives in the fak-private companion, cross-link there.
- **KV-cache Q8 quantization** (`internal/model/kvcache_q8.go`, `kvquant.go`) is
  deliberately NOT a cell: it quantizes fresh activations at runtime by design —
  dynamic quantization of new data, not a static weight-layout mismatch. Different
  issue class; do not fold it into this metric.

Next checkable step: open the NVFP4 W4A4 amortization ticket (swizzle weight
scales once at upload) citing this baseline, and add the perf-receipt probe for
the swizzle cost named in `known_gaps`.

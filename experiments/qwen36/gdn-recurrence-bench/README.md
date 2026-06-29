# GDN recurrence vs projections — device-independent compute ratio (#65 arm A)

Host-runnable witness for the load-bearing number behind the
[#65 CPU-hybrid decision](../metal-gdn-recurrence-decision-2026-06-28.md): *the
Gated-DeltaNet recurrent scan is a tiny fraction of the work, so a GPU scan kernel
(arm B / [#92](https://github.com/anthony-chaudhary/fak/issues/92)) chases the wrong few
percent.* That decision's quantitative anchor — "GDN recurrence ≈ 0.5% of prefill" — was
measured once on an M3 Pro (`FAK_QPROFILE`, status doc §3) and is **not reproducible on a
non-Apple box**. This artifact supplies the **device-independent** half of "benchmark
both": the recurrence-vs-projection **compute ratio** at the real Qwen3.6-27B GDN shapes,
which is exact arithmetic over the layer dimensions and therefore reproducible anywhere.

## Run

```
go run ./experiments/qwen36/gdn-recurrence-bench           # human table
go run ./experiments/qwen36/gdn-recurrence-bench -json     # machine result (result.json)
```

No build tags, no GPU, no model file — pure CPU Go. Verified green on this
`windows/amd64`, `CGO_ENABLED=0` host (`go1.26.3`).

## Result (this box)

| quantity (per token, Qwen3.6-27B linear_attn layer) | value |
|---|---:|
| projection MACs (5 GEMMs, GPU-routed by the CPU-hybrid) | 115,834,880 |
| conv1d MACs (depthwise, K=4) | 40,960 |
| **recurrence MACs (delta-rule scan, 48 heads)** | **3,155,968** |
| **recurrence / projections** (exact FLOP ratio) | **2.725%** |
| recurrence / whole linear_attn layer compute | 2.651% |
| recurrence / total wall-time (pp22, native CPU, f32) | 1.99% |

The FLOP ratio is dtype/device-independent. The wall-time corroboration uses naive f32
matmuls for the projections — Q8/Q4K projections are *faster* per element, so f32
**overstates** projection time, making the reported recurrence fraction a conservative
**lower** bound on its arithmetic share.

## What the number means for the decision

A perfect, zero-cost GPU scan kernel (arm B / #92) could remove **at most** the
recurrence's ~2.7% of the linear_attn layer — and once the MLP GEMMs (≈63% of prefill)
enter the denominator, that share collapses to the ~0.5%-of-prefill the M3 Pro profile
measured. The projections are the ~97% lever and the CPU-hybrid already routes them to the
GPU. So the arithmetic agrees with the on-device profile: **keep the recurrence on the CPU;
do not write a speculative Metal GDN-scan kernel.**

## Companion on-device witness

This directory is the **device-independent compute-ratio** arm. The companion on-device arm
now lives at
[`../metal-gdn-recurrence-m3pro-20260629.json`](../metal-gdn-recurrence-m3pro-20260629.json):
on the M3 Pro resident-Q4_K Metal path at pp22, `[metalprof-hybrid]` measured
`gemm+roundtrip=6051.6 ms` and `rest(recurrence/attn/norm)=669.1 ms`, so projections still
dominate the split. The `rest(...)` bucket is an upper bound on recurrence, not
recurrence-only time. The decode-side cost is still *serialization*, addressed by the
resident forward of #61/#67 — not by a scan kernel (decision doc §4).

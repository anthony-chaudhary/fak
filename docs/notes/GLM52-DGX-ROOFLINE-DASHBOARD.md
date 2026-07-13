---
title: "GLM-5.2 GPU-server roofline dashboard: current vs 80%-target vs ceiling, one row per lane"
description: "A deterministic fold of the committed GLM-5.2 lab run artifacts (experiments/benchmark/runs) against the roofline ceilings, one row per drive lane. CURRENT is measured-only; a lane with no measured artifact is PENDING; ceilings are transcribed roofline estimates from the ceiling note. Regenerate, do not hand-edit."
---

# GLM-5.2 GPU-server roofline dashboard — current vs 80% target vs ceiling

> **Generated, not hand-authored.** This table is folded from the committed run
> artifacts under `experiments/benchmark/runs/by-machine/` by `internal/roofline` (a pure-Go, laptop-composable
> generator — no GPU). Regenerate with
> `ROOFLINE_WRITE_DOC=1 go test ./internal/roofline/ -run TestGenerateRealDashboardDoc -count=1`.
>
> **Honesty contract.** The **Current** column is filled *only* from a measured
> real-753B run artifact — never from the ceiling doc, never inferred. A lane with no
> matching measured artifact is **PENDING** (not zero, not invented). The **Ceiling** and
> **80% target** columns are roofline **ESTIMATES** transcribed from the ceiling note; the
> target is exactly `0.8 × ceiling`.

**Ceiling authority:** [`docs/notes/GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md`](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) — GLM-5.2 on the GPU server 2 / GPU server 3 lab nodes: the theoretical throughput ceiling and the day-scale 80% drive (2026-07-06)

**Folded from 3 GLM-5.2 artifact(s):** 1 lane(s) MEASURED, 3 PENDING.

| Lane | Metric | Node | Current | 80% target | Practical ceiling | Regime | Status | Witness |
|------|--------|------|--------:|-----------:|------------------:|--------|:------:|---------|
| A | single-stream decode | server-3 | **23.4** tok/s | 120–160 | 150–200 | memory-bandwidth bound (batch=1) | MEASURED | `a100-glm52-l1-rowsplit-ab-20260709T234240Z` (-sm layer, batch=1) |
| B | aggregate decode @ concurrency | server-3 | — (PENDING) | 8800–11200 | 11000–14000 | compute-bound (concurrency ~64–128, BF16) | PENDING | — |
| D | prefill | server-3 | — (PENDING) | 8800–11200 | 11000–14000 | compute-bound (~64 GFLOP/token) | PENDING | — |
| G-offload | GLM-5.2 cpu-offload | server-2 | — (PENDING) | host-bound (§3.4) | host-bound (§3.4) | host memory-bandwidth + host GEMM (different law) | PENDING | — |

Ceiling ranges are `lo–hi` (tok/s); the 80% target column is the mechanical `0.8 ×` of each.

## Synthetic kernel-wiring witnesses (NOT the 753B — not counted toward any ceiling)

These runs measure the CUDA decode/prefill *wiring* on reduced synthetic weights. The
artifacts label their own numbers an "optimistic lower bound" on the work the real 753B
MoE must do — i.e. an optimistic *upper* bound on real-model speed. They are listed for
transparency and deliberately do **not** fill any lane's Current column.

| Run | Machine | Synthetic measurement(s) |
|-----|---------|--------------------------|
| `a100-glm52-dsa-decode-ctx-scaling-20260709T233630Z` | a100 | decode-single 26.6 tok/s; prefill 35.9 tok/s |

## Real-model attempts with no usable number (surfaced, still PENDING)

Real GLM-5.2 attempts that were committed as artifacts but produced no clean throughput
(a wedge or an abort). They keep the relevant lane PENDING — a failed attempt is not a
measurement — but are surfaced so the gap is not silent.

| Run | Machine | Why no number |
|-----|---------|---------------|
| `20260627T000000Z-glm52-cpu-wedge` | cpu-server-a | NO usable throughput tok/s obtained on this CPU host. |

## Notes

- **Lanes** are the four current-vs-ceiling rows of the ceiling note §4 (A single-stream
  decode, B aggregate decode, D prefill — all server-3 resident — and G-offload, the
  server-2 cpu-offload path, whose ceiling is a different host-bound law with no numeric
  GPU roofline). The other §6 lanes (E arch/kernel, F GGUF-header ground-truth, G-fit
  smaller-quant resident, H-stock stock-engine baseline, B-cache warm-cache, C harness)
  have no single numeric roofline ceiling and are not rows here.
- The server-2 cpu-offload baseline (0.23–2.62 tok/s) cited in the ceiling note §3.4 comes
  from off-tree 2026-06-25/28 runs; it is **not** a committed artifact under `experiments/benchmark/runs/by-machine/`,
  so this generator does not fill it — G-offload stays PENDING until a real artifact lands.

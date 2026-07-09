# GLM-5.2 DSA — synthetic decode/prefill vs context length (A100, 40 GB)

**Run:** `a100-glm52-dsa-decode-ctx-scaling-20260709T233630Z` · machine `a100` (one card of an 8-GPU datacenter server, sm_80) · backend `cuda`

> ⚠️ **Scope — read first.** These numbers come from `cmd/glmdsatput`, which builds **synthetic** GLM-5.2-*shaped* weights (8 layers, dense FFN, **no MoE**) and benchmarks the native CUDA decode/prefill path. This is a **kernel-wiring + context-scaling witness, NOT the real 753B GLM-5.2 checkpoint.** Because this synthetic 8-layer dense model does far less work than the real 753B MoE, the tok/s here are an **optimistic *upper* bound on real-model speed** (the real model will be slower). The tool stamps this `optimistic-lower-bound` — meaning a lower bound on the *work/latency* the real model must pay. `scope = synthetic-weights;reduced-layers;dense-FFN(no-MoE);optimistic-lower-bound;NOT-the-753B`.

## What this measures

One axis: **context length** (`-decode-prompt`, the prompt fed before decode). Everything else is held fixed — L8 / H2048 / 16 heads / inter 8192 / Q8_0 / **index-topk 256** (MLA + sparse top-k indexer). Two context points were collected on one A100 (40 GB) card.

## Results

| decode-prompt | decode ms/tok | decode tok/s | prefill wall (ms) | prefill tok/s |
|--------------:|--------------:|-------------:|------------------:|--------------:|
| 128           | 37.583        | 26.61        | 3 568.46          | 35.87         |
| 512           | 47.839        | 20.90        | 20 691.61         | 24.74         |
| 2048          | *aborted*     | —            | *aborted*         | —             |

Values are medians over 5 reps (decode = 64 steps). The 2048 point was aborted: O(n²) prefill × 5 reps runs ~15–20 min/point under this un-optimized synthetic kernel, and the GPU was shared.

## Findings

1. **DSA decode is not context-flat.** Per-token decode rose **37.583 → 47.839 ms/tok (+27.3%)** as context went **128 → 512 (4×)**, *despite* top-k=256 sparse selection. That +27.3% is the measured fact. A plausible mechanism — **not** proven by this 2-point run — is that the top-k indexer still scans the full KV each decode step, so decode cost grows with context even though sparsity caps the attention math; treat that as a hypothesis to confirm, not a result.
2. **Prefill degrades super-linearly.** Prefill wall rose **3 568 → 20 692 ms (5.8×)** for a 4× longer prompt; prefill throughput fell **35.87 → 24.74 tok/s** — consistent with O(n²) attention over the prompt.

## Environment

Single A100 (40 GB, sm_80), one card of an 8-GPU datacenter server. At run time a peer job held ~35 GB resident-but-idle (0% util) on **every** card, leaving ~5 GB free — enough for this ~2–3 GB synthetic microbench, and the reason the 32-layer size configs OOM on this node.

## Reproduce

From a fresh checkout on an sm_80 box, build `cmd/glmdsatput` (pulls in `libfakcuda.a`), then per context length:

```bash
cmd/glmdsatput -backend cuda -json \
  -layers 8 -hidden 2048 -heads 16 -inter 8192 \
  -index-topk 256 -decode-prompt <P> -decode-steps 64 -decode-reps 5
```

Sweep `<P>` over `128 512` (add `2048` only if you can spend ~15–20 min/point). Each invocation rebuilds/quantizes the synthetic weights (~23–24 s) before measuring.

## Limitations

- **Two points show direction, not a fitted law.** Treat this as "decode is not context-flat + prefill is super-linear," not as a scaling coefficient.
- **The earlier same-node size sweep is not included here.** Its raw records lived only on the box and were deleted during cleanup; they are deliberately **not** reconstructed from memory.
- Synthetic, 8-layer, dense-FFN — characterizes the CUDA wiring and its context scaling, not model quality or real-model throughput.

_Related prior GLM work: `cpu-server-a/20260627T000000Z-glm52-cpu-wedge` (real 753B, CPU, wedged — negative result)._

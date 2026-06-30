---
title: "GPU server data collection — what runs, how it compares, what's next"
description: "A run log from the datacenter GPU×8 node: the live stock-SGLang Qwen3.6-27B serving numbers measured there, the three real serving/kernel points now on the board with their category boundaries kept honest, why the native fak sweep belongs on the 80GB node, and the prioritized next steps."
---

# GPU server cross-engine data: what runs, how it compares, what's next

_2026-06-25._ A companion run-log to
[GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md).
That note set the honest comparison framework and recorded fak's native kernel
numbers on the 80GB node. This one records **what the 40GB node ("GPU server", datacenter GPU
×8) actually has running**, the **real stock-engine serving number measured on it**,
and the next steps the data points to.

> **Scope & honesty.** The serving number below is a **real engine on a real
> checkpoint**; the fak native number is a **synthetic-weights kernel cost** — they are
> *not* the same category (see §3). Hardware is referred to by class (`a100-40gb-tp8`),
> never by lab host/IP/path, per the public-record scrub discipline.

## 1. What is actually running on the 40GB node

A snapshot of the node found all eight datacenter GPU GPUs ~92% resident (≈37.7 GB/GPU,
util 0% — loaded but idle), held by **one process group**:

- **A stock SGLang server in 8-way tensor parallel** (`sglang::scheduler_TP0..TP7`),
  serving **`Qwen/Qwen3.6-27B`** on `:30000`, `--tp 8 --mem-fraction-static 0.85`,
  `max_model_len 262144`. This is the designated stock-engine comparison server.
- A separate host process staging a different checkpoint (a 120B download) — unrelated
  to the GLM/serving work, noted only so the host-RAM churn isn't mistaken for a leak.

Consequence for fak's own sweep: with ~3 GB free per GPU and all eight busy, the native
`glmdsatput` sweep **cannot run cleanly here**. The 80GB node was fully idle (0 MB/GPU),
so the native kernel curve belongs there — which is exactly where it already ran.

## 2. The number measured on the 40GB node (real serving)

**SGLang stock · `Qwen/Qwen3.6-27B` · datacenter GPU×8 (tp8) · bf16 · 256-token completions:**

| metric | value |
|---|---:|
| single-stream decode | **93.05 tok/s** (median of 3; max 93.1) |
| single-stream latency (256 tok) | 2.75 s |
| throughput @ concurrency 2 | 166.5 tok/s agg |
| throughput @ concurrency 4 | 324.15 tok/s agg |
| throughput @ concurrency 8 | **607.7 tok/s agg** (0 errors) |

Aggregate throughput scales near-linearly to 8 concurrent (≈6.5× the single-stream
rate), with no errors — the server has ample headroom at this request size. This is a
clean baseline data point for "what a stock engine does on this hardware class."

## 3. How the three numbers compare (and why two of them can't be put side by side)

Three serving/kernel numbers now exist on datacenter GPU-class hardware. **Only same-category
rows are comparable:**

| source | node | model / weights | precision | decode tok/s | category |
|---|---|---|---|---:|---|
| **SGLang stock** | 40GB×8 | Qwen3.6-27B (**real**, fits TP8) | bf16 | **93.05** single / 607.7@8 | real serving |
| llama.cpp | 40GB×8 | **GLM-5.2 753B** Q4_K_M (**real**, ~425 GB, CPU-offload) | Q4 | **2.62** single / 4.84@2 | real serving (off-host MoE) |
| **fak native** | 80GB×8 | `glm_moe_dsa` (**synthetic**, dense-FFN, no MoE) | Q8 | 13.5–26.5 | per-token **kernel cost**, fits-one-GPU |

The fak native row (reproduced at/near current HEAD: `16L/2048/topk256 → 13.44 tok/s
decode, 12.95 prefill`; `8L/2048 → 26.53`; small-context `Q8 P=8 → 56.61`) is fak's
device-kernel cost at a reduced synthetic scale. **It is not a serving rate** and must
not be quoted next to either real-serving number — a 26 tok/s synthetic kernel cost vs a
2.62 tok/s 425-GB off-host MoE serve is a model/work/scale category error.

The two **real serving** rows *are* comparable only with the caveat that they serve
different models at different scales (27B dense-ish vs 753B MoE off-host): SGLang's 93
tok/s is a 27B model held entirely in VRAM; llama.cpp's 2.62 tok/s is a 753B checkpoint
streamed off host RAM. The honest takeaway is **not** "SGLang is 35× faster than
llama.cpp" — it's that *model scale and residency dominate*, which is the whole point of
the comparison ladder.

## 4. What fak's number is *for* (the column that matters)

fak's pitch is not raw tok/s — the correctness-first quant kernels are honestly slower
than tensor-core SGEMM. Every comparison row should therefore carry a **correctness
column** and a **safety/reuse column**, not tok/s alone:

- **bit-exact forward** — fak's GLM-5.2 DSA forward is cosine `1.000000`, argmax-exact on
  its own CUDA kernels (the `glm-gpu-witness/1` records). Neither stock engine above
  makes a correctness guarantee.
- the **adversarial capability gate + tamper-evident journal** layered on serving.
- **cross-worker / cross-session KV reuse** (see `SOTA-COMPARISON.md`).

## 5. Next steps (prioritized)

- **P0 — Land the real cross-engine rung (B2).** The 40GB node now has a *live stock
  SGLang server*. The first honest apples-to-apples number is fak vs SGLang vs llama.cpp
  on the **same small real `glm_moe_dsa` checkpoint** (`yujiepan/glm-5-tiny-random`,
  already an oracle here) — held {model, hardware, precision, context, batch} equal,
  reporting decode/prefill/TTFT/throughput@conc **+ a correctness column**. The 27B
  SGLang number above is the stock-engine *capability* baseline; B2 is the
  *same-checkpoint* comparison.
- **P0 (carried) — DSA-kernel illegal-memory-access at the largest synthetic configs.**
  Still the gate on a clean 6-config native curve; single-variable on-box bisection
  (layers/hidden/heads/topk one at a time) to pin the out-of-bounds. Unchanged from the
  companion note.
- **P1 — Make remote benchmark launches reliable.** Two failure modes cost time this
  run and are worth fixing in the tooling: (a) the single-line `base64 -d | bash`
  launcher exceeds the control-channel ~4 KB message limit for any script over ~3 KB and
  silently runs nothing — stage the script in <1.5 KB chunks, then decode+exec; (b) a
  large staged blob floods the readback window so the result scrolls out — read the
  result from a file with a tiny command, never re-echo the blob. Both are now worked
  around by hand; folding the chunked-stage path into the committed runner would make the
  throughput/serving sweeps one-command again.
- **P1 — Pin a free GPU before a native sweep.** The native runner hardcodes GPU 0; a
  transiently-busy GPU 0 produced a false allocation failure earlier. Select the freest
  GPU (`nvidia-smi` min-used) first.
- **P2 — Couple the harness to real weights.** `glmdsatput` uses synthetic weights;
  point the already-arch-blind `cmd/modelbench` at a real `glm_moe_dsa` GGUF for a
  real-weight native tok/s — the rung that turns the synthetic kernel cost into a
  number that survives §3's category test.
- **The wall — 753B native serving (multi-month).** Device NCCL/TP collective +
  MLA-aware sharding + paged experts. Only at that rung does a fak 753B number mean
  anything against the llama.cpp 2.62 tok/s baseline.

## 6. Reproduce

```sh
# The stock-SGLang serving number on the 40GB node (server already running on :30000):
#   single-stream + throughput@{2,4,8} over /v1/completions, 256-token, temp 0.
# (stage the bench script in chunks, exec backgrounded, read the result file)

# The native fak kernel curve on the idle 80GB node:
python tools/dgx_witness_fetch.py dgx3 --runner tools/dgx_glm_throughput_run.sh
# Local single config on any CUDA node:
go run -tags cuda ./cmd/glmdsatput -layers 8 -hidden 2048 -backend cuda -decode-steps 64 -json
```

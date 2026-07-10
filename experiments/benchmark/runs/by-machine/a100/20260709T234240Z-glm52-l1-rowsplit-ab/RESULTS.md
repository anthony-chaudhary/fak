# GLM-5.2 L1 split-mode A/B — `-sm layer` vs `-sm row` (real 753B, 8-GPU datacenter server)

**Issue:** #3075 (L1 tensor/row-split lever) · **Machine:** `a100` (8-GPU datacenter server, sm_80) · **2026-07-09T23:42:40Z**

## Verdict: the L1 "row-split → 3–6× decode speedup" thesis is **REFUTED** for single-stream decode

| split-mode | decode tok/s (median of 5) | prompt-proc tok/s¹ | GPUs busy (sm ≥10%) | per-GPU sm mean |
|---|---|---|---|---|
| `-sm layer` (pipeline) | **23.38** | 59.54 | 8 / 8 | ~9.4% |
| `-sm row` (tensor) | **7.129** | 42.22 | 8 / 8 | ~13.8% |

**`decode_speedup_x = 0.305` → row split is 3.28× SLOWER, not faster.**

¹ prompt-processing throughput (`prompt_toks_median`) at this run's ~12-token prompt — a
prefill-side proxy, not a separately measured long-prompt prefill sweep.

This is the real 753B GLM-5.2 checkpoint (UD-Q4_K_M, ~434 GiB, 11 shards) served by
llama.cpp resident across all 8 GPUs — not the synthetic reduced-layer microbench that the
sibling `glm52-dsa-decode-ctx-scaling` run measures.

## Why the thesis fails

The L1 premise was: `-sm layer` idles all-but-one GPU, so `-sm row` (which shards every
layer's matmul across all cards) would "light all GPUs" and multiply throughput. The data
says the premise is half-right and the conclusion is wrong:

- **Both modes already light all 8 GPUs** (`gpus_busy_ge10pct = 8/8` in both legs). Even
  `-sm layer` keeps every card at ~9–14% sm during sustained decode, because pipeline
  parallelism flows each token through all layer stages.
- **Row's higher per-GPU activity is overhead, not throughput.** Row's mean sm-util is
  *higher* (~13.8% vs ~9.4%) yet its throughput is 3.3× *lower*. For batch=1
  latency-bound decode, `-sm row` forces a cross-GPU all-reduce/all-gather **every token**;
  that collective sync tax dwarfs the tiny per-token compute. `-sm layer` only passes the
  small hidden-state activation across a GPU boundary once per layer.

`-sm row` is designed to help **compute-bound** work (large-batch or long-prompt prefill),
**not** single-stream decode. This A/B isolates single-stream decode, and there row loses
decisively.

## Consequence for the GLM-5.2 epic

- **`-sm layer` (23.4 tok/s) remains the single-stream decode baseline** and is the better
  single-stream config. The presumed dominant lever L1 does not exist for decode on this box.
- The single-stream throughput ceiling has to come from a **different** lever
  (FlashAttention / CUDA-graphs, quantization, speculative decoding, or batching), not from
  `-sm row`. Re-target the epic's "dominant lever" accordingly.

## Caveats (what this run does **not** claim)

- **Single-stream (batch=1) decode only.** Large concurrent batches may shift the layer/row
  tradeoff; not measured here.
- **Small prompt (~12 tokens).** Long-prompt prefill — the compute-bound regime where
  `-sm row` is meant to help — is not measured.
- One quant (UD-Q4_K_M), one backend (llama.cpp), one topology (8-GPU datacenter server). Re-measure
  before generalizing.

## Reproduce

```bash
# on the 8-GPU box (A100 80 GB), from a fresh checkout of origin/main:
SHARD1=<path-to>/GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf \
DEVICES=0,1,2,3,4,5,6,7 PORT=8002 ITERS=5 \
  bash tools/glm52_l1_rowsplit_ab.sh
# -> folds l1-ab-verdict.json (layer leg, row leg, decode_speedup_x)
```

Artifacts: [`l1-ab-verdict.json`](l1-ab-verdict.json) (the fold),
[`bench-layer.json`](bench-layer.json) / [`bench-row.json`](bench-row.json) (per-leg decode
samples + per-GPU sm-util), [`manifest.json`](manifest.json), [`result.json`](result.json).

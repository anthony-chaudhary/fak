# GLM-5.2 in fak's own kernel on the sm_80 datacenter node — load works, load-speed is the open problem (2026-06-25)

Status snapshot for the next session. fak's **own** in-kernel engine (not llama.cpp) now loads
the full 466 GB GLM-5.2 (`glm_moe_dsa`, unsloth UD-Q4_K_M, 11 shards) on the 8× sm_80 datacenter GPU (80GB) box and
binds `/v1/*`. Every loader/fit/inference gap that blocked it is fixed and shipped. The one open
item is **load time** (~100 min) — and the cause is now precisely diagnosed.

## What is fixed and shipped (origin/main)

- **Serve wiring** — `fak serve --gguf <shard1> --backend cuda --cpu-offload-experts
  --context-budget-tokens 8192` (3-file additive wiring: the flag, `gateway.Config.CPUOffloadExperts`,
  `inkernel_planner`). The default 1M context forced a 533 GiB KV plan → `FitTooBig`; pass
  `--context-budget-tokens 8192`.
- **Loader gaps** — the MLA `attn_k_b`(transposed)/`attn_v_b` → `kv_b_proj` merge; the `nextn`/MTP
  head skip (loaders **and** the offload memory-estimate pre-flight).
- **The first-inference panic (root cause)** — serving panicked in `Prefill` at
  `glm_dsa_session.go` (`glmDsaAttentionStep` returned `ok=false`). Cause: `applyGLMMoeDsaConfig`
  derived `QKNopeHeadDim`/`VHeadDim` from `attention.key_length` (576) / `value_length` (512), but
  GLM-5.2 carries **separate per-head MLA dims** under `attention.key_length_mla=256` /
  `value_length_mla=256`. The right per-head dims are `qkNope=192`, `vHead=256`. The 512/512 values
  mis-shaped the KV strides and the `kv_b` split. Fix: prefer the `_mla` keys (origin `c8f3606`).
  Also shipped `FAK_GLMDSA_DEBUG` (names the failing DSA sub-step instead of an opaque panic).
  Verified with the throwaway `cmd/glmcfgdiag` (reads only the metadata shard, seconds, no reload):
  now reports `qkNope=192 vHead=256`.

GLM-5.2 config (from cfgdiag): `NumLayers=79 Hidden=6144 NumHeads=64 IndexNHeads=32 IndexHeadDim=128
IndexTopK=2048 QLoraRank=2048 KVLoraRank=512 FirstKDenseReplace=3`.

## The open problem: ~100 min load is CPU-bound in the loader, NOT disk I/O

Measured, not assumed:

- The model lived on `/projects` which is **NFS**. fak's tensor-by-tensor reads got ~0.07 GB/s and I
  assumed an NFS ceiling — **wrong**. A plain `cp` of all 11 shards NFS → local NVMe
  (`/mnt/sglang_dv3/glm52-q4/`, 1.2 TB free) ran at **2.8 GB/s**. NFS sequential read is fast; fak's
  small per-tensor read pattern is what's slow on NFS.
- But serving **from the NVMe copy** still loads at only **0.12 GB/s**, with `read_bytes` ≈ 0 over a
  12 s window and ~261 threads busy → the load is **CPU-bound**, not disk-bound. The disk was never
  the real limit.
- The CPU cost is the per-tensor **dequant/quantize**. The lean-Q8 path does Q4_K→f32→Q8 per tensor.
  I shipped a resident-Q4_K load (origin `fb81567`: `LoadModelQ4KProfile`, dropped the stale
  `s.Q4K = p.q4k && p.backend==nil` device gate — the cuda HAL serves Q4_K resident via the
  dequant-fused `k_q4k_gemm`) **and** a raw-Q4_K expert split that slices super-blocks with no
  dequant (origin `7d049e2`: `splitGLMMoeDsaExpertsQ4KRaw`, 144 B / 256 weights). Load is **still**
  0.12 GB/s.

### Leading hypothesis (verify first next session)

**UD-Q4_K_M is unsloth's _dynamic mixed_ quant — the experts are not all Q4_K.** The raw-resident
expert split only fires for `info.Type == TensorQ4_K`; any Q6_K / Q5_K expert tensors fall back to
the slow f32 dequant→split. Since the experts are the 417 GB bulk, mixed-quant experts would keep
most of the load on the slow path.

Confirm with: `gguf-dump --no-tensor-data <shard5>.gguf | grep exps` (gguf-dump is at
`/usr/local/bin` on the box). If mixed, extend the raw-resident split to the other K-quants (add
Q6_K=210 B/256w and Q5_K=176 B/256w super-block sizes + `AddResidentQ6K`/`Q5K`, or a generic
raw-resident-by-type splitter keyed on `info.Type`). Second lever: confirm the dequant is actually
parallel across the 261 threads — if it's serial per tensor, thread it.

## Next steps (priority order)

1. Confirm the expert quant histogram (`gguf-dump`).
2. Extend raw-resident split to the mixed K-quants (or fix loader threading) → target ≤10 min load.
3. Serve from the **NVMe** path (`/mnt/sglang_dv3/glm52-q4/...-00001-of-00011.gguf`), not `/projects`
   (NFS). With the MLA dims fixed, the smoke decode should now pass.
4. Run the e2e (`tools/glm52_e2e_after_serve_dgx3.sh`): the #413 serving witness + `fak guard
   --provider openai --base-url http://127.0.0.1:8000/v1 -- claude` against the fak-native endpoint.
   Pass `--model glm-5.2` to label `/v1/models` (today it shows the `--model` default `mock`; the
   in-kernel planner IS active — the tokenizer loads fine, `FromGGML` ok, 154880 tokens / 321649
   merges, `pre=glm4`).

## Operational notes

- **Serve from local NVMe, not network storage.** The first copy of the shards onto a local SSD is a
  one-time sequential read (fast); every load after reads from local disk. (Lab-specific paths, the
  control-bridge session, and the box layout are in the private operator memory, not this doc.)
- The throwaway diagnostics used to pin the config-dim bug — a metadata-shard config dumper and a
  tokenizer-extraction probe — are not committed; re-create them if the next dim question comes up.
  The pattern that paid off: **a cheap metadata-only diag (seconds) found the dim bug without paying
  the full multi-minute reload.**

# GLM-5.2 in fak's own kernel on the sm_80 datacenter node — load works, load-speed is the open problem (2026-06-25)

Status snapshot for the next session. fak's **own** in-kernel engine (not llama.cpp) now loads
the full 466 GB GLM-5.2 (`glm_moe_dsa`, unsloth UD-Q4_K_M, 11 shards) on the 8× sm_80 datacenter GPU (80GB) box and
binds `/v1/*`. Every loader/fit/inference gap that blocked it is fixed and shipped. The one open
item is **load time** (~100 min) — and the cause is now precisely diagnosed.

## Update 2026-06-26 — load-speed levers shipped (on-box re-measure is the open witness)

The two diagnosed causes below are now **fixed in code on `origin/main`**; the remaining work
is to re-measure the 466 GB load on the box.

- **S1 — the load loop was SERIAL (one core of 256).** `QuantModelQ4KProfile` now fans the
  per-tensor read+dequant+split across a bounded worker pool and applies builder mutations
  serially in tensor order, **byte-identical** to a serial load (pinned by
  `TestParallelQuantLoadDeterministic`, workers {1,2,4,8,16} → identical forward logits).
  Tune with `FAK_GGUF_LOAD_WORKERS` (default `min(GOMAXPROCS,16)`). Measured 2.4× on a
  2-channel dev box (memory-bandwidth bound there); the 256-core box has far more aggregate
  bandwidth. (`gguf_parload.go`, commit `1436af2`.)
- **S2 — mixed-quant experts defeated the resident fast path (the dominant cost).** Confirmed:
  only `Q4_K` experts loaded resident; UD-Q4_K_M's `Q6_K`/`Q5_K` experts (a large share of the
  417 GB bulk) fell to the f32 dequant→Q8 round-trip (the multi-GB transient + GC thrash that
  explained "261 threads busy, 0.12 GB/s"). fak now holds `Q5_K`/`Q6_K` experts **resident**
  too (raw bytes, CPU dequant-fused GEMV — the experts are CPU-offloaded, so no GPU kernel
  needed), so the whole expert bulk loads as a **raw byte copy (I/O-bound)**, not dequant-bound.
  Bit-exactness pinned by a hand-derived golden + a GEMV-vs-dequant-ref test; loader routing by
  `TestGLMMoeDsaQ6KExpertsLoadResident`. (`internal/model/quant_kquant.go`, commit `6b9fbc3`.)
- **S4 — no per-path visibility.** The loader now records a per-quant-type resident-vs-dequant
  breakdown, printed to stderr during load (`EmitLoadPathSummary`) and on `/metrics`
  (`fak_model_load_path_{tensors,bytes}` by `quant_type`/`class`/`path`), plus a resident
  `Q5/6_K` row in `ResidentReport`. A nonzero `dequant` row for a large expert quant type is the
  slow-load signal — the diagnosis is now legible in-band, no external `gguf-dump` needed.

## Update 2026-06-27 — WITNESSED on dgx3: 466 GB load ~100 min → **150 s** (3.04 GB/s)

Run on dgx3 (8× A100-80GB, sm_80), `origin/main` `6d727be7`, the staged 466 GB UD-Q4_K_M
checkpoint on local NVMe, via `tools/glm52_load_witness.sh` (`fak serve --gguf <shard1>
--backend cuda --cpu-offload-experts --context-budget-tokens 8192`, `FAK_GGUF_LOAD_WORKERS=64`):

```
BUILD_OK
LOAD_READY 150s (2m30s)  under_10min=YES        # fak_model_load_duration_seconds = 145.5
fak: loading model 100% (1809/1809 tensors, 433.8 GB, 2m23s elapsed, 3.04 GB/s)
fak: load-path breakdown (resident = raw bytes, no dequant; dequant = f32 round-trip):
fak:   Q4_K  expert  resident=38912 (256.5 GB)  dequant=0 (0.0 GB)
fak:   Q5_K  expert  resident=18688 (150.6 GB)  dequant=0 (0.0 GB)
fak:   Q6_K  expert  resident=768   (7.4 GB)    dequant=0 (0.0 GB)
fak:   Q8_0  dense   resident=0                 dequant=712 (16.9 GB)
fak:   F32   dense   resident=0                 dequant=706 (0.5 GB)
```

So **every routed-expert quant type loads resident with zero f32 round-trip** — the 158 GB of
Q5_K/Q6_K experts that used to take the slow path (the ~100-min cause) now copy raw, and the load
is I/O-bound at NVMe speed. Only the small dense set (~18 GB Q8/F32) dequants. A warm-cache re-run
landed in **136 s (3.75 GB/s)**. The `<10-min` load target is **met**.

**Remaining (separate axis):** per-token DECODE is slow — the 753B experts run on the host CPU
under `--cpu-offload-experts` via the correctness-first scalar k-quant GEMV, so a chat smoke can
take many seconds/token (raise `SMOKE_S`/lower `SMOKE_TOKENS` on the witness to confirm a token).
The LOAD is the proven win here; decode throughput is the next perf lever (an int8-SDOT k-quant
GEMV like q4k already has, and/or paging experts to the device).

### Original open-witness note (now closed by the run above)

Re-run the self-staging serve on DGX3 and confirm the load is **< 10 min** with every
routed-expert quant type on the resident path. The expected shape:
`fak_model_load_path_tensors{...,path="dequant"}` ≈ 0 for the expert quant types, and the stderr
load-path summary shows resident for Q4_K **and** Q5/6_K. The remaining serial cost is the small
dense set (attention/router/shared/embed/lm_head); the 417 GB expert bulk is now a raw copy.

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

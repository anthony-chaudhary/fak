---
title: "GLM-5.2 L5 quant sweep — the header-derived columns (resident + active-bytes/token), GPU-free (2026-07-14)"
description: "The header-only half of the L5 decode quant sweep (#3077, epic #3073): the two columns that come straight from the GGUF header — resident bytes (EstimateLoadBytes) and active-bytes/token (RoutedExpertActiveSet.ActiveBytesPerToken) — filled per quant arm with NO serve, NO GPU, NO 400 GiB stage. The UD-Q4_K_M arm is DERIVED from the already-witnessed header; the leaner arms are PENDING one header dump each. tok/s + quality remain the #3077 serve witness. The accounting is unit-tested by internal/ggufload/gguf_glm_quant_sweep_test.go."
---

# GLM-5.2 L5 quant sweep — the header-derived columns

> **What this is.** The GPU-free half of the L5 decode quant sweep
> ([#3077](https://github.com/anthony-chaudhary/fak/issues/3077), Lane A of epic
> [#3073](https://github.com/anthony-chaudhary/fak/issues/3073)). The sweep's result table has four
> columns per arm — **resident GiB**, **active-bytes/token**, **decode tok/s**, **quality probe**.
> The first two are *pure header arithmetic*: they read the per-tensor quant recipe out of each
> checkpoint's GGUF header and need no serve. This note fills them (or marks them PENDING one header
> read) so an operator's GPU server time is spent only on the two columns that actually require a GPU.
>
> **What this is NOT.** Not the served sweep. Decode tok/s and the quality probe still require
> staging/serving the ~400+ GiB checkpoints on a GPU-server sm_80 node — see the
> [L5 serve triage (#3077)](GLM52-L5-QUANT-SWEEP-TRIAGE-2026-07-06.md). No tok/s cell here is a
> measurement.

## 1. The accounting — both columns are header-only

For any GLM-5.2 quant checkpoint, header arithmetic gives both columns exactly (no tensor payload
loaded):

| Column | Function | What it reads |
|---|---|---|
| **resident bytes** | `internal/ggufload.Weights.EstimateLoadBytes()` | every tensor's declared `{shape, quant}` → `blockBytes(quant) × blocks`, summed |
| **active-bytes/token** | `internal/ggufload.Weights.RoutedExpertActiveSet().ActiveBytesPerToken` | `K × (routed_band / n_experts) + non_expert_resident` — the routed band scaled by the measured `expert_used_count` (K) plus the replicated non-expert band |

Both track the quant with zero tensor reads: a narrower expert quant shrinks `blockBytes` for the
`ffn_*_exps` tensors, which flows straight into the routed band (and therefore into both columns).
This is proven as a golden in `internal/ggufload/gguf_glm_quant_sweep_test.go`
(`TestQuantSweepHeaderAccounting`): it drives the same `glm_moe_dsa` fixture through a quant ladder
(F32 → Q8_0 → Q6_K → Q5_K → Q4_K → Q3_K) and asserts (a) resident = expert-band + non-expert band
per arm, (b) active-bytes/token = `K × per-expert + non-expert`, and (c) resident shrinks
**monotonically** as the expert quant narrows — the accounting is header-derived, not tensor-read.

Run the columns turnkey on a node with a shard staged:

```sh
# prints config (incl. experts_used=K) + the derived active_set line, header-only, in seconds:
go run ./cmd/q4kdiag -gguf <arm-shard1.gguf> -plan-only
#   active_set … routed_expert_resident=…GiB per_expert=…GiB non_expert_resident=…GiB \
#                experts_used=K active_bytes_per_tok=…GiB active_params_per_tok=…B
```

## 2. The arms

`resident` and `active-bytes/token` below are the **header-derived** columns; `tok/s` and `quality`
are the **serve** columns owned by #3077 (PENDING an operator).

| Arm | resident (header) | active-bytes/token @ K=8 (header, upper bound) | decode tok/s | quality |
|---|--:|--:|--:|--:|
| **UD-Q4_K_M** (shipped) | **433.82 GiB** WITNESSED | **~32 GiB** DERIVED (routed 12.95 + non-expert 19.31) | 23.2 WITNESSED¹ | baseline |
| **UD-Q4_K_S** | PENDING header² | PENDING header² | PENDING serve | PENDING |
| **Q4_K-pure** | PENDING header² | PENDING header² | PENDING serve | PENDING |
| **Q3 (Q3_K_M/S)** | PENDING header² | PENDING header² | PENDING serve | PENDING |

¹ UD-Q4_K_M decode tok/s carried WITNESSED from the 07-01 full-resident serve note (llama.cpp
layer-split, one GPU active/token). ² Each leaner arm's two header columns are one `q4kdiag
-plan-only` away — no serve — once its shard1 is staged; the sweep's monotonic-shrink property is
already proven by the golden, so the operator only needs to *record* the per-arm numbers, not
re-derive the method.

### UD-Q4_K_M — the fully DERIVED arm

The shipped arm's header is witnessed (`glm52-ep-load-plan-witness-2026-06-30.json` +
`glm52-gguf-header-active-set-2026-07-13.json`), so both header columns are closed:

- **resident** = 433.82 GiB (routed band 414.5 GiB across all 256 experts + non-expert 19.31 GiB).
- **per-expert** = 414.5 GiB / 256 = **1.619 GiB**; **routed active/token** = K8 × 1.619 = **12.95 GiB**.
- **active-bytes/token** = 12.95 (routed) + 19.31 (non-expert) = **~32 GiB** upper bound (token-embd
  is a gather, not a per-token sweep, so the true decode stream is ~1–2 GiB lower — the exact figure
  awaits the box-side per-op trace, #3074 half b).

See the [ceiling doc §2](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) and the
[Lane-F active-set note](GLM52-LANE-F-ACTIVE-SET-FROM-GGUF-HEADER-2026-07-06.md) for the full
derivation and cross-checks (per-expert params 2.869 B × 256 + non-expert = 753.86 B, the witnessed
llama-bench total).

## 3. What the sweep buys — corrected lever priority

The active-bytes/token column is *why* L5 matters, and its composition was corrected 2026-07-14
(the earlier "routed is only ~1.87 GiB, so L5 helps little" claim rested on an EP-band÷256 slip and
is **withdrawn**). The true split at K=8:

- **routed experts ≈ 12.95 GiB (~40 %)** — this is exactly what the quant sweep moves. Going
  UD-Q4_K_M → a leaner expert quant shrinks *this* 40 % of the single-stream decode divisor, so L5
  is **high leverage**, not marginal.
- **non-expert band ≈ 19.31 GiB (~60 %)** — attention + shared-expert + dense + output, held at
  high precision (q8_0) across every arm and therefore *unchanged* by an expert quant sweep. This is
  the ceiling of what L5 alone can achieve, and the argument for a *separate* lever that compresses
  this replicated band.

Because the single-stream ceiling scales `1 / active_bytes`, an arm that cuts the routed 12.95 GiB
(e.g. Q3 experts ≈ 0.68× the Q4_K expert bytes → routed ≈ 8.8 GiB → active-bytes ≈ 28 GiB) buys a
~1.15× single-stream headroom from the bytes lever alone — consistent with the ceiling map's
~1.1–1.5× L5 estimate. The exact per-arm numbers land when an operator records the header columns
above and the serve columns from #3077.

## 4. Honesty boundary

Header arithmetic closes the resident and active-bytes/token columns (the latter as an upper bound
pending the decode per-op trace). It does **not** measure decode tok/s, whether UD-Q4_K_S engages
the CUDA resident-Q4_K fast path on sm_80, or output quality — those are the served witness #3077
accepts. Nothing here is a GPU measurement; the UD-Q4_K_M tok/s is the only served cell and it is
carried, cited, from the 07-01 note.

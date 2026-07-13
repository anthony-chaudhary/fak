---
title: "GLM-5.2 Lane F: active-set ground-truth from the witnessed GGUF header (#3073) — three of four roofline inputs pinned, one operator read left (2026-07-06)"
description: "Lane F of epic #3073 replaces the two ESTIMATED roofline inputs (active-params/token, active-bytes/token) with values DERIVED from the already-witnessed GLM-5.2 GGUF header (issue #1728 witness: 79 layers, hidden 6144, 256 experts, 433.82 GiB, exact dtype split). It reconciles the ceiling doc's architecture estimates (which said ~89-92 layers / H≈5120 — inconsistent with the witnessed header) against header truth, derives per-expert resident bytes (1.619 GiB/expert = 414.5 GiB routed band / 256; an earlier 0.2023 GiB EP-band÷256 slip was corrected 2026-07-14), and reduces the whole active-set derivation to ONE unwitnessed scalar: expert_used_count (K, since measured = 8). The estimator already reads K (internal/ggufload applyMoEExpertCounts → NumExpertsPerTok); the only missing action is one header dump on a node that has a shard resident. GPU-free work: no cell here is a served measurement."
---

# GLM-5.2 Lane F: active-set from the witnessed GGUF header

> **What this is.** Lane F of epic [#3073](https://github.com/anthony-chaudhary/fak/issues/3073) —
> "replace the two ESTIMATED model inputs with GGUF-header truth" (ceiling doc §5, L-Lane F).
> The [GPU-server theoretical-ceiling roofline](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) rests on
> two **ESTIMATED** inputs — active-params/token (~32 B) and active-bytes/token (~13 GiB) — and
> **scales inversely with both**. This note pins them from the header that has *already been
> witnessed*, corrects the architecture estimates the roofline used, and reduces the remaining
> unknown to a single scalar an operator can read in one command.
>
> **What this is NOT.** Not a served measurement and not the sweep. It is a header-arithmetic
> increment: it derives from the WITNESSED
> [EP-load-plan witness](../../experiments/glm-gpu-witness/glm52-ep-load-plan-witness-2026-06-30.json)
> (`fak.glm52-ep-load-plan-witness.v1`, issue #1728, 2026-06-30) plus the resident dtype split.
> Nothing here is fabricated; the one input that is *not* in that witness (K) is called out as
> PENDING, not guessed.

## 1. The witnessed header (ground truth we already hold)

From the EP-load-plan witness (`q4kdiag -gguf <shard1> -plan-only`, GGUF header-only, no
tensor payloads loaded) — these are read from the file, not estimated:

| Field | Value | Source |
|---|--:|---|
| `model_type` | `glm_moe_dsa` | GGUF header WITNESSED |
| `block_count` (layers) | **79** | GGUF header WITNESSED |
| `embedding_length` (hidden) | **6144** | GGUF header WITNESSED |
| `expert_count` | **256** | GGUF header WITNESSED |
| tensors / shards | 1809 / 11 | GGUF header WITNESSED |
| resident (monolith) | **433.82 GiB** (465,815,983,104 B) | WITNESSED |
| dtype split | q4_k 256.50 · q5_k 150.56 · q6_k 8.11 · q8_0 18.14 · f32 0.51 GiB | WITNESSED |
| `expert_used_count` (**K**) | **8** | **GGUF header MEASURED on GPU server 2026-07-13** (`glm52-gguf-header-active-set-2026-07-13.json`) |
| `expert_feed_forward_length` | **2048** | GGUF header MEASURED 2026-07-13 |
| `expert_shared_count` | **1** | GGUF header MEASURED 2026-07-13 |
| `leading_dense_block_count` | **3** (→ 76 MoE + 3 dense layers) | GGUF header MEASURED 2026-07-13 |
| MLA ranks (q_lora / kv_lora), heads | 2048 / 512, 64 (1 KV) | GGUF header MEASURED 2026-07-13 |

> **Lane F header read CLOSED (2026-07-13).** `expert_used_count = 8` was read via `gguf-dump` on
> `<resident-gpu-server>:/mnt/sglang_dv3/glm52-q4/GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf` (`go` was
> not on the node's PATH, so the header was dumped directly rather than via `q4kdiag -plan-only`;
> the value is the raw header, tool-independent). The total active-bytes/token still needs the
> decode per-op trace — see §3.

**Per-expert resident bytes (derived, WITNESSED-adjacent).** The routed-expert band across **all
256 experts** is the monolith total minus the replicated (non-expert) rows:
`q4_k 256.50 + q5_k 150.56 + q6_k 7.38 GiB = **414.5 GiB**` (monolith rows of the EP-load-plan
witness; the q4_k/q5_k rows are entirely routed experts). So:

```
per_expert_resident = 414.5 GiB / 256 experts ≈ 1.619 GiB/expert   (UD-Q4_K_M mixed quant)
```

> **Correction (2026-07-14):** an earlier version of this note read **0.2023 GiB/expert** by
> dividing the EP-8 *one-rank* routed band (**51.80 GiB = 32 of 256 experts** on the busiest rank)
> by 256 instead of 32. The correct per-expert is `51.80 GiB / 32 = 1.619 GiB` = `414.5 GiB / 256`.
> Cross-check: per-expert params 2.869 B × 256 + non-expert 19.4 B = **753.86 B**, the witnessed
> llama-bench total. The §3 K=8 section had the same slip via the EP-7 band (59.90 GiB / 256 =
> 0.234); both are corrected below.

This is the load-bearing constant: with K = 8 (measured), active-expert-bytes/token falls straight
out of it.

## 2. Reconciliation — the roofline used estimates that CONTRADICT the witnessed header

The [ceiling doc](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) §2 and the
[decode-path note](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md) built the active-set estimate on
architecture numbers that the header witness (which predates neither) does **not** support:

| Input | Roofline ESTIMATE | Witnessed header | Correction |
|---|--:|--:|---|
| layers | ~89–92 | **79** | over-counted ~13–16% |
| hidden (H) | ≈5120 | **6144** | under-counted ~20% |
| experts | (implicit) | **256** | now pinned |
| K (experts/tok) | ≈8 | **8** (MEASURED 2026-07-13) | pinned — matched the estimate |

The two errors partially offset in the byte estimate (fewer layers × wider hidden), which is
why the ~13 GiB figure was not wildly off — but the roofline should cite **79 / 6144 / 256**,
not the pre-witness estimates. This note is the authority for those three; the ceiling doc
should be updated to source them here (follow-up, §5).

## 3. The active-set derivation — one scalar from done

Active-bytes/token (the single-stream decode divisor, ceiling §3 Ceiling A) is:

```
active_bytes/token ≈ K × per_expert_resident × (moe_layer_fraction)
                   + shared_expert_bytes × moe_layers
                   + attn/dense/router replicated stream
```

**MEASURED (2026-07-13, K = 8; derivation corrected 2026-07-14).** The routed-expert active
stream is `K × per_expert_resident`, where `per_expert_resident = routed_band / 256` already sums
every MoE layer (no extra `moe_layer_fraction` factor). The **routed band is the full monolith
q4_k + q5_k**, which are entirely routed experts:

```
routed_band          = q4_k 256.50 + q5_k 150.56 (+ q6_k tail) ≈ 414.5 GiB   (all 256 experts)
per_expert_resident  = 414.5 GiB / 256 ≈ 1.619 GiB
routed_active/token  = K × per_expert = 8 × 1.619 ≈ 12.95 GiB
   == internal/ggufload.RoutedExpertActiveSet.ActivePerToken (K × RoutedResident/NumExperts)
```

> **Correction (2026-07-14).** An earlier draft of this section put the routed stream at
> ~1.87 GiB by dividing the EP-load-plan witness's **one-rank** band (59.90 GiB, ~37 of 256
> experts on the busiest EP-7 rank) by 256 — a ~7× slip (`0.234` should have been `59.90/37 =
> 1.619`). The correct full-256 routed band is ~414.5 GiB (cross-check: per-expert params
> 2.869 B × 256 + non-expert ≈ 753.86 B, the witnessed total). The "composition inverted /
> expert-tiny" finding that rested on it is **withdrawn**.

**The finding — same order, ~40 / 60, and the estimate was too *optimistic*.** With the
correction:

| Component | Old estimate | Measured-derived (K=8, 2026-07-14) |
|---|--:|--:|
| routed-expert stream | ~10 GiB | **~12.95 GiB** (K=8 × 1.619) — estimate was ~right |
| replicated attn/shared/output band | ~3 GiB | **~19.31 GiB** — estimate under-counted ~6× |
| **total active-bytes/token** | ~13 GiB | **~32 GiB (DERIVED upper bound)** — ~2.5× the estimate |

**The point:** routed experts are **~40 %** of the active stream and the replicated band **~60 %** —
same order, *not* 10:1. Both levers matter: the L5 routed quant-sweep is high-leverage (40 %) **and**
compressing the replicated attn/shared/output band (60 %) helps. Because the divisor is ~2.5×
larger than the estimate, the single-stream ceiling (`tok/s = BW / active_bytes`) is **~2.5× lower**
than the ~13 GiB estimate implied — the raw roofline drops from ~1,230 to **~500 tok/s** (ceiling
doc §3.1). The exact per-token stream (token-embd is a gather, not a sweep, so ~32 GiB is an upper
bound) still awaits the decode per-op trace (#3074 half b). K was the highest-leverage unread scalar
in the whole program.

## 4. The one operator action left (GPU-free otherwise) — ✅ DONE 2026-07-13

> **Done.** The header read below was executed on GPU server against
> `/mnt/sglang_dv3/glm52-q4/GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf`: **`expert_used_count = 8`**,
> recorded in `experiments/glm-gpu-witness/glm52-gguf-header-active-set-2026-07-13.json`. `go` was
> not on the node PATH, so the raw KV was dumped with `gguf-dump` instead of `q4kdiag -plan-only`.
> The recipe below is retained for reproduction.

The estimator **already reads K** — `internal/ggufload.applyMoEExpertCounts` maps
`glm_moe_dsa.expert_used_count` → `cfg.NumExpertsPerTok` (`gguf_config.go:257`). The witness
JSON simply did not emit it. So closing Lane F needs exactly one header read on any node with a
GLM-5.2 shard resident (the EP witness already proved `q4kdiag -plan-only` runs header-only, no
payload, in seconds):

```sh
# on a node with the shard staged (e.g. /projects/glm52-q4/…-00001-of-00011.gguf):
go run ./cmd/q4kdiag -gguf <shard1> -plan-only 2>&1 | grep -i expert_used_count
#   or dump the raw KV:
llama-gguf <shard1> | grep -iE 'expert_used_count|expert_count|expert_feed_forward_length'
```

That one line — `glm_moe_dsa.expert_used_count = <K>` recorded into a witness — closes Lane F.
Everything downstream (the corrected roofline, the L5 quant-sweep target, the 80% number) then
re-derives from measured reality. **No GPU, no serve, no 400 GiB stage — just a header read.**

> **Smallest next step for an operator.** Run the header dump above on GPU server 2 or 3, append
> `glm_moe_dsa.expert_used_count` (and `expert_feed_forward_length` while there) to the
> EP-load-plan witness (or a new `glm52-gguf-header-active-set` witness), and cite it in a
> `(fak docs)` commit against #3073 Lane F. Then update the ceiling doc §2 to `layers=79,
> hidden=6144, K=<measured>` sourced from this note.

## 5. Follow-ups this note opens

- ✅ **Ceiling doc §2 correction** *(done 2026-07-13)* — §2 now reads `79 layers (3 dense + 76 MoE),
  H=6144, 256 experts, K=8, expert_ffn=2048, 1 shared`, sourced to the header witness.
- ✅ **Emit K in the estimator's witness output** *(shipped: `fb7737b36`, `b6fd400f7`)* —
  `cmd/q4kdiag -plan-only` now prints `experts_used=<K>` + `expert_ffn_len` and the derived
  `active_set` line (`internal/ggufload.RoutedExpertActiveSet`). Future Lane F re-derivation is no
  longer blocked on an operator — only on `go` being on the node PATH.
- **Then re-derive the roofline** *(partial)* — the routed-expert active stream is now DERIVED at
  **~12.95 GiB** (K=8 × 1.619; corrected 2026-07-14 from the ~1.87 GiB EP-band÷256 slip), giving a
  ~32 GiB active-bytes/token upper bound; the **exact** per-token stream still needs the **decode
  per-op trace** (#3074 half b, serve-dependent) before Ceiling A's divisor is fully witnessed and
  the 80% single-stream target moves off the upper bound.

*Companions:* [GPU-server theoretical ceiling + lever map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) ·
[decode path to 10 tok/s](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md) ·
[EP-load-plan witness](../../experiments/glm-gpu-witness/glm52-ep-load-plan-witness-2026-06-30.json) ·
[header active-set witness (K=8 MEASURED)](../../experiments/glm-gpu-witness/glm52-gguf-header-active-set-2026-07-13.json) ·
[8-GPU resident serve (23.2 tok/s WITNESSED)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md).

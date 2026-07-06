---
title: "GLM-5.2 Lane F: active-set ground-truth from the witnessed GGUF header (#3073) — three of four roofline inputs pinned, one operator read left (2026-07-06)"
description: "Lane F of epic #3073 replaces the two ESTIMATED roofline inputs (active-params/token, active-bytes/token) with values DERIVED from the already-witnessed GLM-5.2 GGUF header (issue #1728 witness: 79 layers, hidden 6144, 256 experts, 433.82 GiB, exact dtype split). It reconciles the ceiling doc's architecture estimates (which said ~89-92 layers / H≈5120 — inconsistent with the witnessed header) against header truth, derives per-expert resident bytes (0.2023 GiB/expert from the EP-8 routed band), and reduces the whole active-set derivation to ONE unwitnessed scalar: expert_used_count (K). The estimator already reads K (internal/ggufload applyMoEExpertCounts → NumExpertsPerTok); the only missing action is one header dump on a node that has a shard resident. GPU-free work: no cell here is a served measurement."
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
| `expert_used_count` (**K**) | **PENDING** — not captured in the witness | GGUF header, unread |

**Per-expert resident bytes (derived, WITNESSED-adjacent).** The EP-8 plan splits the 256
routed experts across 8 ranks; the routed-expert band is **51.80 GiB across all 256 experts**
(sum of `gguf-ep-routed-expert-shard` rows ×8 ranks). So:

```
per_expert_resident ≈ 51.80 GiB / 256 experts ≈ 0.2023 GiB/expert   (UD-Q4_K_M mixed quant)
```

This is the load-bearing constant: once K is known, active-expert-bytes/token falls straight
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
| K (experts/tok) | ≈8 | **PENDING** | still the one unknown |

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

With the witnessed constants (per_expert_resident = 0.2023 GiB, moe_layers ≈ 79 minus any
dense prefix), the expert stream is **K × 0.2023 GiB × moe_layers-fraction**. The two bracketing
cases:

| If K = | expert stream/token (rough) | vs roofline's ~13 GiB estimate |
|--:|--:|---|
| 8 | ~ K·0.20·(moe frac) → order ~10–13 GiB | consistent with the estimate |
| 4 | ~ half of that | roofline **too pessimistic** → ceiling ~2× higher |

**The point:** the single-stream ceiling (~150–200 tok/s practical) scales `1/active_bytes`, so
whether K is 4 or 8 moves the target by ~2×. This is not a rounding detail — it is the difference
between "5–7× to go" and "3–4× to go." K is the highest-leverage unread scalar in the whole
program.

## 4. The one operator action left (GPU-free otherwise)

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

- **Ceiling doc §2 correction** — replace `~89–92 layers, H≈5120` with the witnessed
  `79 layers, 6144 hidden, 256 experts` and re-source to this note. (Doc edit; do in the same
  pass as the K read so the whole active-set line lands consistent.)
- **Emit K in the estimator's witness output** — `cmd/q4kdiag -plan-only` and the
  `fak.glm52-ep-load-plan-witness.v1` schema read K into `cfg` but do not print it; add
  `expert_used_count` / `NumExpertsPerTok` to the emitted record so no future Lane F re-derivation
  is blocked on an operator. This is the durable fix (the code already has the value in hand).
- **Then re-derive the roofline** — with K measured, Ceiling A's `active_bytes/token` divisor is
  witnessed, and the 80% single-stream target moves off the estimate.

*Companions:* [GPU-server theoretical ceiling + lever map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) ·
[decode path to 10 tok/s](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md) ·
[EP-load-plan witness](../../experiments/glm-gpu-witness/glm52-ep-load-plan-witness-2026-06-30.json) ·
[8-GPU resident serve (23.2 tok/s WITNESSED)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md).

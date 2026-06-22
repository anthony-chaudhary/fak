# Model-arch seam (#487): code-grounded status decomposition

_Tracking epic: **[#487](https://github.com/anthony-chaudhary/fak/issues/487)** — "adapt the in-kernel
fusion past Llama-only to the top-10 model families."_

**Grounded at:** `HEAD = ac44a24` (2026-06-21). Every `file:line` below was opened and read while
writing this doc; nothing here is a restatement of the issue text. This is a **status artifact for a
maintainer**, not an implementation of any fusion stage — no code changed (`go build ./...` green).

---

## TL;DR — the epic thesis is stale; most of the seam already shipped

The epic opens with *"the in-kernel model leaf runs exactly ONE architecture today — a Llama-family
dense GQA pre-RMSNorm SwiGLU decoder."* **On disk at HEAD that is no longer true.** The
`internal/model` leaf carries a full **additive dispatch seam** — a topology axis (`BlockTopology`),
an FFN interface (`ffnKind`), a KV-layout interface (`kvLayout`), ~10 mechanical config axes, a
family-aware tensor-name resolver, fused-tensor split, longrope, sliding-window masking, a GGUF
ecosystem leaf, and a synthetic-fixture MLA path — most of it witnessed by `Float32bits`-equality and
HF-oracle tests that keep the Llama rung byte-for-byte green.

Counts by stage (S1–S8, plus the per-family rows):

| Stage | What it is | Status at HEAD | Owning issue |
|---|---|---|---|
| **S1** | SEAM-0: one `blockStep`, unified RoPE, name-resolver, sharded loader, `Float32bits` gate | **shipped** (1 residual: `batch.go` un-folded) | **#490 CLOSED** |
| **S2** | Mechanical arch axes as config flags | **shipped** (~10 axes, Llama-no-op gated) | folded into #490 |
| **S3** | Sliding-window attention as a read-time mask | **shipped** (option (a); ring-buffer deferred) | folded into #490 |
| **S4** | Block-topology dispatch (post/sandwich/parallel) | **partial** — proof path yes; accel/quant decode no | folded into #490 |
| **S5** | MoE FFN interface (router + experts + weighted sum) | **shipped** (Mixtral/Qwen-MoE/gpt-oss/GLM/MiniMax) | folded into #490 |
| **S6** | Fused-tensor split + longrope | **shipped** (Phi-3/3.5/4; bit-exact evict test) | folded into #490 |
| **S7** | Ecosystem loaders: GGUF + int4/MXFP4 | **partial** — GGUF+Q4_0/Q4_1/Q4_K int4 yes; **MXFP4 not-started** | **#489 OPEN** |
| **S8** | MLA + `kvLayout` interface | **partial** — interface + naive MLA + synthetic fixture; **no real DeepSeek checkpoint** | **#25-gated / #473 scaffold** |
| **S0** | Closed-API transcript adapters (zero-weight tier) | **shipped** | **#491 CLOSED** |
| **ST** | `internal/tokenizer` leaf | **shipped** | **#488 CLOSED** |

**The single load-bearing residual** that cuts across S4/S5/S8: the topology/MoE/MLA-aware path is the
**scalar f32 `blockStep` + cacheless `layer()`** path (the proof/oracle path). The **accelerated
twins** — HAL decode, Metal prefill, the register-blocked quant-batch prefill, and the multi-user
`batch.go` decode — still hardcode Llama PreNorm and `panic` honestly (`requirePreNorm`,
`internal/model/kv.go:522`) rather than silently diverge. Generalizing those hot-path copies is the
real remaining work, and it is exactly what SEAM-0's "fold the 7 hand-copies" was meant to unblock.

**The permanent residual** (epic carries it forever): no correctness claim transfers for free — every
non-Llama family is *asserted, not proven* until it has its own re-exported HF oracle
(**#474**, per-family oracle matrix, OPEN). Several families are config-derived and topology-tested
but lack an on-disk numeric oracle today.

---

## Method & honest caveats

- **The roadmap doc `fak/MODEL-ARCH-SEAM.md` is NOT on disk.** The public-release squash removed it;
  `git log -- '**/MODEL-ARCH-SEAM.md'` returns nothing and the only surviving copy is the goal-prompt
  scratch under `.goal-runs/.../prompts/487-archseam-epic.md`. The `§2a/§3/§4/§6/§7/§8` cross-refs in
  the epic therefore point at a document that no longer exists in the tree (the code comments still
  cite "MODEL-ARCH-SEAM SEAM-0", e.g. `kv.go:519`). **This decomposition is grounded purely against
  on-disk code at HEAD**, which is the stronger evidence anyway.
- **Sibling-issue numbering correction.** The epic body lists `S0..S8 = #17..#26` and `ST = #26`.
  Those are **stale internal-tracker numbers** — they do not map to live GitHub issues. The real
  GitHub owners (verified via `gh issue view`) are in the [issue map](#sibling-issue-map) below. There
  is **no dedicated live GitHub issue for S2/S3/S4/S5/S6/S8**; those stages landed folded into the
  #490 SEAM-0 work or under the family-specific epics (#447 qwen35, #414/#413 GLM-DSA).
- **Every citation is real.** Where a stage cannot be fully ground-truthed from disk (a numeric oracle
  needing a real non-Llama checkpoint), it is marked **`needs-runtime-witness`** rather than guessed.

---

## Sibling-issue map (verified via `gh issue view`)

| Issue | State | Owns | On-disk landing |
|---|---|---|---|
| **#487** | OPEN | the epic itself | this doc |
| **#491** | **CLOSED** | S0 closed-API adapters | `internal/agent/adapters.go` |
| **#490** | **CLOSED** | S1 SEAM-0 (blockStep/RoPE/resolver/sharded loader/Float32bits gate) | `kv.go`, `arch.go`, `tensor_resolver.go` |
| **#488** | **CLOSED** | ST tokenizer leaf | `internal/tokenizer/` |
| **#489** | OPEN | S7 GGUF + int4/MXFP4 loaders | `internal/ggufload/` (Q4_0/Q4_1 landed, commit `13ec795`) |
| **#473** | OPEN | family tensor-name mappings | `internal/model/tensor_resolver.go` (commit `d94d136`) |
| **#474** | OPEN | per-family HF oracle matrix (the permanent gate) | `oracle_test.go` optional/skip tests |
| **#478** | OPEN | arm64 NEON Q8 GEMM tile (perf, not arch) | `quant_*_arm64*` |
| **#447** | OPEN | qwen35 hybrid Gated-DeltaNet epic | `qwen35.go`, `minimax_m3.go` |
| **#414 / #413** | OPEN | GLM-5.2 DSA (exact-span evict / full-size serving) | `glm_dsa.go`, `dsa_index.go` |
| **#86** | OPEN | GLM-DSA has no `compute.Backend` path (`requireGLMDsaSession` panics) | `kv.go:551` |
| **#479** | OPEN | device-resident KVStore (sibling, not a dependency) | `internal/compute` |
| **#33** | OPEN | bit-exact middle-evict under paged/block KV | `kv.go` Evict |

---

## S1 — SEAM-0 ✅ shipped (one residual)

**Owning issue: #490 (CLOSED).** The load-bearing risk-retirement; prerequisite for every fusion stage.

| Sub-item | Status | Deciding evidence | Residual |
|---|---|---|---|
| Single `blockStep` decoder block | shipped | `internal/model/kv.go:701` `func (s *Session) blockStep(l, qpos int, x, cos, sin []float32, mat matKernel) []float32`; called by f32 decode `kv.go:598` and Q8/Q4K decode `quant_forward.go:204` | `batch.go` not routed (below) |
| RoPE unification | shipped | one builder `invFreq` `kv.go:330`; `ropeRow`/`ropeRowForLayer` `kv.go:462`; per-layer theta for Gemma3 via the `layer` arg | — |
| Tensor-name resolver | shipped | `ResolveTensorNames` `tensor_resolver.go:94`; family switch `resolveSpecFor` `tensor_resolver.go:121` (9 families + identity default) | per-family name coverage is the #473 long tail |
| Sharded-safetensors loader | shipped | acceptance gate `sharded_weightsource_test.go` (synthetic 2-shard + `index.json` weight_map, quant-on-load straight to Q8_0) | — |
| `Float32bits`-equality gate | shipped | `TestArchLlamaNoOp` `arch_test.go:62`; `assertFloat32BitsEqual(... prefill/decode ...)` `arch_test.go:72,80`; bit-compare helper `arch_test.go:666` | — |

**Residual (the un-folded 7th block):** `internal/model/batch.go`'s multi-user `stepBatchF32` /
`stepBatchQ` decode does **not** route through `blockStep` (confirmed: `grep blockStep
internal/model/batch.go` → no hits). It is a hand-copy of the Llama PreNorm block, and the HF argmax
oracle drives `Forward/Prefill/Step/Generate` but **not** `BatchSession` — so an arch axis applied to
the 6 folded blocks would silently diverge in batched decode. Folding it is in-scope SEAM-0 cleanup,
not done.

---

## S2 — Mechanical arch axes as config flags ✅ shipped

**Owner: folded into #490.** Each axis is a `Config` flag whose Llama value lowers to the verbatim
Llama op (proven a no-op by the S1 `Float32bits` gate). All helpers live in `internal/model/arch.go`:

| Axis | Evidence (`arch.go` unless noted) | Family it serves |
|---|---|---|
| llama3 rope-scaling (low/high-freq factors) | `applyRopeScaling` `arch.go:26` | Llama-3.x |
| per-projection bias (q/k/v/o, gate/up/down) | `applyProjBias` / `addBiasIfPresent` (`kv.go:764`, `moe.go:85`) | Qwen2, GPT-NeoX |
| qk-norm (per-head RMSNorm on q,k) | `applyLayerQKNorm` `arch.go:205`; witness `arch_test.go:516` (cached-decode==prefill) | OLMo2, Gemma3 |
| norm-gain `(1+w)` | `arch.go:253` (`if cfg.NormGain1p { gain = 1 + gain }`) | Gemma2/3 |
| GeGLU vs SwiGLU | `act` `arch.go:415` (`ActGeluTanh`/`ActGeluErf` → gelu, else silu) | Gemma (tanh), Cohere (erf) |
| attn / final soft-cap (tanh) | `softcap` `arch.go:347`; logit path `logitScaleInPlace` `arch.go:388` | Gemma2 |
| embedding scale `sqrt(d)` | `scaleEmbedInPlace` `arch.go:376` | Gemma |
| final logit scale | `logitScaleInPlace` `arch.go:388` (Cohere `0.0625`) | Cohere |
| per-head attn-scale (`query_pre_attn_scalar`) | `attnScale` `arch.go:275` | Gemma2 |
| alibi score-bias (no RoPE) | `alibiScoreBias` `arch.go:283` | MPT |
| attn output-gate | `kv.go:749` (doubles q_proj, sigmoid-gates) | Qwen3.5/3.6 hybrid |

**Residual:** the gated axes are applied in the scalar `blockStep`/`layer()` paths; the
register-blocked batched-prefill and HAL/Metal twins force the scalar path when any non-default axis is
set (`kv.go:927,1009,1019`), so these axes are **correct but not yet accelerated** on the hot path.

---

## S3 — Sliding-window attention (read-time mask, option (a)) ✅ shipped

**Owner: folded into #490.** Per-layer window as a `pos[]`-keyed lower-bound mask — **not** a ring
buffer (the ring-buffer option (b) that would break the eviction proof is explicitly deferred).

- Config field `Window []int` — `weights.go:131`.
- Per-layer resolution `windowForLayer` `weights.go:773`; lower-bound `windowLoStep` `weights.go:841`.
- Applied in cached attention `kv.go:783` (`lo := windowLoStep(s.Cache.pos, nPos, qpos, cfg.windowForLayer(l))`).
- Witnesses: `TestSWAWindowUnsetIsNoOp` `swa_test.go:30`, `TestSWAWindowMasksOldKeys` `swa_test.go:102`.
- Gemma3 local/global alternation: `SlidingWindowPattern` → per-layer `LayerTypes`/`RopeThetaPerLayer`
  (`weights.go:567` region), witnessed by `TestOptionalGemma3OracleCoversLocalGlobalAttention`
  `oracle_test.go:398` (optional — `needs-runtime-witness: a real Gemma3 checkpoint under .cache/`).

**Residual:** mask applies on the scalar path; SWA + accelerated/batched decode shares the S4 residual.

---

## S4 — Block-topology dispatch ⚠️ partial (proof path shipped; accel/quant decode not)

**Owner: folded into #490.** The topology axis is real and tested on the proof path; the genuine
remaining work is the hot-path copies. This is the **most nuanced status in the epic** — do not read it
as "shipped" or "not-started."

- Enum `BlockTopology` — `arch.go:530` `PreNorm` (Llama, zero value), `:533` `PostNorm` (OLMo2),
  `:536` `SandwichNorm` (Gemma2/3), `:541` `ParallelResidual` (GPT-NeoX/Cohere/Falcon-parallel).
- Single-position dispatch `composeBlock` `arch.go:593`, called in `blockStep` `kv.go:730`
  (parallel-residual MLP-norm handled at `kv.go:725`).
- Sequence/prefill dispatch `composeSeqSublayer` `forward.go:333` (`case PostNorm` `:336`,
  `case SandwichNorm` `:344`), driven by the cacheless `layer()` `forward.go:97`.
- Family derivation: `weights.go:493` SandwichNorm, `:495` PostNorm, `:496` ParallelResidual; helper
  `topologyForFamily` `weights.go:732`.
- Witnesses: `TestSandwichNormUsesDistinctFeedForwardNorms`, `TestParallelResidualDoesNotRequireSeparateMLPNorm`,
  `TestBlockTopologyDiffersFromPreNorm` (`arch_test.go`); config derivation `config_test.go`.

**Residual (the honest boundary):** `requirePreNorm` `kv.go:522` **panics** on any non-PreNorm
topology for the accelerated paths — callers: HAL decode `kv.go:533`, HAL prefill `kv.go:903,982`,
Metal prefill `kv.go:907,991`. So PostNorm/SandwichNorm/ParallelResidual run correctly on the **scalar
f32 `blockStep` + cacheless `layer()`** (oracle) path, but **not** on HAL/Metal or the register-blocked
quant-batch prefill, and **not** in `batch.go`. Generalizing those copies is the open S4 work.

---

## S5 — MoE FFN interface ✅ shipped

**Owner: folded into #490 (interface); family epics #447 (qwen35), #414/#86 (GLM) for the heavy variants.**

- `ffnKind` interface `moe.go:32`; selector `ffnFor` `moe.go:39`; per-layer hybrid selector
  `ffnForLayer` `moe.go:52` (dense layer 0 / sparse layer 1 pattern). Dispatched in the block at
  `kv.go:718`.
- Standard MoE `moeFFN.apply` `moe.go:277`: router logits `moe.go:191`
  (`logits := mat.mul(routerName(layer), xn, E, cfg.HiddenSize)`), full softmax `softmaxOf` `moe.go:248`,
  top-k stable-sort `moe.go:199`, renorm `if cfg.NormTopKProb` `moe.go:214`, per-expert SwiGLU
  `expertSwiGLU` `moe.go:114`, weighted sum `moe.go:290` (`delta[i] += pk.weight * out[i]`).
- gpt-oss variant (top-k **before** softmax + bias + clipped sigmoid-gate): `routeTopKSoftmax`
  routed at `moe.go:193`, expert `moe.go:284`.
- GLM group-routing + shared experts: `glmRoute` `moe.go:317`, `glmMoeFFN` (selected `moe.go:56`).
- MiniMax MoE: `minimaxMoeFFN` (selected `moe.go:60`).
- HF-order witnesses: `TestMoERoutingHandComputed` `moe_test.go:250`,
  `TestGPTOSSRouterUsesTopKSoftmaxAndBias` `moe_test.go:367`.

**Residual:** in-scope target is the small f32 export; the 671B-class flagships are loader/RAM-bound,
not arch-bound. GLM-DSA MoE decode is CPU-resident only (`requireGLMDsaSession` `kv.go:551`, **#86**).

---

## S6 — Fused-tensor split + longrope ✅ shipped

**Owner: folded into #490.** Phi-3/3.5/4.

- Fused split (load-time, contiguous axis-0 byte-range cut): `splitFusedProjections` `fused_split.go:54`
  → `splitOneFused` `fused_split.go:83`, on `self_attn.qkv_proj.weight` (`:28`) and
  `mlp.gate_up_proj.weight` (`:29`). Witnesses: `TestFusedSplitMatchesSeparate` `fused_split_test.go:41`,
  `TestFusedSplitForwardEqualsUnfused` `fused_split_test.go:106`.
- longrope (long/short factor **pinned at session start**, never mid-session): `ropeLongFactor`
  `longrope.go:36`, session-lifetime guard `longropeFactorPinned` `longrope.go:55`. Bit-exact eviction:
  `TestLongropeEvictRepositionBitExact` `longrope_test.go:140`.

**Residual:** none structural for the supported path; mid-session longrope flips remain explicitly
unsupported (they would break the byte-identity eviction proof) — out of scope by design.

---

## S7 — Ecosystem loaders ⚠️ partial (GGUF + int4 shipped; MXFP4 not-started)

**Owner: #489 (OPEN).** The `internal/ggufload` leaf exists and is substantial; #489 is therefore a
*finish/MXFP4* ticket, not greenfield.

| Sub-item | Status | Evidence |
|---|---|---|
| GGUF header/metadata/tensor parse | shipped | `gguf.go` `Read()` (header/KV/tensor directory) |
| Q4_0 / Q4_1 legacy 32-elem dequant (the #489 leg) | shipped | switch `dequantF32` `gguf.go:1686`; `case TensorQ4_0` `:1717` → `dequantQ4_0` `:1825`; `case TensorQ4_1` `:1726` → `dequantQ4_1` `:1844` (commit `13ec795`) |
| Q5_0/Q5_1/Q8_0/Q2_K..Q6_K dequant | shipped | same switch `gguf.go:1735–1812` |
| Resident int4 Q4_K (no f32 round-trip) | shipped | `quant_q4k_loader.go:64` (`ResidentQ4KEligible` → `AddResidentQ4K`) |
| Split/sharded GGUF (`-NNNNN-of-MMMMM.gguf`) | shipped | `gguf_split_test.go`; `OpenWeights` split path `gguf.go:315` |
| GGUF correctness gate | shipped (coherence, not R2/R14) | `coherence_gguf_test.go` (end-to-end ChatML answer; regression for the rotary-unpermute bug) |
| **MXFP4 (FP4 / micro-scaling) dequant** | **not-started** | enum `TensorMXFP4 = 39` `gguf.go:76`, name `gguf.go:120`, but it falls through to the default error `gguf.go:1814` (`"... cannot dequantize to f32 yet"`) — **no handler** |

**Residual:** MXFP4 dequant-on-load is unimplemented (the gpt-oss native quant). int4 is
resident-Q4_K only.

---

## S8 — MLA + `kvLayout` interface ⚠️ partial (interface + naive MLA + synthetic fixture; no real DeepSeek)

**Owner: #25-gated (research); scaffold under #473.** The honest boundary of the whole effort.

- `kvLayout` interface `kvlayout.go:28` (`name` / `cacheStride` / `reconstructKV`); dispatch
  `modelLayout` `kvlayout.go:176` (`if m.MLA != nil` → MLA, else standard) — note there is **no
  `attnVariant` enum field**; the variant is the implicit `MLA != nil` check.
- Standard layout proven byte-identical to the inline `blockStep` attention: `TestStandardLayoutNoOp`
  `kvlayout_test.go:14` (asserts `max|Δ|=0`).
- Naive MLA (low-rank latent KV + decoupled-RoPE key, decompress-then-attend): `MLAConfig`
  `kvlayout.go:84`, write `mlaProject` `kvlayout.go:151`, read `reconstructKV` `kvlayout.go:114`.
  Synthetic fixture + equivalence: `newSyntheticMLA` `kvlayout_test.go:119`,
  `TestMLANaiveMatchesReference` `kvlayout_test.go:154`.
- DeepSeek V2/V3 real-checkpoint path: **scaffold only** — `deepSeekMLASpec` `tensor_resolver.go:425`
  deliberately leaves `perLayer` nil (the q_a/q_b/kv_a/kv_b names need a real manifest to pin);
  `TestOptionalDeepSeekV2OracleDocumentsMLABoundary` `oracle_test.go:459` skips absent the checkpoint.

**Distinct, do not conflate** (all four are separately implemented in the leaf):
- **MLA** (latent KV compression, DeepSeek) — `kvlayout.go`.
- **DSA** (learned per-key indexer + sparse softmax, GLM-5.2) — `glm_dsa.go`, `dsa_index.go`.
- **MSA** (block-level sparse selection on *uncompressed* GQA K/V, MiniMax-M3) — `minimax_m3.go`,
  `msa_index.go`.
- **Linear/recurrent** (Gated-DeltaNet hybrid, Qwen3.5/3.6) — `qwen35.go`, `IsQwen35Hybrid` `:26`.

**Residual / `needs-runtime-witness`:** real DeepSeek-V2/V3 numeric correctness needs a real checkpoint
(no tiny anchor model the way SmolLM2-135M anchors Llama); only the synthetic 2-layer fixture is proven.

---

## Per-family rows (top-10 grid)

Status legend: **proof-path** = correct on scalar f32 `blockStep`/cacheless `layer()` + topology/config
tests; **accel** column = whether HAL/Metal/quant-batch decode is generalized (mostly **no**, gated by
the S4 residual); **oracle** = whether an on-disk re-exported HF numeric oracle exists (mostly **no**,
the #474 permanent gate).

| Family | Resolver spec | Mechanical axes wired | Topology | Proof path | Accel decode | On-disk oracle |
|---|---|---|---|---|---|---|
| **Gemma2** | `gemmaSpec` | sandwich-norm, soft-caps, embed-scale, `(1+w)`, query_pre_attn | SandwichNorm `weights.go:493` | yes | no (S4) | `needs-runtime-witness` |
| **Gemma3** | `gemmaSpec` | + local/global SWA, per-layer RoPE theta | SandwichNorm | yes | no (S4) | optional `oracle_test.go:398` (skip w/o ckpt) |
| **OLMo2** | `olmo2Spec` | qk-norm | PostNorm `weights.go:495` | yes | no (S4) | `needs-runtime-witness` |
| **Cohere / Command-R** | `cohereSpec` | logit-scale `0.0625`, LayerNorm, GeGLU-erf | ParallelResidual `weights.go:496` | yes | **no — panics** `kv.go:522` | `needs-runtime-witness` |
| **GPT-NeoX** | `gptNeoXSpec` | partial-rotary, DenseMLP, GeGLU-erf | ParallelResidual | yes | no — panics | `needs-runtime-witness` |
| **Falcon** | `falconSpec` | DenseMLP, GeGLU-erf, dim-infer `weights.go:433` | ParallelResidual (if `parallel_attn`) | yes | no — panics | `needs-runtime-witness` |
| **MPT** | `mptSpec` | **alibi** (no RoPE) `arch.go:283`, DenseMLP | PreNorm | yes | partial | `needs-runtime-witness` |
| **StableLM** | `stableLMSpec` | partial-rotary `0.25` | PreNorm | yes | partial | `needs-runtime-witness` |
| **gpt-oss** | `gptOSSSpec` | MoE top-k-softmax+bias, clipped sigmoid-gate | PreNorm (MoE) | yes | no (MoE scalar) | `needs-runtime-witness`; **MXFP4 loader not-started (#489)** |
| **DeepSeek V2/V3** | `deepSeekMLASpec` (**scaffold**, perLayer nil) | MLA latent + decoupled RoPE | — (MLA) | synthetic fixture only | no | `needs-runtime-witness` (#25; `oracle_test.go:459` skip) |
| _GLM-5.2 (DSA)_ | `isGLMMoeDsa` `weights.go:682` | MLA-style proj + learned indexer + MoE | — | CPU-resident only | **no — #86** | gated |
| _MiniMax-M3 (MSA)_ | `isMiniMaxSparseAttn` `weights.go:703` | block-sparse GQA + MoE | — | partial | no | gated (#447 family) |
| _Qwen3.5/3.6 (hybrid)_ | `IsQwen35Hybrid` `qwen35.go:26` | linear/full alternation, attn output-gate | PreNorm | yes | no | gated (#447) |

> The bottom three rows (GLM/MiniMax/Qwen3.x) are beyond the epic's nominal "top-10" but already have
> real arch code in the leaf; listed for completeness since they exercise the same S5/S8 interfaces.

---

## `needs-runtime-witness` summary (what this doc could NOT fully ground from disk)

These require running code against a real non-Llama checkpoint, which is not available in this
environment — they are flagged, not guessed:

1. **Per-family numeric oracles** (Gemma2, OLMo2, Cohere, NeoX, Falcon, MPT, StableLM, gpt-oss): config
   derivation and topology equivalence are tested, but a re-exported HF argmax oracle is the gate, and
   most are absent on disk (the `TestOptional*Oracle*` tests `t.Skip` when `.cache/oracle-*` is missing).
   Owner: **#474**.
2. **Gemma3 local/global** end-to-end: `oracle_test.go:398` proves it *when a Gemma3 checkpoint is
   present*; absent here.
3. **DeepSeek-V2/V3 MLA** real-checkpoint correctness: only the synthetic fixture is proven; the real
   tensor-name mapping is deliberately unpinned (`deepSeekMLASpec` perLayer nil). Owner: **#25**.
4. **GLM-DSA accelerated serving** and the `compute.Backend` path: CPU-resident only today. Owner: **#86 / #413**.

---

## Recommended epic checkbox updates (#487)

On-disk evidence at HEAD proves the following should move from "Open next steps" to shipped (with the
residual noted inline):

- [x] **S0 (#491)** — closed-API adapters — **shipped/CLOSED**.
- [x] **S1 (#490)** — SEAM-0 — **shipped/CLOSED**; residual: `batch.go` un-folded 7th block.
- [x] **S2** — mechanical axes — **shipped** (~10 axes, Llama-no-op gated, `Float32bits`-proven).
- [x] **S3 (#20→folded)** — SWA read-mask option (a) — **shipped**.
- [~] **S4** — topology dispatch — **partial**: proof path shipped; HAL/Metal/quant-batch decode panics
  (`kv.go:522`), still open.
- [x] **S5** — MoE FFN interface — **shipped** (Mixtral/Qwen-MoE/gpt-oss/GLM/MiniMax).
- [x] **S6** — fused-split + longrope — **shipped** (Phi-3/3.5/4).
- [~] **S7 (#489)** — loaders — **partial**: GGUF + Q4_0/Q4_1/Q4_K int4 shipped; **MXFP4 not-started**.
- [~] **S8 (#25)** — MLA + `kvLayout` — **partial**: interface + naive MLA + synthetic fixture; real
  DeepSeek checkpoint outstanding.
- [x] **ST (#488)** — tokenizer leaf — **shipped/CLOSED**.

The biggest single unblock for S4/S5/S8 throughput is generalizing the **accelerated hot-path copies**
(HAL/Metal/quant-batch + `batch.go`) past Llama PreNorm — the work `requirePreNorm` (`kv.go:522`) marks
with an honest panic today.

---

_Generated as the #487 deliverable: a grounded decomposition for a maintainer to act on. All `file:line`
citations were read at `HEAD = ac44a24`. No code changed; `go build ./...` green._

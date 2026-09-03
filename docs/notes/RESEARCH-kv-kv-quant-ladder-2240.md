---
title: "Design dossier — KV-cache quantization ladder: int8/fp8 KV tier"
description: "Design dossier for #2240: the KV-quant ladder's policy plane is already built in-tree; the missing piece is the engine store that holds quantized KV bytes."
---

# Design dossier — KV-cache quantization ladder: int8/fp8 KV tier states + CacheGen-class cold-tier compression (issue #2240)

> Grounded design + staged-implementation dossier for
> [#2240](https://github.com/anthony-chaudhary/fak/issues/2240) (epic #2236, matrix row M7,
> milestone M2 "The KV cache value is owned, observed & 2x" — GLM-5.2 serving performance).
> **Design only — no code shipped by this note.** Every "witnessed" claim below cites the
> file:line it was read from on 2026-07-06; everything under "proposed" is design, not a
> shipped capability. fak-guard arbitration for this work:
> **ADMIT** — `dos_arbitrate(lane="engine", mode="shared", tree=["internal/engine/**"])` →
> `outcome=acquire, reason="cluster lane 'engine' free — admitted."` (decision only; no
> lease journaled — a build increment should `dos lease-lane acquire --lane engine` first).

## (a) Problem, restated from the issue

fak is at R0 on KV-side quantization while every serious engine ships something (TRT-LLM
FP8-KV default on Hopper+, LMDeploy int8/int4 KV, LMCache CacheGen bitstreams for cold
tiers, vLLM/SGLang FP8 options). KV bytes/token is the capacity denominator for every other
M2 row: tiering, eviction, and transfer all move fewer bytes when KV is quantized. The issue
asks for KV precision as a **tier state, not a global flag** — hot spans stay high-precision,
cold spans requantize as they demote — with:

- int8 KV first (matching the proven int8 decode competence in-tree), drift-witnessed;
- fp8 KV on the CUDA path when device GEMM lands;
- a 3/4-bit rotated Lloyd-Max rung with shape-gain norm correction and asymmetric K>V bits (#10710);
- a CacheGen-class bitstream codec for disk/L4 tiers;
- interaction fences: quantized spans must compose with exact-span eviction, and
  spec-decode verify must reject on drift;
- an accuracy ladder per precision, published as capacity-vs-accuracy curves (R3).

Done-when rungs: R1 int8 KV behind a flag with drift witness · R2 demote-requantize live on
the ladder (#1474 satisfied through this) · R3 3/4-bit rotated Lloyd-Max rung with witnessed
production design (#10710) · R3 bench curves vs published TRT-LLM/LMDeploy points.

## (b) Current seam — what the tree actually has today (witnessed)

The striking finding: **the ladder's policy plane is already built end-to-end; the missing
piece is exactly one thing — the engine/store realization that actually holds quantized
bytes.** Every plank below was read from the tree, not inferred.

### b.1 The precision ladder type exists, with two rungs, planner-only

- `internal/compute/kvprecision.go:32` — `type KVPrecision uint8`; rungs
  `KVPrecisionF32` (zero value, three f32 rows) at `kvprecision.go:37` and `KVPrecisionQ8`
  at `kvprecision.go:41` (attended post-RoPE K and V rows q8_0, **pre-RoPE Kraw kept f32 so
  Evict's single-rotation re-positioning stays exact**).
- `internal/compute/kvprecision.go:87-100` — `perTokenPerLayerBytes`: the q8 tier is
  f32-Kraw + two q8_0 rows; `kvQ8RowBytes` at `kvprecision.go:105-112` prices q8_0 exactly
  as llama.cpp block_q8_0 (34 bytes / 32 elems). So q8 is honestly ~2x, not the naive 4x.
- `internal/compute/kvprecision.go:121-143` — `AutoSelectKVPrecision(cfg, budget, wantTokens)`
  auto-steps f32→q8 only when f32 forces a too-small context. Fail-open.
- `internal/compute/compute.go:260-266` — `KVConfig{..., Precision KVPrecision}` already
  carries the tier into every `Backend.NewKV(cfg)` call.
- **The gap is documented in the file itself**: `kvprecision.go:22-26` — "This file is the
  PLANNER/arithmetic half… The denser KVStore *realization* (the byte movement that makes a
  position's K/V actually q8_0-resident…) is the engine half… and is not in this file."
  A repo-wide grep for `KVPrecision` confirms: it appears only in `internal/compute`
  (arithmetic), `internal/cachemeta/quantized_demote.go` (policy packaging),
  `internal/covmatrix/precision.go` (matrix bookkeeping), and docs. **No KVStore anywhere
  stores quantized KV bytes today.**

### b.2 The stores this would extend (all f32-resident today)

- `internal/compute/compute.go:276-289` — the `KVStore` interface:
  `AppendKV(layer, kRaw, kRoPE, v Tensor, pos)`, `KeysView/ValuesView(layer) Tensor`,
  `Evict(from, n)`, `Clone()`. Attention consumes views, never raw storage
  (`compute.go:341`: `Attention(q, kv KVStore, layer, causal, grp, scale)`).
- `internal/compute/cpuref.go:290-298` — `(*cpuBackend).NewKV` returns `cpuKV`
  (`cpuref.go:303-310`): three flat f32 slices per layer, **`cfg.Precision` is ignored** —
  this is the exact line the first slice extends.
- `internal/compute/cuda.go:1295-1313` — `(*cudaBackend).NewKV` allocates device rows at
  `F32.Bytes()` per element (`cuda.go:1307-1309`); the fp8 rung would land here, gated
  `-tags cuda`.
- `internal/model/kvcache.go:11-20` — the in-kernel `model.KVCache` (f32 `K`/`Kraw`/`V`);
  its evictor re-derives every shifted survivor's post-RoPE K from the lossless pre-RoPE
  `Kraw` in a single rotation (`kvcache.go:139-155`) — the bit-exact-evict moat any lossy
  tier must not break.

### b.3 The demote-requantize policy plane is ALREADY SHIPPED (#1474's cachemeta half)

- `internal/cachemeta/quantized_demote.go:37-51` — `QuantizedDemoteTarget{To
  compute.KVPrecision, ResidentBytes int64, Quality QualityEvidence}` — the requantize-down
  demote target, provenance-tied to this ladder.
- `internal/cachemeta/quantized_demote.go:87-94` — `ApplyTo(req *PlacementRequest)` arms
  PlanPlacement's compress-demote lever (`req.CompressedSizeBytes`, `req.Quality`).
- `internal/cachemeta/placement.go:239-246` — `ActionCompressDemote` is admitted **only**
  when `req.Quality.Acceptable()` and retain-beats-recompute; unproven quantization evicts.
- `internal/cachemeta/materialization.go:142-159` — `QualityEvidence{Measured, QualityDelta,
  FaultsObserved, MaxQualityDelta}` / `Acceptable()`: an unmeasured span is never acceptable.
- The fence, stated in the file: `quantized_demote.go:20-22` — "cachemeta owns no bytes and
  never requantizes. The requantization codec and the MEASURED QualityEvidence live in the
  engine (#118)". **That engine codec does not exist yet — it is this ticket.**

### b.4 The executor already routes compress-demote; only the bytes are missing

- `internal/engine/capacity_adapter.go:97-150` — `CapacityAdapter.Execute` handles
  `ActionCompressDemote` in the same fail-safe stage-then-evict family (stage to colder tier
  via `KV.StageSpan` by digest, evict from live tier only on confirmed OK; a staging fault
  retains the live span). Today the "compressed" part is only `EstMoveBytes` accounting.
- `internal/engine/capacity_sweep.go:214-221` — the pressure sweep's drop-action set already
  includes `ActionCompressDemote`; `capacity_sweep.go:85-157` is the bounded sweep loop that
  would drive demote-requantize live.
- `internal/abi/registry.go:642-658` — `abi.KVBackend` with `StageSpan/RestoreSpan`
  addressed by digest, typed OK|MISS|FAULT — the seam a CacheGen-class serialized bitstream
  rides behind (the in-process default answers StageSpan as a no-op OK).
- `internal/engine/capacity_disk.go:38-45` — `DiskPressure` probes the real spill
  filesystem; `capacity_disk.go:92-97` plans against HBM+DRAM+disk at once. The cold tier
  the bitstream codec pairs with is live.

### b.5 The quant kernel competence the int8 rung reuses

- `internal/model/quant.go:110-150` — `quantizeQ8(w, out, in)`: block-32, `d = amax/127`,
  deterministic ties-away `q8round` (`quant.go:89-104`, `//go:noinline` for cross-arch
  bit-identical codes — determinism the evict-composability witness in §(e) leans on).
- `internal/compute/cpuref.go:61` — `QuantizeQ8(be, shape, w, block)` builds Q8_0 weight
  tensors; `compute.go:59` declares the `Q8_0` dtype; the Approx-lane gates are already
  law: cosine ≥ 0.995 (`internal/compute/compute_test.go:150`) and the tighter Q8-lane
  0.999 (`internal/compute/cuda.go:107-112`).
- `FP8` exists as a declared 1-byte dtype (`internal/compute/compute.go:382` test table;
  `compute.go:106` counts it quantized) — **no fp8 KV kernel or KV usage exists anywhere.**

### b.6 One issue-body reference that does NOT ground

The issue's interaction fence "spec-decode rollback (`internal/spec`) must reject on
quantized verify" cites a package that **does not exist in the tree**. There is no
`internal/spec`; the nearest name, `internal/abi/speculate.go`, is TOOL-CALL speculation
(Promote/Rollback of provisional tool effects, `speculate.go:99-132, 326-329`), not
token-level speculative decode. I found no token-level spec-decode verify path to fence.
That fence is recorded as **not groundable today** — it becomes real only if/when a token
spec-decode path lands, and is excluded from the staged plan below rather than invented.

## (c) Proposed mechanism (design — nothing below exists yet)

The design principle, following the tree's own split: compute owns the arithmetic and the
store realization behind `KVStore`; the engine owns the measured evidence and the demote
wiring; cachemeta stays byte-free. Precision is **per-store** (KVConfig already carries it),
and per-SPAN precision emerges from the tier ladder: a span demoted off the hot store is
requantized at the demote boundary, not in place.

### c.1 R1 — realize `KVPrecisionQ8` in the CPU reference store (int8 rung)

Extend `(*cpuBackend).NewKV` (`cpuref.go:290`) to dispatch on `cfg.Precision`:

```go
func (c *cpuBackend) NewKV(cfg KVConfig) KVStore {
    if cfg.Precision == KVPrecisionQ8 {
        return newCPUKVQ8(c, cfg)
    }
    return &cpuKV{...} // unchanged f32 path
}

// cpuKVQ8 realizes the q8 tier: pre-RoPE Kraw stays f32 (the exact-evict row),
// post-RoPE K and V rows are q8_0 codes + per-block(32) f32 scales.
type cpuKVQ8 struct {
    be   *cpuBackend
    cfg  KVConfig
    Kraw [][]float32 // f32, exact — Evict re-RoPEs from this, then requantizes
    kq   [][]int8    // [layer] post-RoPE K codes, row-major per position
    kd   [][]float32 // [layer] per-block scales for kq
    vq   [][]int8    // [layer] V codes
    vd   [][]float32 // [layer] V scales
    pos  []int
}
```

Load-bearing choices:

- **Quantization granularity: per-position row, block-32 within the row** — reusing the
  proven `d=amax/127` + `q8round` scheme (`quant.go:110-150`). For HeadDim=128 this is 4
  scales per head — *finer* than the issue's "per-head scale" ask, chosen for kernel reuse
  and because block-32 is the layout `kvQ8RowBytes` already prices
  (`kvprecision.go:105-112`), so the realization matches the shipped byte arithmetic
  exactly. A quantized block never spans positions, which is what keeps eviction clean
  (§c.2).
- **Views dequantize**: `KeysView/ValuesView` materialize an f32 `[pos, nKV*hd]` tensor by
  dequantizing codes×scales. Attention (`compute.go:341`) is untouched in R1 — correct
  first, fused later. The decode-path dequant-fused-into-attention the issue names is a
  follow-on (`Caps().FusedAttn` backends), not the first slice.
- **AppendKV quantizes at append**: `kRoPE` and `v` rows are quantized on write; `kRaw`
  is stored f32 verbatim.
- **Selection is already wired**: `--kv-precision q8` parses via `ParseKVPrecision`
  (`kvprecision.go:72-81`) and `AutoSelectKVPrecision` (`kvprecision.go:121`) can auto-step;
  R1 makes the token mean what it says on the CPU reference instead of being planner-only.

### c.2 Evict composability (the issue's fence, made concrete)

`cpuKVQ8.Evict(from, n)` compacts `Kraw` (f32, exact) and the code/scale slabs by whole
positions, then for every survivor whose index changed: re-RoPE from f32 `Kraw` at the new
position (exactly `model.KVCache`'s single-rotation discipline, `kvcache.go:139-155`) and
**requantize that one row**. Because (1) `Kraw` is lossless, (2) RoPE at position p' is a
pure function, and (3) `q8round` is deterministic across arches (`quant.go:80-89`), the
survivor's codes are **bit-identical to a q8 store that never saw the evicted span**. The
quarantine witness survives quantization in a strengthened form: not "f32-bit-exact" (the
attended rows are lossy by construction) but **code-exact — evict == never-saw at the
quantized-representation level, max|Δcodes| = 0**. An evicted span's rows are physically
gone (whole-position rows; no cross-position block sharing), so "provably gone" holds.

### c.3 R2 — the engine requantize codec + measured evidence (arms #1474)

New engine-side seam (the codec `quantized_demote.go:20-22` defers to the engine):

```go
// internal/engine/kvrequant.go (proposed)
// MeasureRequantTarget quantizes live span [from,from+n) of kv to the `to` tier,
// measures attention-output drift against the exact rows on the CPU oracle, and
// packages the result as the QuantizedDemoteTarget cachemeta's compress-demote
// lever reads. It never mutates kv.
func MeasureRequantTarget(kv compute.KVStore, cfg compute.KVConfig, from, n int,
    to compute.KVPrecision, maxDelta float64) (cachemeta.QuantizedDemoteTarget, error)
```

`ResidentBytes` comes from `compute.EstimateKVStoreBytes` (`capacity.go:106-110`) so the
decision is priced by the same arithmetic the planner already trusts; `Quality` is a real
`Measured: true` observation (per-span cosine drift → `QualityDelta`), so
`PlanPlacement`'s `Acceptable()` gate (`placement.go:239`) finally has evidence to admit.
The sweep (`capacity_sweep.go:85`) then executes `ActionCompressDemote` through the
existing `CapacityAdapter.Execute` stage-then-evict path (`capacity_adapter.go:120-146`)
with no adapter change. Drift threshold: adopt the Q8 lane's cosine ≥ 0.999
(`cuda.go:107-112`) as `MaxQualityDelta = 0.001` on (1 − cosine), stated in the flag's help.

### c.4 R3a — fp8 KV on the CUDA path (design sketch only; not buildable here)

`(*cudaBackend).NewKV` (`cuda.go:1295`) gains the same `cfg.Precision` dispatch; K/V slabs
allocate at `FP8` width (dtype already declared, `compute.go:382`) with per-head or
per-block scales in a sibling slab (the two-buffer pattern resident Q8_0 weights already
use, `cuda.go:292`). Requires device dequant-fused attention kernels
(`cuda_kernels.cu`) that do not exist; gated `-tags cuda`. **This win32 CPU-only dev box
cannot build or witness any of it** (see
`docs/notes/RESEARCH-turboquant-kv-quant-triage-1266.md`'s AVOID-TESTING-ON-THIS-MACHINE
citation) — fp8 stays a named, unstarted rung with its seam identified.

### c.5 R3b — CacheGen-class bitstream codec for the disk/L4 tier

Not a `KVPrecision` rung (it is not attendable-in-place; it is an at-rest/transfer
encoding) — so it does **not** join the enum. It lives behind the `abi.KVBackend.StageSpan/
RestoreSpan` digest seam (`registry.go:642-658`): a `KVBackend` decorator that, on
StageSpan to `TierDisk`, serializes the span (delta across adjacent positions per channel +
layer-wise quant + entropy coding, per CacheGen) and on RestoreSpan decodes it back. The
CapacityAdapter's fail-safe ordering (`capacity_adapter.go:120-143`) already guarantees a
codec fault retains the live span. Pairs with #2169 (object tier) and the M4 transfer
child. Codec correctness (round-trip, drift bound, compression ratio) is fully witnessable
on this CPU-only host — unlike fp8.

### c.6 — 3/4-bit rotated Lloyd-Max rung (witnessed production design)

Production evidence from `varjoranta/turboquant-vllm@c8a7e0a73b2b9bb93dc66c9380dceab985a0fbc5` establishes a high-fidelity sub-4-bit KV quantization tier (#10710). Naive scalar quantization at 3–4 bits degrades generation quality due to outlier coordinate distortion, centroid shrinkage, and variance amplification. This rung addresses those failure modes through five specific mechanisms:

1. **Shape-gain norm correction:** Low-bit Lloyd-Max optimal scalar quantizers suffer from centroid shrinkage, where reconstructed codebook centroids systematically contract vector magnitude toward zero. In multi-head attention dot products ($QK^T$), this shrinkage attenuates attention logits across sequential transformer layers.

   The rung applies shape-gain norm correction per coordinate group (citing `varjoranta/turboquant-vllm@c8a7e0a73b2b9bb93dc66c9380dceab985a0fbc5` `torch_ops.py:269-300`). After projecting each group through Lloyd-Max centroids, a scalar gain factor compensates for shrinkage: `g = original_norm / (reconstruction_norm + eps)`. Reconstruction multiplies dequantized centroid vectors by `g`. This matches original vector magnitude and prevents downstream logit attenuation without extra codebook bits.

2. **Asymmetric K>V bits:** Attention sensitivity is asymmetric. Keys determine query-key routing; errors in key vectors are exponentiated by softmax, misdirecting attention across context tokens. Values enter as a linear weighted sum where coordinate errors cancel out across attended positions.

   The rung allocates 4 bits for keys (16 Lloyd-Max centroids) and 3 bits for values (8 Lloyd-Max centroids), averaging 3.5 bits per attended element. Crucially, the pre-RoPE `Kraw` row is preserved in full f32 (`internal/compute/capacity.go:103`). This preserves fak's exact-eviction invariant: mid-run cache eviction (`internal/model/kvcache.go:139-155`) re-rotates surviving keys from lossless `Kraw`, guaranteeing that survivor codes match a cache that never observed evicted tokens ($max|\Delta\text{codes}| = 0$).

3. **QJL off by default:** Quantized Johnson-Lindenstrauss (QJL) lemma embeddings use a 1-bit stochastic residual projection to yield an unbiased inner-product estimator in expectation. While mathematically unbiased, the 1-bit residual introduces high per-token variance. In auto-regressive multi-turn generation, nonlinear softmax exponentiates this residual variance, causing cumulative generation degradation (`torch_ops.py:442-447`). The production design keeps QJL disabled by default, using full-bit MSE Lloyd-Max PolarQuant with shape-gain correction.

4. **Sub-byte 3-bit packing and block-diagonal WHT:** Storage footprint and positional structure require two coordinate-level adaptations:
   - *Sub-byte 3-bit packing:* Storing 3-bit indices in 4-bit nibbles wastes 25% of memory. The storage layout packs 8 3-bit indices into 3 contiguous bytes (24 bits total), achieving exactly 3.0 bits per value element without padding overhead.
   - *Block-diagonal WHT for partial-rotary models (#10715):* Orthogonal rotation (Walsh-Hadamard Transform, WHT) normalizes outliers prior to scalar quantization. However, partial-rotary architectures (e.g., Qwen3.6-35B-A3B and MiniMax M2.5) apply RoPE only to a prefix of the head dimension. A monolithic head-wide WHT mixes rotated prefix dimensions with un-rotated suffix dimensions, destroying positional encoding. Block-diagonal WHT partitions the transform into orthogonal diagonal blocks ($WHT_{rotary} \oplus WHT_{non-rotary}$), isolating rotary coordinates while equalizing coordinate distributions within each subspace.

5. **Demote admission gates in `internal/cachemeta`:** Sub-4-bit compression operates under strict admission gates in `internal/cachemeta`. `QuantizedDemoteTarget` (`quantized_demote.go:37-51`) couples tier sizing to `QualityEvidence` (`materialization.go:142-159`). `PlanPlacement` admits `ActionCompressDemote` only when `req.Quality.Acceptable()` evaluates to true (`placement.go:239-246`).

   An unmeasured span is never admitted (`Acceptable() == false`). The engine must measure attention-output cosine and L2 drift against the f32 oracle (`MeasureRequantTarget`) before committing a span to the K4/V3 tier. If observed drift exceeds `MaxQualityDelta` (e.g., drift > 0.005), demotion is rejected, falling back to retention or eviction rather than allowing unverified lossy spans to corrupt inference.

### c.7 KV precision ladder summary comparison

| Rung | Storage format (Kraw / K / V) | Effective bytes/token (hd=128, 1 head) | Outlier & norm handling | Evict composability | Admission gate & policy seam | Issue tracker |
|---|---|---|---|---|---|---|
| **F32** (Baseline) | f32 / f32 / f32 | 1536 B (1.00×) | Exact f32 representation | Exact bit-identity | Default tier (no gate) | Shipped |
| **Q8** (R1) | f32 / q8_0 / q8_0 | 784 B (~0.51×) | Block-32 `amax/127` scales | Code-exact ($max\|\Delta\text{codes}\|=0$) | `QualityEvidence.Acceptable()` | #2240 |
| **FP8** (R3a) | f32 / fp8_e4m3 / fp8_e4m3 | ~800 B (~0.52×) | Per-head / per-block scales | Code-exact | `QualityEvidence.Acceptable()` | #2240 (CUDA) |
| **Rotated K4/V3** (R3c) | f32 / K4 / V3 (8 in 3B) | 656 B (~0.43×) | Block-diagonal WHT + shape-gain norm | Code-exact (re-RoPE f32 Kraw) | `QualityEvidence.Acceptable()` (measured) | #10710, #10715 |
| **CacheGen bitstream** (R3b) | Serialized delta bitstream | ~230–380 B (~0.15–0.25×) | Inter-token delta + Huffman coding | Reconstructed on restore | `abi.KVBackend.StageSpan` | #2240, #2169 |

## (d) Staged plan

**Smallest first shippable slice (R1a):** `cpuKVQ8` behind `cfg.Precision ==
KVPrecisionQ8` in `(*cpuBackend).NewKV` (`cpuref.go:290`) — append/view/evict/clone with
q8_0 attended rows + f32 Kraw, plus its drift and code-exact-evict witnesses. One package
(`internal/compute`), no interface change (`KVStore` and `KVConfig.Precision` already
exist), no serve-path change needed to test it. This alone moves M7 from "planner
arithmetic only" to "realized on the reference engine," and it is the oracle every later
rung (CUDA fp8, codec) is graded against.

Follow-ons, in order:

1. **R1b** — serve-path exposure: honor `--kv-precision q8` end-to-end on the CPU serve
   (today `ParseKVPrecision` exists; the realized store makes it true), with the drift
   witness threshold in the flag docs.
2. **R2** — `MeasureRequantTarget` (§c.3) + wiring a `QuantizedDemoteTarget` into the
   pressure sweep's candidate lowering, satisfying #1474 through this ladder
   (demote-requantize live: hot span f32 → under DRAM/L2 pressure requantize to q8 and
   stay resident instead of evicting).
3. **R3c** — 3/4-bit rotated Lloyd-Max rung (#10710, #10715): CPU reference implementation of
   K4/V3 quantization with shape-gain norm correction (`torch_ops.py:269-300`). Includes 8-in-3
   sub-byte packing for V3, block-diagonal WHT for partial-rotary models (#10715),
   and f32 `Kraw` preservation (`capacity.go:103`). Evaluates drift via `MeasureRequantTarget`
   and gates demote transitions via `internal/cachemeta.QualityEvidence.Acceptable()`, with QJL
   disabled by default.
4. **R3b** — CacheGen-class bitstream codec behind StageSpan/RestoreSpan for `TierDisk`
   (§c.5). CPU-witnessable; ships with round-trip + ratio + drift tests.
5. **R3a** — fp8 KV on the CUDA path (§c.4). Blocked on a CUDA host + device attention
   kernels; design seam named, nothing more claimed.
6. **R3 bench** — capacity-vs-accuracy curves (tokens resident per GB per precision) as a
   bench child of #2236, on a serving host.

## (e) Test plan — witnesses per slice

R1a (package `internal/compute`):

- `TestKVStoreQ8DriftWithinGate` — same random appends into `cpuKV` (f32) and `cpuKVQ8`;
  attention outputs per layer must have cosine ≥ 0.999 vs the f32 store (the Q8-lane gate,
  `cuda.go:107-112`). This is the issue's "bit-exactness fence replaced by a drift witness
  with a stated threshold" — and, per the TurboQuant triage note's discipline, it is an
  attention-fidelity witness, **not** a generation-quality claim.
- `TestKVStoreQ8EvictCodeExact` — build q8 store A with span S in the middle, evict S;
  build q8 store B that never saw S; assert `kq/kd/vq/vd/Kraw/pos` are byte-identical
  (max|Δcodes| = 0). The quantized quarantine witness (§c.2).
- `TestKVStoreQ8ResidentBytesMatchEstimate` — the realized store's resident bytes equal
  `EstimateKVStoreBytes(cfg_q8, n)` (`capacity.go:106`), locking realization to the shipped
  planner arithmetic (whose own tests live in `kvprecision_test.go:37-76`).
- `TestKVStoreQ8CloneIndependent` — Clone then append/evict diverge without aliasing
  (prefix-reuse safety, mirroring `cpuKV`'s contract at `cpuref.go:300-302`).

R2 (package `internal/engine`):

- `TestMeasureRequantTargetProducesMeasuredEvidence` — evidence has `Measured=true`, a real
  `QualityDelta`, and `ResidentBytes < exact` (IsDowngrade,
  `quantized_demote.go:58-60`); an over-bound span yields `Acceptable()==false`.
- `TestCapacitySweepRequantizesInsteadOfEvicting` — a pressure sweep whose candidate
  carries an armed `QuantizedDemoteTarget` executes `ActionCompressDemote` (decision
  reason `compress_demote_beats_recompute`, `placement.go:244`) through
  `CapacityAdapter.Execute`, and a staging fault retains the live span (extends the
  existing adapter tests in `capacity_adapter_test.go`).

R3c (package `internal/compute` and `internal/engine`, #10710, #10715).

- Test `TestKVStoreRotatedK4V3NormCorrection` verifies that `original_norm / reconstruction_norm`
  group scaling preserves vector L2 norm to < 0.1% relative error across varied coordinate
  scales, eliminating centroid shrinkage.
- Test `TestKVStoreRotatedK4V3SubBytePacking` validates dense 8-indices-in-3-bytes packing and
  unpacking, verifying exact round-trip bit recovery without coordinate leakage across byte
  boundaries.
- Test `TestKVStoreBlockDiagonalWHTRotaryIsolation` confirms that block-diagonal WHT on partial-rotary
  dimensions (e.g., 64 rotary prefix + 64 unrotated suffix) leaves cross-boundary transformation
  terms identically zero (#10715).
- Test `TestKVStoreRotatedK4V3EvictCodeExact` builds a K4/V3 store with middle span S and evicts S.
  Re-RoPEing from f32 `Kraw` must yield survivor codes bit-identical ($max|\Delta\text{codes}| = 0$)
  to a store that never received S.
- Test `TestMeasureRequantTargetK4V3DemoteAdmission` verifies that `QualityEvidence.Acceptable()` in
  `internal/cachemeta` admits low-drift spans ($QualityDelta \le MaxQualityDelta$) and refuses
  high-drift spans, protecting `ActionCompressDemote` against lossy degradation.

R3b (package `internal/engine` or a new `internal/kvcodec`):

- `TestKVBitstreamRoundTripDriftBound` — encode a span, decode, drift vs source under the
  stated bound; corrupted bitstream returns a typed FAULT, never silent garbage.
- `TestKVBitstreamSmallerThanExact` — compressed bytes strictly < exact bytes on
  realistic KV tensors (the whole point of the codec), recorded ratio in the test log.

R3a (package `internal/compute`, gated `-tags cuda`, NOT runnable on this host):

- `TestCUDAKVFP8DriftWithinGate` — named now so the rung has a witness slot; runs only on
  a CUDA host.

## (f) Risks & collisions

- **GLM-DSA and hybrid caches are out of scope for R1 and must refuse loudly.** The
  milestone target (GLM-5.2) uses the GLM-MoE-DSA cache path (`model/kvcache.go:18-19,
  123-124, 158-167`) and hybrid Gated-DeltaNet models hold recurrent state with a typed
  evict-refusal (`kvcache.go:54-76`). The q8 realization in R1 covers the standard
  softmax-KV `compute.KVStore` path only; a q8 request for a DSA/recurrent cache must be a
  typed refusal (mirroring `RecurrentEvictUnsupportedError`'s fail-closed style), not a
  silent partial quantization. This narrows R1's direct GLM-5.2 payoff to the byte-count /
  ladder plumbing; quantizing DSA state is its own follow-on and must be said so.
- **The spec-decode fence in the issue is not groundable** (§b.6): no `internal/spec`
  exists; `internal/abi/speculate.go` is tool-effect speculation. Do not invent the fence;
  file it against the ticket when a token spec-decode path exists.
- **Quality witness ceiling on this host.** Attention-cosine is necessary, not sufficient
  (fak's own recorded discipline — `RESEARCH-turboquant-kv-quant-triage-1266.md`, whose
  source's 99.5% cosine coexisted with generation failures). The R1 drift witness gates
  the mechanism; the served generation-quality curve (R3) needs a model-serving host this
  win32 box is not.
- **Partial-rotary coordinate corruption under rotation (#10715):** Applying monolithic WHT
  across the full head dimension mixes RoPE-encoded prefix coordinates with un-encoded suffix
  coordinates, corrupting position encoding. Block-diagonal WHT is required whenever head
  dimension exceeds rotary dimension.
- **QJL residual variance penalty:** The 1-bit stochastic residual in QJL introduces per-token
  variance that is exponentially amplified by softmax in autoregressive multi-turn loops. QJL
  must remain disabled by default.
- **Sub-byte 3-bit packing geometry:** Dense 8-in-3 packing requires dimension and group
  boundaries to align to multiples of 8 elements. Non-multiple configurations must fail closed
  during store initialization.
- **fp8 is design-only here** — no CUDA build on this box; claiming more would be an
  unwitnessed rung.
- **Tree overlap:** arbitration ADMITted lane `engine` (shared, `internal/engine/**`) with
  no live-lease conflict at decision time. Note the R1a slice actually lands in
  `internal/compute/**` (the store realization) — a build increment must arbitrate for
  that tree too, and the current working tree shows heavy uncommitted churn in
  `cmd/fak/**`, `internal/agent/**`, `internal/gateway/**` (disjoint from both), plus
  `internal/engine/capacity_*.go` files listed modified in `git status` — re-arbitrate and
  re-read those seams before editing.
- **Determinism dependency:** the code-exact evict witness leans on `q8round`'s
  cross-arch determinism (`quant.go:80-89`). Any future SIMD KV-quantize path must keep
  the bit-match gate its weight-side siblings already enforce
  (`quant_quantize_vec_arm64_test.go:18`).
- **Estimate drift:** if the realized layout ever diverges from `kvQ8RowBytes`
  (`kvprecision.go:105`), the planner would price demotes wrong;
  `TestKVStoreQ8ResidentBytesMatchEstimate` is the tripwire.

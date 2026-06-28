# Metal hybrid (Qwen3.6 Gated-DeltaNet) prefill twin — implementation spec (#71)

**Status: SPEC / host-independent slice.** This document is the code-level recipe a
Mac-equipped worker executes to land the GPU kernel of [#71](https://github.com/anthony-chaudhary/fak/issues/71)
(*lift `requirePreNorm` so the hybrid Qwen3.6 Gated-DeltaNet arch can use the Metal prefill*).
It was authored on a `windows/amd64`, `CGO_ENABLED=0` box, where the `metal_prefill.go`
kernel (`//go:build darwin && cgo && fakmetal`) **cannot be compiled, run, or measured** — so
the two genuinely-gated steps (the `fakmetal` build and the on-device M3 Pro measurement) are
named here as the remaining work, not faked green. #71 stays **OPEN** until a Mac worker lands
and witnesses the kernel using this spec. The grounding is the *shipped, proven* CPU template
`prefillQwen35HybridQ4K` (commit 87a3859) — every structural decision below maps to a line of
code that already exists and is already pinned by a parity test.

Sibling evidence in this cluster:
[`QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
(the `FAK_QPROFILE` per-phase breakdown this spec's design decision rests on, and the
`51.55 tok/s pp22` llama.cpp-Metal bar).

---

## 1 — The fence today (precise mechanism)

The issue body says `prefillBatchedMetal` "calls `requirePreNorm("Metal prefill")`, so the
hybrid … cannot use the Metal prefill at all (it falls back to CPU)." Reading the tree, the
*mechanism* is more specific than that one line implies, and getting it right is what makes
this spec executable:

- **`requirePreNorm` does NOT panic on the Qwen3.6 hybrid.** It panics only on a non-`PreNorm`
  `BlockTopology` or a layer-specific RoPE theta (`internal/model/kv.go:157`). But the hybrid
  *is* `PreNorm` with a single RoPE theta — `q8Qwen35HybridPrefillOK`
  (`internal/model/qwen35_prefill.go:14`) admits it only when
  `cfg.BlockTopology == PreNorm && !cfg.hasLayerSpecificRopeTheta()`. So the fence is **not**
  the panic.
- **The real fence is that `prefillBatchedMetal` is full-attention-only.** Its per-layer body
  (`internal/model/metal_prefill.go:226`) has **no `isLinearAttnLayer(l)` branch** — it runs
  q/k/v → RoPE → causal GQA → o, then SwiGLU, for *every* layer. On a Qwen3.6-27B that is a
  **3:1 interleave** (48 `linear_attention` Gated-DeltaNet layers + 16 gated-full-attention
  layers, per `cfg.LayerTypes`), so feeding the 48 GDN layers through that body would silently
  produce wrong activations. The hybrid is therefore kept off the Metal lane and routed through
  the CPU hybrid paths (`prefillQwen35HybridQ4K` for `FAK_Q4K`, `prefillQwen35HybridQ` for Q8)
  — which is the "falls back to CPU" the user observes.
- The same upload-side limit holds: `metalWeights()` (`internal/model/metal_prefill.go:68`)
  uploads only the seven full-attention projection names (`metalProjNames`) for *every* layer.
  A hybrid layer has no `self_attn.{q,k,v,o}_proj` — it has `linear_attn.{in_proj_qkv,
  in_proj_z, in_proj_b, in_proj_a, out_proj}`. So the GPU weight table is wrong for the hybrid
  before the forward even starts.

**So "lift `requirePreNorm`" means:** add a hybrid-aware Metal prefill twin and a gate that
routes the hybrid to it; keep `requirePreNorm` as the honest seam marker for the topologies
that genuinely have no Metal path yet. It is the exact move already shipped twice on the CPU
side — `prefillQwen35HybridQ` (Q8) and `prefillQwen35HybridQ4K` (resident-Q4_K) are the
hybrid-aware twins of the generic batched prefills, gated by `q8Qwen35HybridPrefillOK` /
`q4kQwen35HybridPrefillOK`.

---

## 2 — The gate change (`internal/model/kv.go`)

Add a Metal-lane sibling of the existing hybrid gates. In `Prefill` (currently
`kv.go:534-538`):

```go
if s.Metal {
	// Qwen3.6 hybrid (Gated-DeltaNet): batch the projection/MLP GEMMs on the GPU,
	// keep the GDN recurrence on the CPU. The full-attention-only prefillBatchedMetal
	// cannot run the 3:1 interleave; route the hybrid to its Metal twin instead.
	if metalQwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
		return s.headQ(s.prefillBatchedMetalQwen35Hybrid(ids))
	}
	s.requirePreNorm("Metal prefill")
	return s.headQ(s.prefillBatchedMetal(ids))
}
```

Mirror the same two lines into `PrefillNoLogits` (currently `kv.go:631-635`), calling a
`…NoLogits` wrapper exactly as the CPU lane does (`prefillQwen35HybridQ4KNoLogits`,
`qwen35_prefill_q4k.go:60`).

The gate predicate is the same architecture gate the CPU paths use — only the kernel differs,
so it can simply delegate:

```go
// metalQwen35HybridPrefillOK gates the Metal hybrid prefill twin. Same arch gate the Q8 / Q4K
// hybrid paths use (only the projection kernel differs — GPU GEMM vs CPU), the way
// q4kQwen35HybridPrefillOK delegates to q8Qwen35HybridPrefillOK (qwen35_prefill_q4k.go:47).
func metalQwen35HybridPrefillOK(cfg Config, promptLen int) bool {
	return q8Qwen35HybridPrefillOK(cfg, promptLen)
}
```

`q8Qwen35HybridPrefillOK` already requires `IsQwen35Hybrid && promptLen >= 16 && AttnOutputGate
&& !MoE && !DenseMLP && !Alibi && NormGain1p && !LayerNorm && BlockTopology == PreNorm &&
!hasLayerSpecificRopeTheta` — every property the twin assumes. Put `metalQwen35HybridPrefillOK`
in the **non-tagged** `kv.go` (it is pure config logic, no cgo), so the gate compiles in the
default build; only the twin body is tag-gated.

---

## 3 — The twin (`internal/model/metal_prefill.go`, `-tags fakmetal`)

`prefillBatchedMetalQwen35Hybrid` is a **structural copy of `prefillQwen35HybridQ4KHidden`**
(`internal/model/qwen35_prefill_q4k.go:64`) with exactly one substitution: every CPU
projection GEMM becomes a GPU GEMM. The CPU template's `proj` closure dispatches per weight to
`q4kGemmDispatch` / `qGemm8`; the twin's `mm` closure dispatches every projection to the
already-uploaded f16 device handle via `metalgemm.Weight.MatMul` — the identical call
`prefillBatchedMetal` already uses (`metal_prefill.go:202-212`):

```go
mm := func(name string, X []float32, out int) []float32 {
	Y := make([]float32, len(X)/inFor(name)*out) // P*out
	gw[name].MatMul(X, P, Y)                      // Y[P,out] = X[P,in] * W[name]^T on the GPU
	return Y
}
```

### What goes to the GPU (the projections — the wall)

One GPU GEMM per projection, batched over the whole P-token panel:

| layer kind | projections routed to GPU (`mm`) |
|---|---|
| **full-attention** (16 of 64) | `self_attn.q_proj` (width `2·nH·hd` — q ⧺ gate), `k_proj`, `v_proj`, `o_proj` |
| **linear-attention / GDN** (48 of 64) | `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` |
| **all layers** | `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` |

### What stays on the CPU (verbatim from the template)

Copy these blocks **unchanged** from `prefillQwen35HybridQ4KHidden` and its two layer helpers
(`prefillQwen35LinearLayerQ4K`, `prefillQwen35FullAttnLayerQ4K`) — they are pure f32 Go, no
cgo, and are already the single proven reference for the recurrence:

- the input/post RMSNorm (`NormGain1p` 1+w form), the embedding scale;
- **full-attn layer:** split q⧺gate, `applyProjBias` / `applyLayerQKNorm`, RoPE
  (`ropeRowQKInto`), causal GQA (`attnPrefillInto`), the **output gate** `attnOut[i] *=
  sigmoid(gate[i])`, the KV-cache appends (`Kraw` pre-RoPE, `K` post-RoPE, `V`, `pos`);
- **GDN layer:** the causal depthwise `conv1d`+SiLU, q/k L2-norm with the `1/sqrt(kHd)` scale,
  the per-head **delta-rule recurrent scan** (`g=exp(-aExp·softplus(a+dt_bias))`,
  `beta=sigmoid(b)`, rank-1 state update), and the per-head **gated RMSNorm** readout
  (`rmsNormGatedInPlace`), plus the `linearAttnCache` conv/recurrent state;
- SwiGLU (`act(G)·U`), all residual adds, the final norm.

The cache the twin builds is the **same f32 object** the CPU template builds (same `Kraw/K/V/
pos` plus `linearAttnCache`), so decode / Evict / Clone stay valid — identical to the contract
`prefillMetalResident` already honors (`metal_prefill.go:128-172`).

### Upload side — a hybrid-aware weight table

`metalWeights()` (full-attention-only) cannot be reused as-is. Add a hybrid variant that, per
layer, uploads the correct projection set for that layer's kind (full vs linear) plus
`mlp.{gate,up,down}` for all layers, each dequantized from the Q8_0 store exactly as
`dequantQ8` does today (`metal_prefill.go:48`), freeing the f32 staging buffer after each
`metalgemm.Upload` so peak host overhead stays one matrix. The conv1d weight, `A_log`,
`dt_bias`, `linear_attn.norm.weight`, and all biases are **not** GEMMs — they stay host-side
vectors fed to the CPU recurrence, no upload.

---

## 4 — Design decision: CPU recurrence, GPU projections (measured, not assumed)

Per #977's "find the serial step" discipline, the GDN recurrence is the candidate serial step.
The decision to keep it on the **CPU** (rather than write a Metal GDN-scan kernel) is backed by
on-disk measurement, not assumption:

- the `FAK_QPROFILE` per-phase breakdown captured on this M3 Pro
  ([status doc §3](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)) attributes the prefill
  wall to the projection GEMMs — **`mlp_gate_up ≈ 43%`, `mlp_down ≈ 20%`** — while the **GDN
  recurrence is ≈ 0.5%**. #65 carries the same finding (the GDN scan is ~0.5% of prefill on
  this box — the projections are the wall).
- So moving the ~63% of wall that is MLP GEMMs (plus the q/k/v/o and in/out projections) to the
  GPU's FLOP-rich path is the entire lever; a CPU-hybrid recurrence at 0.5% is sufficient. This
  is the exact split `prefillBatchedMetal` already makes for the full-attention case (GPU GEMMs,
  CPU elementwise) — the twin extends it across the GDN layers.

A Metal GDN-scan kernel is therefore **explicitly out of scope** for #71; it would chase 0.5%.
If a later on-device profile shows the recurrence has become the wall *after* the projections
move to the GPU, that is a new, separately-measured ticket — not an assumption to bake in now.

---

## 5 — Default-build stub

`prefillBatchedMetal` has a non-`fakmetal` stub (`metal_prefill_stub.go:9`) so `kv.go` compiles
cgo-free. Add the same one-line panic stub for the new twin so the default build stays one pure
-Go artifact:

```go
func (s *Session) prefillBatchedMetalQwen35Hybrid(ids []int) []float32 {
	panic("model: Metal hybrid prefill not compiled in (build with -tags fakmetal)")
}
```

(`metalQwen35HybridPrefillOK` lives in `kv.go`, un-tagged, so the gate itself needs no stub —
only the twin body does.)

---

## 6 — The remaining gate (the honest "not yet")

Two steps remain, both requiring an Apple-Silicon M3 Pro with the macOS Metal toolchain — the
capability this `windows/amd64` host does not have:

1. **`fakmetal` build + parity witness.** On the Mac:
   `go test ./internal/model -tags fakmetal -run Qwen35Hybrid -count=1`. Add a parity test that
   mirrors `TestPrefillQwen35HybridQ4KMatchesTokenLoop` (`qwen35_prefill_q4k_test.go`): run the
   twin and the proven CPU template on the same prompt and assert the hidden states match within
   the documented Q8 deferred-reduction / FMA-rounding drift band. **This is the green gate that
   closes the correctness half of #71** — it cannot be run here (no darwin/cgo).
2. **On-device tok/s measure.** `FAK_QPROFILE=1` prefill of Qwen3.6-27B q4_k_m at pp22, against
   the `51.55 tok/s` llama.cpp-Metal bar, captured **without a co-resident llama-server** (the
   status doc §3 swap-contamination rule: on a 36 GiB box two 27B copies page to swap; trust the
   clean isolated number). Record the result in this cluster.

**Next checkable step:** apply §2–§5 on a Mac node, run gate (1). Until that test is green and
on a witnessed commit, #71 is `not yet` — the kernel is authored-as-spec, not shipped.

---

## 7 — Why this is the right increment (and what it is not)

This is the host-independent slice: it turns the issue's prose ask into a code-level recipe
bound to the *proven* CPU template, fixes the one inaccuracy in the issue body (the fence is
`prefillBatchedMetal` being full-attention-only, not a `requirePreNorm` panic), and records the
measured design call (CPU recurrence). It is **not** a claim that the Metal kernel is built,
compiled, or measured — those are the gated steps in §6, and #71 stays open until a Mac worker
witnesses them. No fabricated pass: the only gate run for *this* change is the doc/commit gate
on the `experiments` lane.

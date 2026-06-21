# N4 · model/mlp+residual

This module is the position-wise feed-forward and residual-stream half of the
transformer decoder block in `fak/internal/model`. For each token position it
computes the MLP delta — the gated linear unit `down(act(gate(x)) · up(x))`
(SwiGLU when `act = SiLU`, GeGLU when `act` is tanh/erf-GELU), or, for a sparse
layer, a Mixture-of-Experts delta `Σ_{e∈topk} w_e · expert_e(x)` — and folds that
delta back into the residual stream by exact addition under the configured block
topology. "Correct" here is **regime N (numerical)**: the computed tensor must equal
the mathematically-defined function. Where the path is integer-exact-representable
we demand bit-identity (`max|Δ|=0`); where projections reduce in float we demand
oracle/metamorphic parity within a stated tolerance. The witnesses below are
hand-computed or independent-reference oracles — they do **not** depend on the
optional HF export cache, so they earn PROVEN on a clean checkout.

---

## THEOREM 1 — SwiGLU / GeGLU MLP computes `down(act(gate(x)) · up(x))`

**REGIME** N (numerical, bit-exact for the dense gate; 1e-6 for the GeGLU+bias arithmetic).

**STATEMENT** For every layer and normalized input `x`, the dense MLP delta equals
`down_proj( act(gate_proj(x)) · up_proj(x) )` elementwise, with `act = SiLU` for
Llama/Qwen and tanh/erf-GELU for Gemma's GeGLU, and optional per-projection bias
added before the activation and after the down-projection.

**PROOF** `denseSwiGLU.apply` (`fak/internal/model/moe.go:76`) computes `g =
gate_proj(x)`, `u = up_proj(x)`, adds optional bias (`moe.go:82-83`), then
`g[i] = act(g[i]) * u[i]` (`moe.go:85`) and `out = down_proj(g)` (`moe.go:87`) — i.e.
verbatim `down(act(gate)·up)`. `act()` (`arch.go:405`) selects SiLU
(`forward.go:421`: `z/(1+exp(-z))`) or tanh/erf-GELU per `cfg`. The witness is exact
oracle parity against an **independent** inline reference `inlineDenseFFN`
(`moe_test.go:29`), which recomputes the same form via `matRows`/`silu` with a literal
`silu(g)*u`; `TestMoEDenseNoOpIdentical` asserts `Float32bits`-identity
(`assertFloat32BitsEqual`, `max|Δ|=0`) over 8 pseudo-random hidden vectors per layer —
stronger than a cosine bound. `TestDenseActivationMLPWithBias` independently pins the
GeGLU(erf)+bias path to hand-computed scalars within `1e-6`.

**WITNESS**
```
go test -run 'MoEDenseNoOpIdentical|DenseActivationMLPWithBias' ./internal/model/ -count=1 -timeout 120s -v
```

**VERDICT** PROVEN (2026-06-20, macOS native go1.26). `--- PASS:
TestMoEDenseNoOpIdentical`; `--- PASS: TestDenseActivationMLPWithBias`; package `ok
0.213s`. `TestMoEMixedDenseAndSparseLayerDispatch` also re-asserts the dense delta
bit-exactly.

**DOS** bound at ship.

---

## THEOREM 2 — the residual stream is exact addition (no scaling/clipping unless the arch defines it)

**REGIME** N (metamorphic / structural oracle parity, tolerance `1e-5` carried by the
projection reductions, not by the add).

**STATEMENT** The residual update is exact elementwise addition of the sub-layer
delta into the stream — `x += body(norm(x))` for the PreNorm (Llama/Qwen) default —
with no scaling or clipping. The only departures are the architecture-**defined**
topologies: PostNorm `x += norm(body(x))`, SandwichNorm `x += post(body(pre(x)))`,
ParallelResidual `x += attn_delta + mlp_delta` summed into one stream.

**PROOF** `composeSeqSublayer` (`fak/internal/model/forward.go:322`) implements the
residual add per topology; the PreNorm default (`forward.go:341-347`) is the bare loop
`x[tt][i] += out[tt][i]` — no scale, no clip. The witness is structural oracle parity:
`referenceBlock` (`arch_test.go:816`) recomputes one decoder block from primitives,
where the residual add is `addSeq` (`arch_test.go:868`), a pure `x[t][i] += d[t][i]`;
for PreNorm the reference is `addSeq(x, body(norm(x)))`.
`TestBlockTopologyComposition` asserts the production `composeBlock` output equals this
independent re-derivation within `max|Δ|=1e-5` for **all four** topologies — a spurious
scale or clip in the add would diverge well past tolerance. The float tolerance
reflects reduction-order in the projections, not the add, which is a clean float `+`.
`TestSandwichNormUsesDistinctFeedForwardNorms` cross-checks the norm-placement variant.

**WITNESS**
```
go test -run 'BlockTopologyComposition' ./internal/model/ -count=1 -timeout 120s -v
```

**VERDICT** PROVEN (2026-06-20). `--- PASS: TestBlockTopologyComposition` with all four
subtests PASS (`PreNorm`, `PostNorm`, `SandwichNorm`, `ParallelResidual`); package `ok
0.213s`.

**DOS** bound at ship.

---

## THEOREM 3 — MoE routing selects top-k experts and the combine is a correct weighted sum

**REGIME** N (hand-computed oracle parity: exact expert indices, `1e-6` gate weights,
`1e-5` weighted-sum delta).

**STATEMENT** MoE routing computes `logits = router(x)`, `probs = softmax` over **all**
experts, selects the top-k by prob with torch.topk's stable tie-break (largest first,
lower index on ties), renormalizes the k gate weights to sum to 1 when `NormTopKProb`,
and the FFN delta is the gate-weighted sum `delta = Σ_{e∈topk} w_e · expert_e(x)`.

**PROOF** `route` (`fak/internal/model/moe.go:185`) computes `logits = router·x`,
`probs = softmaxOf` over all `E` experts (`moe.go:193`), sorts indices by prob-desc via
`sort.SliceStable` (`moe.go:200` — stable ⇒ lower index wins ties = torch.topk), takes
the first `K` picks (`moe.go:206`), and divides by their sum when `NormTopKProb`
(`moe.go:211`). `moeFFN.apply` (`moe.go:274`) accumulates `delta[i] += pk.weight*out[i]`
over the picks (`moe.go:287`) — the gate-weighted sum. The witness is a fully
hand-computed oracle: `TestMoERoutingHandComputed` (`moe_test.go:250`) sets router rows
so `logits=[2,3,2,-2]`, computes the reference softmax / top-2 by hand (`e1`, then the
`e0`/`e2` tie at `2` broken to the lower index `e0`) and the renormalized weights, then
asserts (a) `picks[0..1].expert == {1,0}` exactly, (b) each gate weight within `1e-6` of
`probs[e]/sumSel`, and (c) the full weighted-sum delta within `1e-5` of `Σ
w_e·refExpert_e`. `TestMoEWiring` independently checks `K` distinct in-range picks with
weights summing to 1; `TestGPTOSSRouterUsesTopKSoftmaxAndBias` covers the
select-then-softmax-top-k variant plus router bias.

**WITNESS**
```
go test -run 'MoERoutingHandComputed|MoEWiring|GPTOSSRouterUsesTopKSoftmaxAndBias' ./internal/model/ -count=1 -timeout 120s -v
```

**VERDICT** PROVEN (2026-06-20). `--- PASS: TestMoERoutingHandComputed`; `--- PASS:
TestMoEWiring`; `--- PASS: TestGPTOSSRouterUsesTopKSoftmaxAndBias`. Note:
`TestOptionalQwen3MoEOracleCoversHybridDenseSparseLayers` **SKIPPED** (no
`.cache/oracle-qwen3moe` fixture) — that is a redundant end-to-end HF-oracle
confirmation of hybrid dense/sparse layers, not the load-bearing witness; the
hand-computed test discharges the theorem. The skipped oracle would, if its fixture
were exported, additionally confirm parity against real Qwen3-MoE weights.

**DOS** bound at ship.

---

## Notes on honesty / residual

- The three theorems are discharged by **author-independent** witnesses (an inline
  reference, a primitive block re-derivation, and a by-hand softmax/top-k), so a bug
  that re-used the production code as its own oracle could not pass them silently.
- The only un-run rung is the **optional** Qwen3-MoE HF oracle (fixture absent → SKIP).
  It is not promoted to PROVEN and is not needed: it would confirm end-to-end parity
  against real weights, which is strictly weaker evidence for these three specific
  properties than the hand-computed tests already provide.
- All runs were on the macOS native toolchain (the Windows/WSL machinery in the root
  `CLAUDE.md` does not apply on this node); re-run with the exact `-run` commands above.

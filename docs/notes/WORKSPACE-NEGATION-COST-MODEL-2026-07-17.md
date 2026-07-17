# Workspace-to-negation cost model — 2026-07-17

## Claim and variables

This note makes the negation program falsifiable. It does **not** assume that every model
pays a negation tax; it names the quantities that must exhibit one before fak may claim it.

For a matched task pair, define:

- `D`: effective forward-pass depth required before the correct state crystallizes;
- `W`: deliberate-workspace occupancy from concurrent goals or transformations;
- `E`: representational entanglement between a concept and its negated form;
- `T`: measured per-token or forward-pass latency;
- `A`: held-out task accuracy;
- `R`: operator rung/route (`L0` through `L4`).

A compact hypothesis is:

```text
C_neg = alpha*D + beta*W + gamma*E + T
Delta_net(R) = (A_R - A_base) - lambda*(C_R - C_base)
```

where `alpha`, `beta`, `gamma`, and `lambda` are non-negative measurement weights, not
fitted facts. The thesis predicts that naive negation raises `D` and/or workspace-sensitive
error because the model constructs a concept and maintains an inversion/suppression state.
A useful operator must improve `Delta_net`: accuracy alone is insufficient if its added
latency or routing cost is larger than the depth it removes.

## Numbered predictions and falsifiers

The machine-readable authority is
[`internal/negframe/testdata/negation_cost_predictions.json`](../../internal/negframe/testdata/negation_cost_predictions.json).
`TestNegationPredictions` rejects duplicate ids, missing variables, missing falsifiers,
missing measurement surfaces, and unknown rung names.

1. **NEG-COST-01 — workspace contention.** At fixed capacity, negation accuracy should
   degrade faster than matched affirmative accuracy under deliberate load. Equal slopes
   across three or more load levels falsify this prediction. Measure with the negation
   witness suite crossed with workspace-selectivity loads. This can discriminate every
   rung because a successful operator should reduce the differential.
2. **NEG-COST-02 — entanglement.** Held-out crystallization depth/error should increase
   with activation overlap after prompt-length control. No positive relationship falsifies
   it. Join `NegationGeometryReport` to layerwise `LayerLogits`; this targets L2/L3.
3. **NEG-COST-03 — causal offload.** At fixed or lower depth, an L3 operator should beat
   base emulation and survive activation-patch controls. No equal-depth gain, or a gain that
   vanishes under causal controls, falsifies it. Measure the shipped adapter and affine
   fresh-forward residual-hook witnesses.
4. **NEG-COST-04 — depth/latency tax.** At non-worse accuracy, the operator should reduce
   depth or latency on a committed matched corpus. If neither quantity falls, the payoff
   claim is false. This spans L0-L4 and binds the planned depth/latency rows to logit-lens
   crystallization.
5. **NEG-COST-05 — affirmative bypass.** A silent detector must execute zero adapter MACs,
   allocate no adapter state, and preserve output bits. Any violation falsifies zero-cost
   routing. The shipped `TestNegAdapterApplyAndZeroCostBypass` measures L3 and constrains L4.

## Rung interpretation

- **L0/L1:** lexical or request-time positive rewrites can reduce prompt-level inversion,
  but cannot by themselves establish a forward-pass mechanism.
- **L2:** detection/classification predicts where tax should occur; false-positive routing
  is counted as cost rather than hidden.
- **L3:** activation operators must show causal equal-depth gains and an inert clean path.
- **L4:** native primitives may make inversion structural, but synthetic involutivity is
  only a seam witness until a weight-backed task confirms `Delta_net > 0`.

A failed prediction is a result, not a defect to hide. Near-zero or negative measured tax
must remain in the artifact and blocks any net-true performance claim.

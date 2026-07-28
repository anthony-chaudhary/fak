# Model architecture seam

## Activation-space residual hook

`internal/model.ResidualHook` is the gated L3 write seam for activation-space operators.
Install it with `Model.SetResidualHook`; enable invocation explicitly with
`Config.EnableResidualHook`. The zero-value config is off, and an unset hook is inert.

The hook runs once after each complete transformer block has formed its residual stream,
for PreNorm, PostNorm, SandwichNorm, and ParallelResidual topologies. Its signature is:

```go
type ResidualHook func(layer int, hidden []float32)
```

`hidden` is mutable model-owned storage. A hook may update it in place, must not retain the
slice, and must preserve its length. Because invocation follows the complete block residual
add, operators see the same semantic install point for every topology. Identity hooks must
leave downstream logits bit-identical; the disabled path performs no allocation or copy.
## L3a affine negation operator

`internal/model.FitNegationAffine` fits both candidates at a captured layer: the
single-vector baseline `h += v_L` and the general map `h -> A h + b`. Training and held-out
pairs are distinguished by their `Split`; `SweepNegationAffine` selects a layer only from
held-out affine reconstruction error and records steering, affine, unpatched, and deterministic
random-control scores in `testdata/negation_affine_layer_sweep.json`.

Install a fitted operator on a fresh forward pass with `Model.SetResidualHook(op.Hook())` and
set `Config.EnableResidualHook`. `Hook` mutates only `op.Layer`; all other layers are identity.
`NegationAffineOperator.PatchActivation` exposes the same fitted state through the reusable
`ActivationPatch` capture/inject harness. Shape mismatches fail rather than partially patching.

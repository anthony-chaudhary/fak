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

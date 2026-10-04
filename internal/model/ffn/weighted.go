package ffn

import "fmt"

// AddScaled accumulates a gate-weighted expert output into a running
// accumulator: dst[i] += weight*src[i] for every element, in increasing index
// order. It is the projection-free accumulation half of a routed-expert
// contraction, shared by every MoE consumer so the reduction order that future
// changes must preserve lives in exactly one place.
//
// The caller owns dst; src is read-only. The length contract is validated
// BEFORE the first mutation, so a refused call leaves dst and src
// byte-identical: dst and src must be the same length. Equal empty slices are a
// valid no-op. The helper has no finite-value policy and returns no error for
// NaN/Inf — its callers own that decision. On success the elements are
// accumulated in increasing index order with zero allocations.
func AddScaled(dst, src []float32, weight float32) error {
	if len(dst) != len(src) {
		return fmt.Errorf("ffn: dst length %d does not match src length %d", len(dst), len(src))
	}
	for i := range dst {
		dst[i] += weight * src[i]
	}
	return nil
}

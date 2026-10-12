package model

import (
	"fmt"
	"math"
)

// v41IndexWeightsBF16 implements the pinned model's standalone BF16 weight
// boundaries under the PyTorch v2.9.0 CUDA multiply source contract. The
// projected tensor is BF16; the combined Python scalar is converted once to
// F32 opmath, multiplied in F32, then returned as BF16. In particular, the
// scalar is not rounded to BF16 before multiplication. This does not pin the
// reference's runtime version or emulate its FP8/FP4 projection arithmetic.
// Both copybacks own their output, so failure never partly changes the caller.
func v41IndexWeightsBF16(l int, projected []float32, dim, nHeads int) ([]float32, error) {
	return v41IndexWeightsBF16Device(l, projected, dim, nHeads, nil, nil)
}

// v41IndexWeightScale is the F32 opmath scalar both arms multiply by. Keep both
// factors and their product in F64 before CUDA's F32 scalar extraction.
// Positive int geometry bounds the resulting scale to (0, 1]. This preserves
// evaluation precision, not generic Python/Go libm bits.
func v41IndexWeightScale(dim, nHeads int) float32 {
	return float32(math.Pow(float64(dim), -0.5) * math.Pow(float64(nHeads), -0.5))
}

// v41IndexWeightsBF16Device rounds the projection through the session's
// ordinary device publication when one is bound. A bound scale seam then runs
// the F32 multiply and the scaled-weight BF16 publication on the device; nil
// keeps the host multiply. Under device-only the caller passes a refusing seam,
// so no host multiply is reachable there.
func v41IndexWeightsBF16Device(l int, projected []float32, dim, nHeads int, publish v41BF16BoundaryFunc, scaled v41IndexWeightScaleFunc) ([]float32, error) {
	const leaf = "indexer.weights_proj.weight"
	if dim <= 0 || nHeads <= 0 || len(projected) != nHeads {
		return nil, &V41ProjectionOperationError{
			Layer: l, Leaf: leaf, Stage: string(v41StageIndexer),
			Cause: v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: invalid index weight geometry", ErrV41ForwardStage)),
		}
	}
	weights, err := v41IndexBF16Publish(l, leaf, "projection", projected, publish)
	if err != nil {
		return nil, err
	}
	scale := v41IndexWeightScale(dim, nHeads)
	if scaled != nil {
		out, err := scaled(l, weights, scale)
		if err != nil {
			return nil, err
		}
		if len(out) != len(weights) {
			return nil, &V41IndexWeightScaleOperationError{Layer: l, Cause: errV41IndexWeightScaleResult}
		}
		return out, nil
	}
	for h := range weights {
		weights[h] = float32(weights[h] * scale)
	}
	return v41IndexBF16Publish(l, leaf, "scaled weight", weights, publish)
}

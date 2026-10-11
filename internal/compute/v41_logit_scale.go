package compute

import (
	"fmt"
	"math"
)

// V41LogitScaleBackend is an optional exact binary32 scalar multiplication.
// Input is an unquantized row-major F32 [N] tensor of finite values; scale is
// finite, including either signed zero and subnormals. Each output is exactly
// the separately rounded binary32 product input[i]*scale, using RNE with
// subnormals and signed zero preserved. Finite overflow to infinity is allowed.
// NaN inputs and their payload propagation are outside this contract.
//
// The result is a fresh caller-owned allocation on this backend. This operation
// does not apply softcap or token suppression. Absence permits host selection;
// after dispatch is selected, an error never permits host replay.
// Ordinary Vulkan builds do not expose the experimental implementation.
type V41LogitScaleBackend interface {
	V41LogitScaleAvailable() bool
	V41LogitScale(input Tensor, scale float32) (Tensor, error)
}

func validateV41LogitScale(input Tensor, scale float32) (int, error) {
	if input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil ||
		len(input.Shape) != 1 || input.Shape[0] <= 0 || input.Shape[0] > math.MaxInt32 || input.Shape[0] > math.MaxInt/4 {
		return 0, fmt.Errorf("compute: V4.1 logit scale requires nonempty row-major F32 [N] in the shader/host byte range")
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return 0, fmt.Errorf("compute: V4.1 logit scale requires a finite binary32 scalar")
	}
	return input.Shape[0] * 4, nil
}

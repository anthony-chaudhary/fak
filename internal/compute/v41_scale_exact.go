package compute

import (
	"fmt"
	"math"
)

// V41ScaleExactBackend is the ordinary build's exact binary32 scalar multiply,
// the V41LogitScaleBackend arithmetic on a registry module instead of the
// witness archive. Input is an unquantized row-major F32 [N] tensor; scale is
// finite, including either signed zero and subnormals. Each output is exactly
// the separately rounded binary32 product input[i]*scale under RNE, with
// subnormals and signed zero preserved. Finite overflow to infinity is allowed.
// NaN inputs and their payload propagation are outside this contract.
//
// Available reports the admitted strict module on the selected device's F32
// DenormPreserve, SignedZeroInfNanPreserve and RoundingModeRTE controls. The
// result is a fresh caller-owned allocation; after dispatch is selected, an
// error never permits host replay.
type V41ScaleExactBackend interface {
	V41ScaleExactAvailable() bool
	V41ScaleExact(input Tensor, scale float32) (Tensor, error)
}

func validateV41ScaleExact(input Tensor, scale float32) (int, error) {
	if input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil ||
		len(input.Shape) != 1 || input.Shape[0] <= 0 || input.Shape[0] > math.MaxInt32 || input.Shape[0] > math.MaxInt/4 {
		return 0, fmt.Errorf("compute: V4.1 exact scale requires nonempty row-major F32 [N] in the shader/host byte range")
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return 0, fmt.Errorf("compute: V4.1 exact scale requires a finite binary32 scalar")
	}
	return input.Shape[0] * 4, nil
}

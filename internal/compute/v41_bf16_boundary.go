package compute

import (
	"fmt"
	"math"
)

// V41BF16BoundaryBackend rounds the bits of an unquantized row-major F32 [N]
// tensor to BF16 RNE and widens the result to F32 storage on this backend.
// The result is a fresh caller-owned N-element allocation, never an input view.
// For each uint32 word u, the exact operation is
// (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000, with unsigned wrapping.
// No floating-point instruction may canonicalize NaNs, flush subnormals, or
// discard a signed zero. Callers reject nonfinite inputs and rounded overflow;
// this raw bit primitive intentionally matches v41RoundBF16 on every word.
// Capability absence permits the existing host path before selection; a
// selected failure has no decline outcome and must never replay on the host.
type V41BF16BoundaryBackend interface {
	V41BF16BoundaryAvailable() bool
	V41BF16Boundary(input Tensor) (Tensor, error)
}

func validateV41BF16Boundary(input Tensor) (int, error) {
	if input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil ||
		len(input.Shape) != 1 || input.Shape[0] <= 0 || input.Shape[0] > math.MaxInt32 || input.Shape[0] > math.MaxInt/4 {
		return 0, fmt.Errorf("compute: V4.1 BF16 boundary requires nonempty row-major F32 [N] in the shader/host byte range")
	}
	return input.Shape[0] * 4, nil
}

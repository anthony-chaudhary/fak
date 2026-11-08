package compute

import "errors"

// SmallmPath is a host-side route decision, not evidence of a loaded kernel.
type SmallmPath uint8

const (
	SmallmGeneral SmallmPath = iota
	SmallmVector
)

var ErrSmallmGeometry = errors.New("compute: invalid small-M geometry")

// SelectSmallmDispatch describes the tiny BF16 decode window of SGLang's
// hopper_bf16_tiny_gemv.cuh at58484246d26523eaf1a7f16ab524deee288d2120.
// The specialization accepts M up to kMaxM (at most32) and vectorized K
// chunks of1024; the documented tiny-N regime is N<=64. Configuring a
// threshold does not override these constraints. K must also fit the
// upstream wrapper and kernel's uint32 representation. Unsupported positive shapes
// stay on the general path, while malformed dimensions fail closed. The
// caller must separately qualify BF16 data, CUDA support and kernel binding.
func SelectSmallmDispatch(m, n, k, threshold int) (SmallmPath, error) {
	if m <= 0 || n <= 0 || k <= 0 || threshold < 1 || threshold > 32 {
		return SmallmGeneral, ErrSmallmGeometry
	}
	if m <= threshold && n <= 64 && uint64(k) <= uint64(^uint32(0)) && k%1024 == 0 {
		return SmallmVector, nil
	}
	return SmallmGeneral, nil
}

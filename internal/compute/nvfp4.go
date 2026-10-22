package compute

import (
	"fmt"
	"math"
)

// NVFP4Matrix is a restricted, canonical logical CPU representation, not a
// checkpoint ABI or a CUDA scale layout. It provides only the decode/reference
// portion of readiness R0 (#13937): no quantizer, Dtype dispatch, loader, or
// hardware support is enabled here.
//
// Packed holds row-major E2M1 codes, even elements in the low nibble. Each row
// has a positive multiple of 16 columns. BlockScales holds one E4M3FN byte per
// consecutive 16 elements, without padding or swizzling. Only nonnegative scales
// with MSB zero are canonical here; unlike cuBLAS UE4M3, we do not ignore bit 7.
// GlobalScale is the finite, nonnegative FP32 *decode* multiplier, not its
// reciprocal. Negative zero scales are rejected; positive zero is allowed.
// Thus the logical payload occupies Rows*Cols/2 + Rows*Cols/16 + 4 bytes,
// excluding shape and Go slice metadata.
//
// NVIDIA specifies x = E2M1 * block scale * global scale:
// https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html
// E2M1 has no Inf/NaN; E4M3FN NaNs are 0x7f/0xff:
// https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html#alternate-floating-point-data-formats
// This oracle claims neither cuBLAS input compatibility nor device bit parity.
// Callers must not mutate the slices concurrently with either method.
type NVFP4Matrix struct {
	Rows, Cols  int
	Packed      []byte
	BlockScales []byte
	GlobalScale float32
}

func (m NVFP4Matrix) validate() (int, error) {
	if m.Rows <= 0 || m.Cols <= 0 || m.Cols%16 != 0 || m.Rows > int(^uint(0)>>1)/m.Cols {
		return 0, fmt.Errorf("NVFP4: invalid or overflowing shape %dx%d", m.Rows, m.Cols)
	}
	n := m.Rows * m.Cols
	if len(m.Packed) != n/2 || len(m.BlockScales) != n/16 {
		return 0, fmt.Errorf("NVFP4: payload lengths do not match shape %dx%d", m.Rows, m.Cols)
	}
	if !nvfp4Finite(m.GlobalScale) || math.Signbit(float64(m.GlobalScale)) {
		return 0, fmt.Errorf("NVFP4: global decode scale must be canonical nonnegative finite FP32")
	}
	for i, s := range m.BlockScales {
		if s&0x80 != 0 || s == 0x7f {
			return 0, fmt.Errorf("NVFP4: block scale %d is not canonical nonnegative finite E4M3FN", i)
		}
	}
	return n, nil
}

func nvfp4Finite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }

func (m NVFP4Matrix) decode(i int) float32 {
	code := m.Packed[i/2] >> uint(4*(i%2)) & 15
	mag := [...]float64{0, .5, 1, 1.5, 2, 3, 4, 6}[code&7]
	if code&8 != 0 {
		mag = -mag // Includes the E2M1 negative-zero code 0x8.
	}
	s := m.BlockScales[i/16]
	exp, mant := int(s>>3), int(s&7)
	var scale float64
	if exp == 0 {
		scale = float64(mant) / 512
	} else {
		scale = math.Ldexp(float64(8+mant), exp-10)
	}
	return float32(mag * scale * float64(m.GlobalScale))
}

// Dequantize returns a new row-major FP32 slice. The exact three-factor product
// is evaluated in FP64 then rounded once to FP32 (nearest, ties to even), with
// subnormals retained and signed zero preserved, including underflow. Any FP32
// overflow is an error. Validation and errors never expose partial output.
func (m NVFP4Matrix) Dequantize() ([]float32, error) {
	n, err := m.validate()
	if err != nil {
		return nil, err
	}
	y := make([]float32, n)
	for i := range y {
		y[i] = m.decode(i)
		if !nvfp4Finite(y[i]) {
			return nil, fmt.Errorf("NVFP4: decoded element %d overflows FP32", i)
		}
	}
	return y, nil
}

// GEMV computes m*x into a new slice, without materializing the decoded matrix.
// Decoding follows Dequantize. Each product and each left-to-right addition is
// separately rounded to FP32, starting from +0; there is no fused multiply-add.
// Inputs must be finite, and any decoded/product/sum overflow returns nil and an
// error. This specifies a deterministic CPU oracle, not GPU reduction behavior.
func (m NVFP4Matrix) GEMV(x []float32) ([]float32, error) {
	if _, err := m.validate(); err != nil {
		return nil, err
	}
	if len(x) != m.Cols {
		return nil, fmt.Errorf("NVFP4: vector length %d, want %d", len(x), m.Cols)
	}
	for i, v := range x {
		if !nvfp4Finite(v) {
			return nil, fmt.Errorf("NVFP4: vector element %d is not finite", i)
		}
	}
	y := make([]float32, m.Rows)
	for r := range y {
		var sum float32
		for c, v := range x {
			w := m.decode(r*m.Cols + c)
			product := float32(float64(w) * float64(v))
			sum = float32(float64(sum) + float64(product))
			if !nvfp4Finite(w) || !nvfp4Finite(product) || !nvfp4Finite(sum) {
				return nil, fmt.Errorf("NVFP4: FP32 overflow at row %d column %d", r, c)
			}
		}
		y[r] = sum
	}
	return y, nil
}

package compute

import (
	"errors"
	"fmt"
	"math"
)

// ErrNVFP4GEMMUnsupported reports that a backend has no native NVFP4 W4A4
// matrix multiplication capability. MatMulNVFP4 never silently widens or falls
// back to another operation.
var ErrNVFP4GEMMUnsupported = errors.New("compute: NVFP4 GEMM unsupported")

// NVFP4MatMulBackend is the optional prequantized NVFP4 W4A4 capability.
// Operands are row-major W[out,k] and A[rows,k]; the result is F32[rows,out].
type NVFP4MatMulBackend interface {
	MatMulNVFP4(w, a Tensor) (Tensor, error)
}

// MatMulNVFP4 dispatches only to an explicitly implemented W4A4 capability.
func MatMulNVFP4(be Backend, w, a Tensor) (Tensor, error) {
	if be == nil {
		return Tensor{}, fmt.Errorf("%w: nil backend", ErrNVFP4GEMMUnsupported)
	}
	native, ok := be.(NVFP4MatMulBackend)
	if !ok {
		return Tensor{}, fmt.Errorf("%w: backend %q", ErrNVFP4GEMMUnsupported, be.Name())
	}
	return native.MatMulNVFP4(w, a)
}

// MatMulNVFP4 is the CPU oracle for the optional native W4A4 capability. It
// preserves the two-level NVFP4 scaling order: local E2M1*E4M3 values, serial
// and separately rounded F32 products/adds, then one tensor-global product.
func (c *cpuBackend) MatMulNVFP4(w, a Tensor) (Tensor, error) {
	if c == nil {
		return Tensor{}, fmt.Errorf("%w: nil cpu backend", ErrNVFP4GEMMUnsupported)
	}
	if w.Backend() != c || a.Backend() != c {
		return Tensor{}, fmt.Errorf("%w: CPU NVFP4 operands must be owned by the receiving backend", ErrNVFP4Input)
	}
	wPacked, wScales, wGlobal, err := checkedNVFP4WeightPayload(w)
	if err != nil {
		return Tensor{}, err
	}
	aPacked, aScales, aGlobal, err := checkedNVFP4WeightPayload(a)
	if err != nil {
		return Tensor{}, err
	}
	out, k := w.Shape[0], w.Shape[1]
	rows := a.Shape[0]
	if a.Shape[1] != k {
		return Tensor{}, fmt.Errorf("%w: activation K is %d, want %d", ErrNVFP4Shape, a.Shape[1], k)
	}
	count, ok := fp8Block32Product(rows, out)
	if !ok || count <= 0 {
		return Tensor{}, fmt.Errorf("%w: output shape [%d %d] overflows or is empty", ErrNVFP4Shape, rows, out)
	}
	alpha := nvfp4RoundedF32(wGlobal * aGlobal)
	if !nvfp4Finite(alpha) {
		return Tensor{}, fmt.Errorf("%w: tensor-global product is non-finite", ErrNVFP4Scale)
	}
	if wGlobal != 0 && aGlobal != 0 && alpha == 0 {
		return Tensor{}, fmt.Errorf("%w: non-zero tensor-global product underflowed", ErrNVFP4Scale)
	}

	y := make([]float32, count)
	blocks := k / NVFP4BlockSize
	for token := 0; token < rows; token++ {
		for row := 0; row < out; row++ {
			var sum float32
			for block := 0; block < blocks; block++ {
				wScale := DecodeE4M3(wScales[row*blocks+block])
				aScale := DecodeE4M3(aScales[token*blocks+block])
				for offset := 0; offset < NVFP4BlockSize; offset++ {
					inner := block*NVFP4BlockSize + offset
					wv := nvfp4LocalValue(wPacked, wScale, row, k, inner)
					av := nvfp4LocalValue(aPacked, aScale, token, k, inner)
					product := nvfp4RoundedF32(wv * av)
					sum = nvfp4RoundedF32(sum + product)
					if !nvfp4Finite(sum) {
						return Tensor{}, fmt.Errorf("%w: FP32 accumulation overflow at output [%d %d], K %d", ErrNVFP4Input, token, row, inner)
					}
				}
			}
			value := nvfp4RoundedF32(sum * alpha)
			if !nvfp4Finite(value) {
				return Tensor{}, fmt.Errorf("%w: FP32 epilogue overflow at output [%d %d]", ErrNVFP4Input, token, row)
			}
			y[token*out+row] = value
		}
	}
	return c.result([]int{rows, out}, y), nil
}

func nvfp4LocalValue(packed []byte, scale float32, row, cols, inner int) float32 {
	index := row*cols + inner
	code := packed[index/2] & 0x0f
	if inner&1 != 0 {
		code = packed[index/2] >> 4
	}
	return nvfp4RoundedF32(mxE2M1Values[code] * scale)
}

func nvfp4RoundedF32(value float32) float32 {
	return math.Float32frombits(math.Float32bits(value))
}

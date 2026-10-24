// Copyright (c) 2026 FlashInfer contributors.
// SPDX-License-Identifier: Apache-2.0
// Numerical staging adapted for Go; see THIRD_PARTY_NOTICES.md.

package compute

import (
	"fmt"
	"math"
)

// NewNVFP4 owns a row-major [out,in] W4A16 weight with in divisible by 16.
// HostBuffer.I8 holds packed E2M1 bytes followed by one raw E4M3FN scale per
// 16 weights; QuantSpec.Scale holds exactly one FP32 tensor scale. Even K is
// the low nibble. The CPU MatMul rounds F32 activations to BF16 RNE, widens
// exact E2M1*E4M3 products, accumulates serially in FP32, then applies the
// tensor scale once. This differs from GemvNVFP4Reference's early scaling.
// Construction does not enable an NVFP4 device backend or model loader.
func NewNVFP4(be Backend, shape []int, packed, blockScales []byte, globalScale float32) (Tensor, error) {
	weightBytes, scaleBytes, err := nvfp4PayloadShape(shape)
	if err != nil {
		return Tensor{}, err
	}
	if len(packed) != weightBytes {
		return Tensor{}, fmt.Errorf("%w: packed weights have %d bytes, want %d", ErrNVFP4Shape, len(packed), weightBytes)
	}
	if len(blockScales) != scaleBytes {
		return Tensor{}, fmt.Errorf("%w: block scales have %d bytes, want %d", ErrNVFP4Scale, len(blockScales), scaleBytes)
	}
	if err := nvfp4ValidateScales(blockScales, globalScale); err != nil {
		return Tensor{}, err
	}
	payload := make([]byte, weightBytes+scaleBytes)
	copy(payload, packed)
	copy(payload[weightBytes:], blockScales)
	q := &QuantSpec{Block: NVFP4BlockSize, Axis: 2, Bits: 4, Symmetric: true, Scale: []float32{globalScale}}
	return makeTensor(be, NVFP4, RowMajor, append([]int(nil), shape...), q,
		&hostBuf{i8: bytesAsI8(payload), nvfp4: true}), nil
}

func nvfp4PayloadShape(shape []int) (int, int, error) {
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%NVFP4BlockSize != 0 {
		return 0, 0, fmt.Errorf("%w: needs positive [out,in] with in divisible by %d", ErrNVFP4Shape, NVFP4BlockSize)
	}
	elements, ok := fp8Block32Product(shape[0], shape[1])
	if !ok {
		return 0, 0, fmt.Errorf("%w: element count overflows", ErrNVFP4Shape)
	}
	return elements / 2, elements / NVFP4BlockSize, nil
}

func nvfp4ValidateScales(scales []byte, globalScale float32) error {
	if !nvfp4Finite(globalScale) || globalScale < 0 {
		return fmt.Errorf("%w: tensor scale must be finite and non-negative", ErrNVFP4Scale)
	}
	for index, raw := range scales {
		value := DecodeE4M3(raw)
		if !nvfp4Finite(value) || value < 0 {
			return fmt.Errorf("%w: block scale %d must be finite and non-negative", ErrNVFP4Scale, index)
		}
	}
	return nil
}

func nvfp4Tensor(w Tensor) bool {
	storage, ok := w.buf.(*hostBuf)
	return w.Dtype == NVFP4 || (ok && storage != nil && storage.nvfp4)
}

func nvfp4WeightPayload(w Tensor) (packed, scales []byte, globalScale float32) {
	packed, scales, globalScale, err := checkedNVFP4WeightPayload(w)
	if err != nil {
		panic(err)
	}
	return packed, scales, globalScale
}

// checkedNVFP4WeightPayload validates the immutable host representation without
// changing the panic contract of the established W4A16 callers above.
func checkedNVFP4WeightPayload(w Tensor) (packed, scales []byte, globalScale float32, err error) {
	if w.Dtype != NVFP4 || w.Layout != RowMajor {
		return nil, nil, 0, fmt.Errorf("%w: requires row-major NVFP4 tensor", ErrNVFP4Shape)
	}
	weightBytes, scaleBytes, err := nvfp4PayloadShape(w.Shape)
	if err != nil {
		return nil, nil, 0, err
	}
	storage, ok := w.buf.(*hostBuf)
	if !ok || storage == nil || !storage.nvfp4 || storage.f32 != nil || len(storage.i8) != weightBytes+scaleBytes {
		return nil, nil, 0, fmt.Errorf("%w: invalid packed host payload", ErrNVFP4Shape)
	}
	q := w.Quant
	if q == nil || q.Block != NVFP4BlockSize || q.Axis != 2 || q.Bits != 4 ||
		!q.Symmetric || q.StaticAct || len(q.ZeroPoint) != 0 || len(q.Scale) != 1 {
		return nil, nil, 0, fmt.Errorf("%w: invalid NVFP4 quantization metadata", ErrNVFP4Scale)
	}
	payload := i8View(storage.i8)
	packed, scales, globalScale = payload[:weightBytes], payload[weightBytes:], q.Scale[0]
	if err := nvfp4ValidateScales(scales, globalScale); err != nil {
		return nil, nil, 0, err
	}
	return packed, scales, globalScale, nil
}

func nvfp4Activations(x Tensor, rows, cols int, batched bool) []float32 {
	count, ok := fp8Block32Product(rows, cols)
	validShape := len(x.Shape) == 1 && x.Shape[0] == cols
	if batched {
		validShape = len(x.Shape) == 2 && x.Shape[0] == rows && x.Shape[1] == cols
	}
	if !ok || rows <= 0 || !validShape || x.Dtype != F32 || x.Layout != RowMajor || x.Quant != nil {
		panic(fmt.Errorf("%w: requires matching row-major F32 activations", ErrNVFP4Input))
	}
	storage, host := x.buf.(*hostBuf)
	if !host || storage == nil || storage.nvfp4 || storage.i8 != nil || len(storage.f32) != count {
		panic(fmt.Errorf("%w: invalid activation host payload", ErrNVFP4Input))
	}
	rounded := make([]float32, count)
	for index, value := range storage.f32 {
		if !nvfp4Finite(value) {
			panic(fmt.Errorf("%w: non-finite activation %d", ErrNVFP4Input, index))
		}
		bits := math.Float32bits(value)
		bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
		rounded[index] = math.Float32frombits(bits)
		if !nvfp4Finite(rounded[index]) {
			panic(fmt.Errorf("%w: BF16 activation overflow at %d", ErrNVFP4Input, index))
		}
	}
	return rounded
}

func (c *cpuBackend) nvfp4MatMul(w, x Tensor, rows int, batched bool) Tensor {
	packed, scales, globalScale := nvfp4WeightPayload(w)
	out, in := w.Shape[0], w.Shape[1]
	count, ok := fp8Block32Product(rows, out)
	if !ok || rows <= 0 {
		panic(fmt.Errorf("%w: output shape overflows or is empty", ErrNVFP4Input))
	}
	activations := nvfp4Activations(x, rows, in, batched)
	y := make([]float32, count)
	blocks := in / NVFP4BlockSize
	for token := 0; token < rows; token++ {
		for row := 0; row < out; row++ {
			var sum float32
			for block := 0; block < blocks; block++ {
				scale := DecodeE4M3(scales[row*blocks+block])
				for j := 0; j < NVFP4BlockSize; j++ {
					k := block*NVFP4BlockSize + j
					pair := packed[(row*in+k)/2]
					code := pair & 15
					if k&1 != 0 {
						code = pair >> 4
					}
					// The finite local product fits BF16 exactly; the tensor scale
					// stays outside this loop to preserve the W4A16 staging.
					weight := mxE2M1Values[code] * scale
					product := float32(weight * activations[token*in+k])
					sum = float32(sum + product)
					if !nvfp4Finite(sum) {
						panic(fmt.Errorf("%w: FP32 accumulation overflow at row %d K %d", ErrNVFP4Input, row, k))
					}
				}
			}
			value := float32(sum * globalScale)
			if !nvfp4Finite(value) {
				panic(fmt.Errorf("%w: FP32 epilogue overflow at row %d", ErrNVFP4Input, row))
			}
			y[token*out+row] = value
		}
	}
	shape := []int{out}
	if batched {
		shape = []int{rows, out}
	}
	return c.result(shape, y)
}

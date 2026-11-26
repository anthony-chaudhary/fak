// Copyright (c) 2026 FlashInfer contributors.
// SPDX-License-Identifier: Apache-2.0
// Numerical staging adapted for Go; see THIRD_PARTY_NOTICES.md.

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"
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
	payload := make([]int8, weightBytes+scaleBytes)
	raw := i8View(payload)
	copy(raw, packed)
	copy(raw[weightBytes:], blockScales)
	return nvfp4HostTensor(be, shape, payload, globalScale, false), nil
}

// NewNVFP4Owned is NewNVFP4 for a payload already laid out as packed E2M1
// bytes followed by the block scales. The tensor takes ownership of payload
// without copying it, so the caller must not write it afterwards, including
// through HostBuffer.I8 on the returned tensor. The scales
// are validated once here, in parallel; because nothing else holds a writable
// alias, the upload path does not validate them again.
func NewNVFP4Owned(be Backend, shape []int, payload []byte, globalScale float32) (Tensor, error) {
	weightBytes, scaleBytes, err := nvfp4PayloadShape(shape)
	if err != nil {
		return Tensor{}, err
	}
	if len(payload) != weightBytes+scaleBytes {
		return Tensor{}, fmt.Errorf("%w: payload has %d bytes, want %d", ErrNVFP4Shape, len(payload), weightBytes+scaleBytes)
	}
	if !nvfp4Finite(globalScale) || globalScale < 0 {
		return Tensor{}, fmt.Errorf("%w: tensor scale must be finite and non-negative", ErrNVFP4Scale)
	}
	if index := nvfp4FirstInvalidScaleParallel(payload[weightBytes:]); index >= 0 {
		return Tensor{}, fmt.Errorf("%w: block scale %d must be finite and non-negative", ErrNVFP4Scale, index)
	}
	owned := unsafe.Slice((*int8)(unsafe.Pointer(&payload[0])), len(payload))
	return nvfp4HostTensor(be, shape, owned, globalScale, true), nil
}

func nvfp4HostTensor(be Backend, shape []int, payload []int8, globalScale float32, scalesValid bool) Tensor {
	q := &QuantSpec{Block: NVFP4BlockSize, Axis: 2, Bits: 4, Symmetric: true, Scale: []float32{globalScale}}
	return makeTensor(be, NVFP4, RowMajor, append([]int(nil), shape...), q,
		&hostBuf{i8: payload, nvfp4: true, nvfp4ScalesValid: scalesValid})
}

// nvfp4ScaleChunk keeps one validation worker's range large enough that
// goroutine startup is noise against the scan.
const nvfp4ScaleChunk = 1 << 20

// nvfp4FirstInvalidScaleParallel returns the same index as
// nvfp4FirstInvalidScale, splitting the scan across cores.
func nvfp4FirstInvalidScaleParallel(scales []byte) int {
	workers := min(runtime.GOMAXPROCS(0), len(scales)/nvfp4ScaleChunk)
	if workers <= 1 {
		return nvfp4FirstInvalidScale(scales)
	}
	found := make([]int, workers)
	var wg sync.WaitGroup
	for w := range workers {
		lo, hi := len(scales)*w/workers, len(scales)*(w+1)/workers
		wg.Add(1)
		go func() {
			defer wg.Done()
			found[w] = nvfp4FirstInvalidScale(scales[lo:hi])
			if found[w] >= 0 {
				found[w] += lo
			}
		}()
	}
	wg.Wait()
	for _, index := range found {
		if index >= 0 {
			return index
		}
	}
	return -1
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
	if index := nvfp4FirstInvalidScale(scales); index >= 0 {
		return fmt.Errorf("%w: block scale %d must be finite and non-negative", ErrNVFP4Scale, index)
	}
	return nil
}

// nvfp4ScaleByteValid is the bytewise form of "DecodeE4M3(b) is finite and not
// below zero": 0x00..0x7E are the non-negative finite codes, 0x7F is NaN, and
// with the sign bit set only negative zero (0x80) passes. Load validates about
// a billion scale bytes, so decoding each one to a float dominated model load.
func nvfp4ScaleByteValid(b byte) bool {
	return b < 0x7F || b == 0x80
}

// nvfp4FirstInvalidScale returns the index of the first scale byte that fails
// nvfp4ScaleByteValid, or -1. Eight bytes are tested per step: a word passes
// when no byte has its sign bit set and no byte's low seven bits are all ones;
// adding one to each masked byte carries into bit 7 exactly for 0x7F, and
// cannot carry across bytes. Words that fail fall back to the bytewise check.
func nvfp4FirstInvalidScale(scales []byte) int {
	const low7, ones, high = 0x7F7F7F7F7F7F7F7F, 0x0101010101010101, 0x8080808080808080
	index := 0
	for ; index+8 <= len(scales); index += 8 {
		word := binary.LittleEndian.Uint64(scales[index:])
		if (word|((word&low7)+ones))&high == 0 {
			continue
		}
		for offset, raw := range scales[index : index+8] {
			if !nvfp4ScaleByteValid(raw) {
				return index + offset
			}
		}
	}
	for ; index < len(scales); index++ {
		if !nvfp4ScaleByteValid(scales[index]) {
			return index
		}
	}
	return -1
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
	if !ok || storage == nil || !storage.nvfp4 || storage.f32 != nil {
		return nil, nil, 0, fmt.Errorf("%w: invalid packed host payload", ErrNVFP4Shape)
	}
	split := storage.nvfp4Scales != nil
	if split && (len(storage.i8) != weightBytes || len(storage.nvfp4Scales) != scaleBytes) ||
		!split && len(storage.i8) != weightBytes+scaleBytes {
		return nil, nil, 0, fmt.Errorf("%w: invalid packed host payload", ErrNVFP4Shape)
	}
	q := w.Quant
	if q == nil || q.Block != NVFP4BlockSize || q.Axis != 2 || q.Bits != 4 ||
		!q.Symmetric || q.StaticAct || len(q.ZeroPoint) != 0 || len(q.Scale) != 1 {
		return nil, nil, 0, fmt.Errorf("%w: invalid NVFP4 quantization metadata", ErrNVFP4Scale)
	}
	payload := i8View(storage.i8)
	packed, scales, globalScale = payload[:weightBytes], payload[weightBytes:], q.Scale[0]
	if split {
		scales = i8View(storage.nvfp4Scales)
	}
	if storage.nvfp4ScalesValid {
		if !nvfp4Finite(globalScale) || globalScale < 0 {
			return nil, nil, 0, fmt.Errorf("%w: tensor scale must be finite and non-negative", ErrNVFP4Scale)
		}
		return packed, scales, globalScale, nil
	}
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

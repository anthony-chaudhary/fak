package model

import (
	"encoding/binary"
	"fmt"
	"math"
)

// quantizeV4DenseFP8Q8 adapts Flash-0731's E8M0 scale grid to the bounded
// FP8-to-Q8 converter. It reuses fp8_quantize.go's attributed ktransformers LUT
// port while retaining decodeV4DenseFP8's scale and non-finite semantics. Only
// the scale grid widens before conversion, avoiding the full decoded-FP8
// temporary. Existing Q8 accelerator caches still apply. The model still stores
// Q8, not native FP8 (partial fak#12639).
func quantizeV4DenseFP8Q8(name string, shape []int, weights, scales []byte) (*q8Tensor, error) {
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 {
		return nil, fmt.Errorf("safetensors: V4 dense weight %s shape %v, want positive rank-2", name, shape)
	}
	rows, cols := shape[0], shape[1]
	wantWeights, ok := checkedShapeProduct(rows, cols)
	if !ok || len(weights) != wantWeights {
		return nil, fmt.Errorf("safetensors: V4 dense weight %s has %d bytes, shape %v implies %d", name, len(weights), shape, wantWeights)
	}
	if cols%qBlk != 0 {
		return nil, fmt.Errorf("safetensors: V4 dense weight %s input %d is not divisible by %d", name, cols, qBlk)
	}
	scaleRows, scaleCols := (rows-1)/fp8BlockDim+1, (cols-1)/fp8BlockDim+1
	wantScales, ok := checkedShapeProduct(scaleRows, scaleCols)
	if !ok || len(scales) != wantScales {
		return nil, fmt.Errorf("safetensors: V4 dense scale for %s has %d bytes, want %d", name, len(scales), wantScales)
	}
	scaleBytes, ok := checkedShapeProduct(wantScales, 4)
	if !ok {
		return nil, fmt.Errorf("safetensors: V4 dense scale for %s overflows float32 byte count", name)
	}
	// Preserve the reference's failure ordering: weight NaNs, then scale NaNs,
	// then finite operands whose product overflows. No Q8 allocation precedes
	// validation of the complete pair.
	for i, value := range weights {
		if value&0x7f == 0x7f {
			return nil, fmt.Errorf("safetensors: V4 dense weight %s byte %d is E4M3 NaN", name, i)
		}
	}
	scaleF32 := make([]byte, scaleBytes)
	for i, value := range scales {
		if value == 0xff {
			return nil, fmt.Errorf("safetensors: V4 dense scale for %s byte %d is E8M0 NaN", name, i)
		}
		scale := float32(math.Ldexp(1, int(value)-127))
		if math.IsInf(float64(scale), 0) {
			return nil, fmt.Errorf("safetensors: V4 dense scale for %s byte %d is non-finite", name, i)
		}
		binary.LittleEndian.PutUint32(scaleF32[i*4:], math.Float32bits(scale))
	}
	for row := 0; row < rows; row++ {
		scaleRow := (row / fp8BlockDim) * scaleCols
		for col := 0; col < cols; col++ {
			i := row*cols + col
			si := scaleRow + col/fp8BlockDim
			scale := math.Float32frombits(binary.LittleEndian.Uint32(scaleF32[si*4:]))
			value := fp8E4M3Lookup(weights[i]) * scale
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("safetensors: V4 dense weight %s decoded value %d is non-finite", name, i)
			}
		}
	}
	return quantizeFP8BlockScaleQ8(name, shape, weights, scaleF32)
}

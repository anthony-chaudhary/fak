package compute

import (
	"errors"
	"fmt"
	"math"
)

const NVFP4BlockSize = 16

var (
	ErrNVFP4Shape = errors.New("compute: invalid NVFP4 shape")
	ErrNVFP4Scale = errors.New("compute: invalid NVFP4 scale")
	ErrNVFP4Input = errors.New("compute: invalid NVFP4 input")
)

// DecodeNVFP4Reference decodes an unswizzled row-major [rows, cols] NVFP4 matrix.
// Every group of 16 K values has one E4M3FN scale byte. Packed E2M1 values place
// even K in the low nibble. This CPU reference implements value = code * blockScale
// * globalScale; it does not describe a padded or 2-D device-kernel scale layout.
func DecodeNVFP4Reference(rows, cols int, packed, blockScales []byte, globalScale float32) ([]float32, error) {
	elems, scaleCols, scales, err := validateNVFP4(rows, cols, packed, blockScales, globalScale)
	if err != nil {
		return nil, err
	}

	decoded := make([]float32, elems)
	for row := 0; row < rows; row++ {
		for k := 0; k < cols; k++ {
			index := row*cols + k
			code := packed[index/2] & 0x0f
			if k%2 == 1 {
				code = packed[index/2] >> 4
			}
			value := mxE2M1Values[code] * scales[row*scaleCols+k/NVFP4BlockSize]
			value *= globalScale
			if !nvfp4Finite(value) {
				return nil, fmt.Errorf("%w: decoded value at row %d K %d is non-finite", ErrNVFP4Scale, row, k)
			}
			decoded[index] = value
		}
	}
	return decoded, nil
}

// DecodeNVFP4Stacks decodes stacks of unswizzled row-major [rows, cols] NVFP4
// matrices laid out contiguously: stack e owns packed[e*rows*cols/2:] and
// blockScales[e*rows*scaleCols:], and every value in stack e is multiplied by
// its own global f32 scale, mirroring per-expert weight_scale_2 tensors. Each
// group of 16 K values has one E4M3FN scale byte; packed E2M1 values place even
// K in the low nibble. The output holds stacks*rows*cols values in stack order,
// so an output-length mismatch cannot occur. NaN or negative block scales and
// non-finite or negative globals refuse with ErrNVFP4Scale; a global-scale
// count that does not match the stack count refuses with ErrNVFP4Shape.
func DecodeNVFP4Stacks(stacks, rows, cols int, packed, blockScales []byte, globalScales []float32) ([]float32, error) {
	if stacks <= 0 {
		return nil, fmt.Errorf("%w: need at least one stack, got %d", ErrNVFP4Shape, stacks)
	}
	if len(globalScales) != stacks {
		return nil, fmt.Errorf("%w: %d global scales for %d stacks", ErrNVFP4Shape, len(globalScales), stacks)
	}
	elems, scaleCols, scales, err := validateNVFP4Matrix(rows, cols, packed, blockScales, stacks)
	if err != nil {
		return nil, err
	}
	for i, g := range globalScales {
		if !nvfp4Finite(g) || g < 0 {
			return nil, fmt.Errorf("%w: global scale %d must be finite and non-negative", ErrNVFP4Scale, i)
		}
	}

	perStack := rows * cols // validated to fit by validateNVFP4Matrix
	decoded := make([]float32, elems)
	for stack := 0; stack < stacks; stack++ {
		global := globalScales[stack]
		base := stack * perStack
		scaleBase := base / NVFP4BlockSize
		for row := 0; row < rows; row++ {
			for k := 0; k < cols; k++ {
				index := base + row*cols + k
				code := packed[index/2] & 0x0f
				if k%2 == 1 {
					code = packed[index/2] >> 4
				}
				value := mxE2M1Values[code] * scales[scaleBase+row*scaleCols+k/NVFP4BlockSize]
				value *= global
				if !nvfp4Finite(value) {
					return nil, fmt.Errorf("%w: decoded value at stack %d row %d K %d is non-finite", ErrNVFP4Scale, stack, row, k)
				}
				decoded[index] = value
			}
		}
	}
	return decoded, nil
}

// GemvNVFP4Reference computes y = weight*x using DecodeNVFP4Reference's CPU layout.
func GemvNVFP4Reference(rows, cols int, packed, blockScales []byte, globalScale float32, x []float32) ([]float32, error) {
	decoded, err := DecodeNVFP4Reference(rows, cols, packed, blockScales, globalScale)
	if err != nil {
		return nil, err
	}
	if len(x) != cols {
		return nil, fmt.Errorf("%w: vector has %d values, want %d", ErrNVFP4Input, len(x), cols)
	}
	for k, value := range x {
		if !nvfp4Finite(value) {
			return nil, fmt.Errorf("%w: vector value %d is non-finite", ErrNVFP4Input, k)
		}
	}

	out := make([]float32, rows)
	for row := 0; row < rows; row++ {
		var sum float32
		for k := 0; k < cols; k++ {
			sum += decoded[row*cols+k] * x[k]
			if !nvfp4Finite(sum) {
				return nil, fmt.Errorf("%w: accumulation at row %d K %d is non-finite", ErrNVFP4Input, row, k)
			}
		}
		out[row] = sum
	}
	return out, nil
}

func validateNVFP4(rows, cols int, packed, blockScales []byte, globalScale float32) (int, int, []float32, error) {
	elems, scaleCols, scales, err := validateNVFP4Matrix(rows, cols, packed, blockScales, 1)
	if err != nil {
		return 0, 0, nil, err
	}
	if !nvfp4Finite(globalScale) || globalScale < 0 {
		return 0, 0, nil, fmt.Errorf("%w: global scale must be finite and non-negative", ErrNVFP4Scale)
	}
	return elems, scaleCols, scales, nil
}

// validateNVFP4Matrix validates copies of a [rows, cols] NVFP4 matrix laid out
// contiguously (stacked experts) and decodes every e4m3 block scale, refusing
// NaN or negative scales before any decode runs.
func validateNVFP4Matrix(rows, cols int, packed, blockScales []byte, copies int) (int, int, []float32, error) {
	if rows <= 0 || cols <= 0 || cols%NVFP4BlockSize != 0 {
		return 0, 0, nil, fmt.Errorf("%w: [%d %d] needs positive dimensions and K divisible by %d", ErrNVFP4Shape, rows, cols, NVFP4BlockSize)
	}
	if copies <= 0 {
		return 0, 0, nil, fmt.Errorf("%w: need at least one matrix, got %d", ErrNVFP4Shape, copies)
	}
	matrixElems, ok := fp8Block32Product(rows, cols)
	if !ok {
		return 0, 0, nil, fmt.Errorf("%w: [%d %d] overflows element count", ErrNVFP4Shape, rows, cols)
	}
	elems, ok := fp8Block32Product(matrixElems, copies)
	if !ok {
		return 0, 0, nil, fmt.Errorf("%w: %d matrices of [%d %d] overflow element count", ErrNVFP4Shape, copies, rows, cols)
	}
	if len(packed) != elems/2 {
		return 0, 0, nil, fmt.Errorf("%w: packed weights have %d bytes, want %d", ErrNVFP4Shape, len(packed), elems/2)
	}
	scaleCols := cols / NVFP4BlockSize
	matrixScales, ok := fp8Block32Product(rows, scaleCols)
	if !ok {
		return 0, 0, nil, fmt.Errorf("%w: [%d %d] overflows scale count", ErrNVFP4Shape, rows, scaleCols)
	}
	scaleCount, ok := fp8Block32Product(matrixScales, copies)
	if !ok {
		return 0, 0, nil, fmt.Errorf("%w: %d matrices of [%d %d] overflow scale count", ErrNVFP4Shape, copies, rows, scaleCols)
	}
	if len(blockScales) != scaleCount {
		return 0, 0, nil, fmt.Errorf("%w: block scales have %d bytes, want %d", ErrNVFP4Scale, len(blockScales), scaleCount)
	}

	scales := make([]float32, scaleCount)
	for i, raw := range blockScales {
		scale := DecodeE4M3(raw)
		if !nvfp4Finite(scale) || scale < 0 {
			return 0, 0, nil, fmt.Errorf("%w: block scale %d must be finite and non-negative", ErrNVFP4Scale, i)
		}
		scales[i] = scale
	}
	return elems, scaleCols, scales, nil
}

func nvfp4Finite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

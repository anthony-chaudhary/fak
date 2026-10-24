// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified and ported to Go as a CPU reference implementation.

package compute

import (
	"fmt"
	"math"
)

const (
	nvfp4E2M1Max       = 6.0
	nvfp4E4M3Max       = 448.0
	nvfp4E4M3Min       = 1.0 / 512.0
	nvfp4CombinedMax   = nvfp4E2M1Max * nvfp4E4M3Max
	nvfp4ZeroBlockCode = byte(0x38) // ModelOpt's canonical unit scale for a zero block.
)

// QuantizeNVFP4Reference quantizes a finite row-major [rows, cols] matrix into
// the unswizzled layout consumed by DecodeNVFP4Reference. The numerical recipe
// follows NVIDIA Model Optimizer at commit 0638a274cf195870564d8e3c164e4f859e5be2a2:
// one FP32 tensor scale and one positive E4M3FN scale per 16 consecutive K
// values, followed by saturating E2M1 round-to-nearest-even conversion.
//
// An all-zero matrix has the canonical representation of zero packed data,
// unit block scales, and a zero global scale. A nonzero tensor whose global
// scale cannot be represented as a positive float32 is rejected.
func QuantizeNVFP4Reference(rows, cols int, values []float32) (packed, blockScales []byte, globalScale float32, err error) {
	weightBytes, scaleBytes, err := nvfp4PayloadShape([]int{rows, cols})
	if err != nil {
		return nil, nil, 0, err
	}
	elements := weightBytes * 2
	if len(values) != elements {
		return nil, nil, 0, fmt.Errorf("%w: values have %d elements, want %d", ErrNVFP4Shape, len(values), elements)
	}

	var tensorAMax float32
	for index, value := range values {
		if !nvfp4Finite(value) {
			return nil, nil, 0, fmt.Errorf("%w: value %d is non-finite", ErrNVFP4Input, index)
		}
		magnitude := float32(math.Abs(float64(value)))
		if magnitude > tensorAMax {
			tensorAMax = magnitude
		}
	}

	packed = make([]byte, weightBytes)
	blockScales = make([]byte, scaleBytes)
	if tensorAMax == 0 {
		for index := range blockScales {
			blockScales[index] = nvfp4ZeroBlockCode
		}
		return packed, blockScales, 0, nil
	}

	globalScale = float32(float64(tensorAMax) / nvfp4CombinedMax)
	if globalScale == 0 || !nvfp4Finite(globalScale) {
		return nil, nil, 0, fmt.Errorf("%w: tensor maximum %g has no positive finite FP32 global scale", ErrNVFP4Scale, tensorAMax)
	}
	// Keep the largest reconstructable E2M1*E4M3 value finite. Rounding the
	// division above upward by one ULP can otherwise overflow at MaxFloat32.
	for !nvfp4Finite(float32(nvfp4CombinedMax) * globalScale) {
		globalScale = math.Nextafter32(globalScale, 0)
		if globalScale == 0 {
			return nil, nil, 0, fmt.Errorf("%w: tensor maximum %g has no safe FP32 global scale", ErrNVFP4Scale, tensorAMax)
		}
	}

	scaleCols := cols / NVFP4BlockSize
	for row := 0; row < rows; row++ {
		for block := 0; block < scaleCols; block++ {
			start := row*cols + block*NVFP4BlockSize
			var blockAMax float32
			for offset := 0; offset < NVFP4BlockSize; offset++ {
				magnitude := float32(math.Abs(float64(values[start+offset])))
				if magnitude > blockAMax {
					blockAMax = magnitude
				}
			}

			scaleIndex := row*scaleCols + block
			if blockAMax == 0 {
				blockScales[scaleIndex] = nvfp4ZeroBlockCode
				continue
			}
			scale := float64(blockAMax) / (nvfp4E2M1Max * float64(globalScale))
			if scale < nvfp4E4M3Min {
				scale = nvfp4E4M3Min
			} else if scale > nvfp4E4M3Max {
				scale = nvfp4E4M3Max
			}
			rawScale := nvfp4EncodePositiveE4M3RNE(scale)
			blockScales[scaleIndex] = rawScale
			effectiveScale := float64(globalScale) * float64(DecodeE4M3(rawScale))
			if effectiveScale == 0 || math.IsInf(effectiveScale, 0) || math.IsNaN(effectiveScale) {
				return nil, nil, 0, fmt.Errorf("%w: block scale %d is not usable", ErrNVFP4Scale, scaleIndex)
			}

			for offset := 0; offset < NVFP4BlockSize; offset++ {
				index := start + offset
				code := nvfp4EncodeE2M1RNE(float64(values[index]) / effectiveScale)
				if index&1 == 0 {
					packed[index/2] = code
				} else {
					packed[index/2] |= code << 4
				}
			}
		}
	}
	return packed, blockScales, globalScale, nil
}

// QuantizeNVFP4 builds an immutable NVFP4 tensor from finite row-major values.
func QuantizeNVFP4(be Backend, shape []int, values []float32) (Tensor, error) {
	if len(shape) != 2 {
		return Tensor{}, fmt.Errorf("%w: needs rank-2 [out,in] shape", ErrNVFP4Shape)
	}
	packed, scales, global, err := QuantizeNVFP4Reference(shape[0], shape[1], values)
	if err != nil {
		return Tensor{}, err
	}
	return NewNVFP4(be, shape, packed, scales, global)
}

func nvfp4EncodeE2M1RNE(value float64) byte {
	sign := byte(0)
	if math.Signbit(value) {
		sign = 8
		value = -value
	}
	if value >= nvfp4E2M1Max {
		return sign | 7
	}
	best := byte(0)
	bestDistance := math.Inf(1)
	for code := byte(0); code < 8; code++ {
		distance := math.Abs(value - float64(mxE2M1Values[code]))
		if distance < bestDistance || distance == bestDistance && code&1 == 0 {
			best, bestDistance = code, distance
		}
	}
	return sign | best
}

func nvfp4EncodePositiveE4M3RNE(value float64) byte {
	if value <= 0 {
		return 0
	}
	if value >= nvfp4E4M3Max {
		return 0x7e
	}
	best := byte(0)
	bestDistance := math.Inf(1)
	for raw := byte(0); raw <= 0x7e; raw++ {
		distance := math.Abs(value - float64(DecodeE4M3(raw)))
		if distance < bestDistance || distance == bestDistance && raw&1 == 0 {
			best, bestDistance = raw, distance
		}
	}
	return best
}

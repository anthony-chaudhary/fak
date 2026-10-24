// Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified Go scale-layout adapter for fak; see the pinned source below.

package compute

import "fmt"

const (
	nvfp4ScaleTileRows  = 128
	nvfp4ScaleTileCols  = 4
	nvfp4ScaleTileBytes = nvfp4ScaleTileRows * nvfp4ScaleTileCols
)

// NVFP4 block-scale layout is adapted from NVIDIA TensorRT-LLM (Apache-2.0),
// pinned at ecae3c97e2b646580415518a45a72bc5a5a67075:
// https://github.com/NVIDIA/TensorRT-LLM/blob/ecae3c97e2b646580415518a45a72bc5a5a67075/cpp/tensorrt_llm/kernels/quantization.h
// License: https://github.com/NVIDIA/TensorRT-LLM/blob/ecae3c97e2b646580415518a45a72bc5a5a67075/LICENSE
//
// The byte permutation groups scaleCols (K/16) into 128-row by 4-column
// tiles. It changes only the scale layout; it does not encode or decode FP4.

// SwizzleNVFP4Scales converts row-major NVFP4 block-scale bytes into the
// TensorRT-LLM 128x4 tiled layout. The returned buffer includes zero-filled
// padding to complete the final row and column tiles.
func SwizzleNVFP4Scales(rows, scaleCols int, linear []byte) ([]byte, error) {
	linearLen, swizzledLen, tileCols, err := nvfp4ScaleLayoutGeometry(rows, scaleCols)
	if err != nil {
		return nil, err
	}
	if len(linear) != linearLen {
		return nil, fmt.Errorf("%w: linear scale layout has %d bytes, want %d", ErrNVFP4Scale, len(linear), linearLen)
	}

	swizzled := make([]byte, swizzledLen)
	for row := 0; row < rows; row++ {
		for col := 0; col < scaleCols; col++ {
			swizzled[nvfp4ScaleLayoutOffset(row, col, tileCols)] = linear[row*scaleCols+col]
		}
	}
	return swizzled, nil
}

// UnswizzleNVFP4Scales converts a canonical TensorRT-LLM 128x4 tiled scale
// buffer to row-major order and strips its zero padding.
func UnswizzleNVFP4Scales(rows, scaleCols int, swizzled []byte) ([]byte, error) {
	linearLen, swizzledLen, tileCols, err := nvfp4ScaleLayoutGeometry(rows, scaleCols)
	if err != nil {
		return nil, err
	}
	if len(swizzled) != swizzledLen {
		return nil, fmt.Errorf("%w: swizzled scale layout has %d bytes, want %d", ErrNVFP4Scale, len(swizzled), swizzledLen)
	}

	paddedRows := swizzledLen / (tileCols * nvfp4ScaleTileCols)
	paddedCols := tileCols * nvfp4ScaleTileCols
	for row := 0; row < paddedRows; row++ {
		for col := 0; col < paddedCols; col++ {
			if (row >= rows || col >= scaleCols) && swizzled[nvfp4ScaleLayoutOffset(row, col, tileCols)] != 0 {
				return nil, fmt.Errorf("%w: non-zero padding at row %d scale column %d", ErrNVFP4Scale, row, col)
			}
		}
	}

	linear := make([]byte, linearLen)
	for row := 0; row < rows; row++ {
		for col := 0; col < scaleCols; col++ {
			linear[row*scaleCols+col] = swizzled[nvfp4ScaleLayoutOffset(row, col, tileCols)]
		}
	}
	return linear, nil
}

func nvfp4ScaleLayoutGeometry(rows, scaleCols int) (linearLen, swizzledLen, tileCols int, err error) {
	if rows <= 0 || scaleCols <= 0 {
		return 0, 0, 0, fmt.Errorf("%w: rows and scaleCols must be positive, got [%d %d]", ErrNVFP4Shape, rows, scaleCols)
	}

	maxInt := int(^uint(0) >> 1)
	if rows > maxInt/scaleCols {
		return 0, 0, 0, fmt.Errorf("%w: [%d %d] overflows linear scale count", ErrNVFP4Shape, rows, scaleCols)
	}
	linearLen = rows * scaleCols

	paddedRows, ok := nvfp4ScaleRoundUp(rows, nvfp4ScaleTileRows)
	if !ok {
		return 0, 0, 0, fmt.Errorf("%w: row count %d overflows 128-row scale tiling", ErrNVFP4Shape, rows)
	}
	paddedCols, ok := nvfp4ScaleRoundUp(scaleCols, nvfp4ScaleTileCols)
	if !ok || paddedRows > maxInt/paddedCols {
		return 0, 0, 0, fmt.Errorf("%w: [%d %d] overflows padded scale layout", ErrNVFP4Shape, rows, scaleCols)
	}
	swizzledLen = paddedRows * paddedCols
	tileCols = paddedCols / nvfp4ScaleTileCols
	return linearLen, swizzledLen, tileCols, nil
}

func nvfp4ScaleRoundUp(value, multiple int) (int, bool) {
	maxInt := int(^uint(0) >> 1)
	if value > maxInt-(multiple-1) {
		return 0, false
	}
	return ((value + multiple - 1) / multiple) * multiple, true
}

func nvfp4ScaleLayoutOffset(row, col, tileCols int) int {
	return ((row/nvfp4ScaleTileRows)*tileCols+col/nvfp4ScaleTileCols)*nvfp4ScaleTileBytes +
		(row%32)*16 + ((row%nvfp4ScaleTileRows)/32)*4 + col%nvfp4ScaleTileCols
}

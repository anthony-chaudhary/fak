package model

import (
	"fmt"
	"math"
)

// v41CombinedAttention composes already-resolved logical window rows with
// compressed source rows in ONE sparse sink contraction. compressedIDs are
// canonical source-group IDs, never already offset by the window width. Repeated
// IDs and the same token's separate window/compressed representations remain
// independent contributions to the common denominator.
//
// Source: inference/model.py:700-780 at dba1be0a40aa45a94ad051997016db3960a90277.
// This internal projected-input seam does not supply FP8 window or FP4 index/KV
// quantization and does not widen ratio-one public forward admission. Inputs are
// already normalized/rotated; the caller applies inverse output RoPE once after
// this combined result, never separately to two attention streams.
func v41CombinedAttention(layer, pos, sourceRatio, heads, headDim int, q, sink []float32, window, compressed [][]float32, compressedIDs []int32, scale float32, attend v41SharedAttentionFunc) ([]float32, error) {
	fail := func(message string) ([]float32, error) {
		return nil, v41StageErr(v41StageAttention, layer, fmt.Errorf("%w: combined attention %s", ErrV41ForwardStage, message))
	}
	if layer < 0 || pos < 0 || sourceRatio <= 0 || heads <= 0 || headDim <= 0 {
		return fail("has invalid position or geometry")
	}
	maxInt := int(^uint(0) >> 1)
	if len(window) > math.MaxInt32 || len(compressed) > math.MaxInt32-len(window) || len(compressedIDs) > maxInt-len(window) {
		return fail("row or index count overflows")
	}
	rows := len(window) + len(compressed)
	slots := len(window) + len(compressedIDs)
	qCount, qOK := checkedMulInt(heads, headDim)
	kvCount, kvOK := checkedMulInt(rows, headDim)
	_, qBytesOK := checkedMulInt(qCount, 4)
	_, kvBytesOK := checkedMulInt(kvCount, 4)
	_, idxBytesOK := checkedMulInt(slots, 4)
	if !qOK || !kvOK || !qBytesOK || !kvBytesOK || !idxBytesOK || (attend != nil && slots > math.MaxInt32) ||
		len(q) != qCount || (sink != nil && len(sink) != heads) {
		return fail("input shape or byte count is invalid")
	}
	if !finite32(scale) || scale == 0 {
		return fail("scale must be finite and nonzero")
	}
	for _, values := range [][]float32{q, sink} {
		if err := finiteRow32(values, "combined attention input"); err != nil {
			return nil, v41StageErr(v41StageAttention, layer, err)
		}
	}
	// Validate every original row and ID before causal masking. An invalid future
	// row or padding ID cannot disappear merely because this query cannot see it.
	for _, stream := range [][][]float32{window, compressed} {
		for _, row := range stream {
			if len(row) != headDim {
				return fail("row width differs from head width")
			}
			if err := finiteRow32(row, "combined attention row"); err != nil {
				return nil, v41StageErr(v41StageAttention, layer, err)
			}
		}
	}
	for _, id := range compressedIDs {
		if id < -1 || int(id) >= len(compressed) {
			return fail("canonical compressed row ID is out of range")
		}
	}
	flat := make([]float32, 0, kvCount)
	idx := make([]int32, 0, slots)
	for i, row := range window {
		flat = append(flat, row...)
		idx = append(idx, int32(i))
	}
	for _, row := range compressed {
		flat = append(flat, row...)
	}
	for _, id := range compressedIDs {
		// Avoid (pos+1)/ratio and (id+1)*ratio overflow at extreme positions.
		if id < 0 || pos < sourceRatio-1 || int(id) > (pos-(sourceRatio-1))/sourceRatio {
			idx = append(idx, -1)
		} else {
			idx = append(idx, int32(len(window)+int(id)))
		}
	}
	out, err := v41SparseAttentionSinkWithDevice(layer, q, flat, sink, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: heads, HeadDim: headDim, N: rows, TopK: len(idx), Softmax: scale,
	}, attend)
	if err != nil {
		return nil, err
	}
	if len(out) != qCount {
		_, err = fail("returned an invalid output width")
	} else if finiteErr := finiteRow32(out, "combined attention output"); finiteErr != nil {
		err = v41StageErr(v41StageAttention, layer, finiteErr)
	}
	if err != nil {
		if attend != nil {
			return nil, &V41SharedAttentionOperationError{Layer: layer, Cause: err}
		}
		return nil, err
	}
	return append([]float32(nil), out...), nil
}

package model

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
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
	return v41CombinedAttentionResident(layer, pos, sourceRatio, heads, headDim, q, sink, window, compressed, compressedIDs, scale, nil, attend)
}

func v41CombinedAttentionResident(layer, pos, sourceRatio, heads, headDim int, q, sink []float32, window, compressed [][]float32, compressedIDs []int32, scale float32, resident *v41ResidentKVView, attend v41SharedAttentionFunc) ([]float32, error) {
	return v41CombinedAttentionWithContinuation(layer, pos, sourceRatio, heads, headDim, q, sink, window, compressed, compressedIDs, scale, resident, attend, nil)
}

func v41CombinedAttentionWithContinuation(layer, pos, sourceRatio, heads, headDim int, q, sink []float32, window, compressed [][]float32, compressedIDs []int32, scale float32, resident *v41ResidentKVView, attend v41SharedAttentionFunc, continuation *v41SharedAttentionContinuation) ([]float32, error) {
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
	idx := make([]int32, 0, slots)
	for i := range window {
		idx = append(idx, int32(i))
	}
	for _, id := range compressedIDs {
		// Avoid (pos+1)/ratio and (id+1)*ratio overflow at extreme positions.
		if id < 0 || pos < sourceRatio-1 || int(id) > (pos-(sourceRatio-1))/sourceRatio {
			idx = append(idx, -1)
		} else {
			idx = append(idx, int32(len(window)+int(id)))
		}
	}
	opt := V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: heads, HeadDim: headDim, N: rows, TopK: len(idx), Softmax: scale,
	}
	var out []float32
	var ownedOutput bool
	var err error
	if attend != nil && resident != nil && !v41ResidentKVDisabled {
		request := v41SharedAttentionRequest{
			mode: compute.V41SharedAttentionPlain, q: q, sink: sink, idx: idx, plain: opt,
			rowKV:    &v41SharedAttentionRows{window: window, compressed: compressed},
			resident: resident, sinkKey: resident.sinkKey,
		}
		c := continuation
		if c != nil && !c.used && c.layer == layer && c.pos == pos && c.sourceRatio == sourceRatio &&
			c.heads == heads && c.headDim == headDim && c.scale == scale && c.live() {
			// All original rows were checked above. Nothing can call user code
			// between sealing this local request and the private preparation.
			validation := &v41SharedAttentionValidation{continuation: c, request: &request, rows: request.rowKV,
				window: window, compressed: compressed}
			var result v41SharedAttentionResult
			result, err = c.binding.native.core.validated(layer, &request, validation)
			out, ownedOutput = result.values, err == nil
			if err != nil {
				err = &V41SharedAttentionOperationError{Layer: layer, Cause: v41StageErr(v41StageAttention, layer, err)}
			}
		} else {
			out, err = v41SelectedSharedAttention(layer, request, attend)
		}
	} else {
		flat := make([]float32, 0, kvCount)
		for _, stream := range [][][]float32{window, compressed} {
			for _, row := range stream {
				flat = append(flat, row...)
			}
		}
		out, err = v41SparseAttentionSinkResident(layer, q, flat, sink, idx, opt, resident, attend)
	}
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
	// The nil callback uses the host primitive, which owns its fresh result.
	// Ordinary callbacks can return borrowed storage and still need a copy.
	if ownedOutput || attend == nil {
		return out, nil
	}
	return append([]float32(nil), out...), nil
}

package model

import (
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Prepare outside the attention timer, just as the original flat packing was.
// Borrow only fully valid cold plain inputs with an already-admitted owner.
// The generic row and flat validators have different error precedence: any
// questionable input must retain the original flat path and its first error.
// This predicate performs no dispatch, device work, or state mutation. The
// selected adapter still performs its complete validation after admission.
func v41ColdPlainAttentionInput(q, sink []float32, kvRows [][]float32, keys, idx []int32, opt V41SparseAttentionSinkOptions, owner *V41AttentionState, attend v41SharedAttentionFunc) (flat []float32, window [][]float32) {
	if attend != nil && owner != nil && !v41ResidentKVDisabled &&
		v41ColdPlainAttentionCanBorrow(q, sink, kvRows, keys, idx, opt) {
		first, end := int(keys[0]), int(keys[len(keys)-1])+1
		return nil, kvRows[first:end:end]
	}
	// Keep historical packing even for malformed rows: the flat validator, not
	// admission, owns their error and its ordering relative to other inputs.
	flat = make([]float32, 0, len(keys)*opt.HeadDim)
	for _, key := range keys {
		flat = append(flat, kvRows[key]...)
	}
	return flat, nil
}

func v41ColdPlainAttentionCanBorrow(q, sink []float32, kvRows [][]float32, keys, idx []int32, opt V41SparseAttentionSinkOptions) bool {
	if opt.B != 1 || opt.M != 1 || opt.TopK < 0 || opt.TopK > math.MaxInt32 || opt.Inverse != nil ||
		len(keys) == 0 || len(keys) != opt.N || len(idx) != opt.TopK ||
		!finite32(opt.Softmax) || opt.Softmax == 0 {
		return false
	}
	if opt.TopKLength != nil && (len(opt.TopKLength) != 1 || opt.TopKLength[0] < 0 || int(opt.TopKLength[0]) > opt.TopK) {
		return false
	}
	qSize, _, err := v41SharedAttentionGeometry(opt.Heads, opt.HeadDim, opt.N)
	if err != nil || len(q) != qSize || (sink != nil && len(sink) != opt.Heads) {
		return false
	}
	for _, values := range [][]float32{q, sink} {
		for _, value := range values {
			if !finite32(value) {
				return false
			}
		}
	}
	for i, key := range keys {
		if key < 0 || int(key) >= len(kvRows) || (i > 0 && int64(key) != int64(keys[i-1])+1) {
			return false
		}
		row := kvRows[key]
		if len(row) != opt.HeadDim {
			return false
		}
		for _, value := range row {
			if !finite32(value) {
				return false
			}
		}
	}
	for _, row := range idx {
		if row < -1 || int(row) >= opt.N {
			return false
		}
	}
	return true
}

// Dispatch once at the existing timed boundary. Borrowed rows live only for
// this synchronous callback. A capability decline inside the adapter keeps its
// existing selected-row packing; no new backend capability is assumed here.
func v41ColdPlainAttentionSink(layer int, q, flat, sink []float32, window [][]float32, idx []int32, opt V41SparseAttentionSinkOptions, resident *v41ResidentKVView, attend v41SharedAttentionFunc) ([]float32, error) {
	// Recheck the original dispatch guards after the timer boundary. If a
	// hook changed either, reconstruct the original flat payload rather than
	// assuming admission permanently bound residency or a selected callback.
	if window != nil && (attend == nil || v41ResidentKVDisabled) {
		flat = make([]float32, 0, len(window)*opt.HeadDim)
		for _, row := range window {
			flat = append(flat, row...)
		}
		window = nil
	}
	if window == nil {
		return v41SparseAttentionSinkResident(layer, q, flat, sink, idx, opt, resident, attend)
	}
	return v41SelectedSharedAttention(layer, v41SharedAttentionRequest{
		mode: compute.V41SharedAttentionPlain, q: q, sink: sink, idx: idx, plain: opt,
		rowKV: &v41SharedAttentionRows{window: window}, resident: resident, sinkKey: resident.sinkKey,
	}, attend)
}

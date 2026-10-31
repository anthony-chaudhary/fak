package model

// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2026 Shih-Ying Yeh (aka KohakuBlueLeaf, KBlueLeaf).
// Modified native Go translation for FAK: bounded per-session full/ring owners,
// rather than Tokenomos's batched GPU pools. Source: KohakuBlueleaf/Tokenomos
// eeb47ee09a3c5e66411e90313b017c211d6a3fbb, models/contract.py,
// core/pools.py, models/blocks/attention.py, kernels/attention/paged.py.
// The complete Apache-2.0 license is retained in the repository root LICENSE.
// Preserve the donor's separate K/V planes, absolute-position ring addressing,
// and attention-before-write ordering. SharedKV/recurrent/device pools are NOT
// implemented by this owner.

import (
	"fmt"
	"math"
)

// attentionRowSpec describes one layer, not a uniform model-wide KV stride.
// Window < 0 is full causal; Window > 0 retains exactly that many positions.
type attentionRowSpec struct {
	KVHeads, HeadDim, Window int
}

type attentionLayerRows struct {
	spec             attentionRowSpec
	stride, capacity int
	k, v             []float32
}

// attentionStateRows owns a bounded continuation, including its exact token
// history. It never aliases a KVCache or a transient projection. Full layers
// reserve MaxTokens rows once; ring layers reserve min(Window, MaxTokens).
// No state allocation occurs during publish. The byte charge covers the owned
// K/V and reserved history payload, not Go headers, allocator rounding, transient
// projections/scores/logits, model weights or process RSS. Each clone owns and
// charges a separate full reservation; there is no shared mutable scratch.
type attentionStateRows struct {
	layers       []attentionLayerRows
	history      []int
	maxTokens    int
	payloadBytes int64
}

// attentionStateRowBytes validates EVERY layer and product before allocating
// state. An explicit budget is required; zero never means unlimited.
func attentionStateRowBytes(specs []attentionRowSpec, maxTokens int, budget int64) (int64, error) {
	if len(specs) == 0 || maxTokens <= 0 || budget <= 0 || int64(maxTokens) > math.MaxInt64/hostPositionBytes {
		return 0, fmt.Errorf("model: invalid attention row reservation")
	}
	bytes := int64(maxTokens) * hostPositionBytes
	for l, spec := range specs {
		stride, ok := checkedMulInt(spec.KVHeads, spec.HeadDim)
		if !ok || spec.KVHeads <= 0 || spec.HeadDim <= 0 || spec.Window == 0 {
			return 0, fmt.Errorf("model: invalid attention row geometry at layer %d", l)
		}
		capacity := maxTokens
		if spec.Window > 0 && spec.Window < capacity {
			capacity = spec.Window
		}
		cells, ok := checkedMulInt(capacity, stride)
		if !ok || int64(cells) > math.MaxInt64/8 || bytes > math.MaxInt64-int64(cells)*8 {
			return 0, fmt.Errorf("model: attention row reservation overflows at layer %d", l)
		}
		bytes += int64(cells) * 8 // independent F32 K and V planes
	}
	if bytes > int64(int(^uint(0)>>1)) {
		return 0, fmt.Errorf("model: attention row reservation exceeds addressable payload")
	}
	if bytes > budget {
		return 0, fmt.Errorf("model: attention row reservation needs %d bytes, budget %d", bytes, budget)
	}
	return bytes, nil
}

func newAttentionStateRows(specs []attentionRowSpec, maxTokens int, budget int64) (*attentionStateRows, error) {
	bytes, err := attentionStateRowBytes(specs, maxTokens, budget)
	if err != nil {
		return nil, err
	}
	out := &attentionStateRows{layers: make([]attentionLayerRows, len(specs)),
		history: make([]int, 0, maxTokens), maxTokens: maxTokens, payloadBytes: bytes}
	for l, spec := range specs {
		capacity := maxTokens
		if spec.Window > 0 && spec.Window < capacity {
			capacity = spec.Window
		}
		stride := spec.KVHeads * spec.HeadDim // checked above, before any allocation
		out.layers[l] = attentionLayerRows{spec: spec, stride: stride, capacity: capacity,
			k: make([]float32, capacity*stride), v: make([]float32, capacity*stride)}
	}
	return out, nil
}

// row borrows an already committed absolute position for synchronous attention.
// The current token is deliberately NOT stored here until every layer succeeds.
func (s *attentionStateRows) row(layer, position int) ([]float32, []float32) {
	r := &s.layers[layer]
	start := len(s.history) - r.capacity
	if start < 0 {
		start = 0
	}
	if position < start || position >= len(s.history) {
		panic("model: attention row position is not retained")
	}
	offset := (position % r.capacity) * r.stride
	return r.k[offset : offset+r.stride], r.v[offset : offset+r.stride]
}

type attentionPendingRow struct{ k, v []float32 }

// publish validates the complete token before any ring overwrite. Callers must
// finish attention against old rows plus current projections FIRST. Thus wrap
// cannot destroy an old row while a later layer still needs it.
func (s *attentionStateRows) publish(id int, pending []attentionPendingRow) {
	if len(s.history) >= s.maxTokens || len(pending) != len(s.layers) {
		panic("model: attention row publication exceeds reservation")
	}
	for l, row := range pending {
		if len(row.k) != s.layers[l].stride || len(row.v) != s.layers[l].stride {
			panic("model: attention row publication geometry mismatch")
		}
	}
	position := len(s.history)
	for l, row := range pending {
		r := &s.layers[l]
		offset := (position % r.capacity) * r.stride
		copy(r.k[offset:offset+r.stride], row.k)
		copy(r.v[offset:offset+r.stride], row.v)
	}
	s.history = append(s.history, id) // exact reserved capacity; never grows
}

func (s *attentionStateRows) clone() *attentionStateRows {
	if s == nil {
		return nil
	}
	out := &attentionStateRows{layers: make([]attentionLayerRows, len(s.layers)),
		history: make([]int, len(s.history), s.maxTokens), maxTokens: s.maxTokens, payloadBytes: s.payloadBytes}
	copy(out.history, s.history)
	for l, r := range s.layers {
		out.layers[l] = r
		out.layers[l].k = make([]float32, len(r.k))
		out.layers[l].v = make([]float32, len(r.v))
		copy(out.layers[l].k, r.k)
		copy(out.layers[l].v, r.v)
	}
	return out
}

func (s *attentionStateRows) close() {
	if s != nil {
		s.layers, s.history = nil, nil
		s.payloadBytes = 0
	}
}

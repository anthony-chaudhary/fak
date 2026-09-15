package v41

// Copyright (c) 2023 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Adapted from deepseek-ai/DeepSeek-V4.1-Flash inference/model.py:656-763,
// 1166-1180, 1242-1272 at dba1be0a40aa45a94ad051997016db3960a90277 (MIT).
//
// The reference keeps one process-global SharedAttentionRuntime plus per-layer
// window/compress/index buffers, and each Compressor/Indexer pools its own
// projected input before Attention reads the cache. This file provides the
// session-owned bounded equivalent of that shared state: projected window KV in
// a 128-row circular buffer, caller-supplied completed compressed rows and
// index-key cache per source layer, and the source-published candidate/top-k
// selection slots. Inputs are already projected; this file runs no projection,
// attention contraction, inverse RoPE, tensor loading, or Metal kernel. Native
// execution remains gated by ErrV41NativeUnsupported elsewhere.

import (
	"fmt"

	model "github.com/anthony-chaudhary/fak/internal/model"
)

// v41WindowSize is the pinned sliding-window width from the reference ModelArgs
// default (inference/model.py:79). Window KV lives in a ring of this many rows.
const v41WindowSize = 128

// V41AttentionStateRef identifies one layer in the sequential stack. LayerID
// is zero-based as in the pinned source. Ratio is that layer's compress ratio;
// ratio 0 disables compressed state for the layer. IsKVSource and IsIndexSource
// mark the layers that publish the shared compressed KV and index keys.
type V41AttentionStateRef struct {
	LayerID       int
	Ratio         int
	IsKVSource    bool
	IsIndexSource bool
}

// V41AttentionStateUpdate carries one layer's projection-free publication for a
// single Prefill or Step call. Latent, when non-nil, is a compressed row the
// caller produced for the group that just completed; IndexKey is the matching
// index-key row. A caller that did not complete a group supplies neither.
type V41AttentionStateUpdate struct {
	Ref      V41AttentionStateRef
	Latent   []float32
	IndexKey []float32
}

// V41AttentionState is session-owned state shared by the sequential attention
// stack. One instance replaces the reference's process-global shared runtime so
// concurrent sessions cannot alias each other. A source layer publishes before
// its consumers read within the same Prefill or Step call; readers reuse the
// latest publication.
//
// The state is bounded: window KV is a fixed 128-row ring and compressed rows
// are appended only when the caller supplies a completed group.
type V41AttentionState struct {
	windowSize int
	headDim    int
	ratioCap   int

	nextWindowPos   int
	nextCompressRow int

	window    [][]float32
	kvRows    map[int][][]float32
	indexKeys map[int][][]float32

	candidates     []bool
	candidateRatio int
	candidatesSet  bool
	topk           [][]int32
	topkRatio      int
	topkSet        bool
}

// NewV41AttentionState validates the pinned geometry and returns empty state.
// headDim is the KV latent width; ratioCap bounds any layer's compress ratio so
// malformed updates are rejected before mutation.
func NewV41AttentionState(headDim, ratioCap int) (*V41AttentionState, error) {
	if headDim <= 0 {
		return nil, fmt.Errorf("model: V41 attention state head dim %d must be positive", headDim)
	}
	if ratioCap < 1 {
		return nil, fmt.Errorf("model: V41 attention state ratio cap %d must be positive", ratioCap)
	}
	s := &V41AttentionState{
		windowSize: v41WindowSize,
		headDim:    headDim,
		ratioCap:   ratioCap,
		window:     make([][]float32, v41WindowSize),
		kvRows:     make(map[int][][]float32),
		indexKeys:  make(map[int][][]float32),
	}
	for i := range s.window {
		s.window[i] = make([]float32, headDim)
	}
	return s, nil
}

// Prefill seeds the window ring from a projected KV chunk and publishes the
// updates. It is atomic: any validation failure leaves the receiver unchanged.
// Projected rows are the full prefill chunk, oldest position first.
func (s *V41AttentionState) Prefill(projectedKV [][]float32, updates []V41AttentionStateUpdate) error {
	if s == nil {
		return fmt.Errorf("model: nil V41 attention state")
	}
	if len(projectedKV) == 0 {
		return fmt.Errorf("model: V41 attention state prefill requires at least one position")
	}
	if s.nextWindowPos != 0 {
		return fmt.Errorf("model: V41 attention state prefill requires an empty state, window position %d", s.nextWindowPos)
	}
	if len(projectedKV) > s.windowSize {
		return fmt.Errorf("model: V41 attention state prefill chunk %d exceeds window %d", len(projectedKV), s.windowSize)
	}
	if err := s.validateKV(projectedKV); err != nil {
		return err
	}
	if err := s.validateUpdates(updates); err != nil {
		return err
	}

	for i, row := range projectedKV {
		copy(s.window[i], row)
	}
	s.nextWindowPos = len(projectedKV)
	s.nextCompressRow = len(projectedKV)
	s.publish(updates)
	return nil
}

// Step advances the ring by one projected decode position and publishes the
// updates. It is atomic: a validation failure leaves the receiver unchanged.
func (s *V41AttentionState) Step(projectedKV []float32, updates []V41AttentionStateUpdate) error {
	if s == nil {
		return fmt.Errorf("model: nil V41 attention state")
	}
	if s.nextWindowPos == 0 {
		return fmt.Errorf("model: V41 attention state step requires a prefilled state")
	}
	if len(projectedKV) != s.headDim {
		return fmt.Errorf("model: V41 attention state step KV width %d, want %d", len(projectedKV), s.headDim)
	}
	for i, value := range projectedKV {
		if !model.Finite32(value) {
			return fmt.Errorf("model: V41 attention state KV[%d] is non-finite", i)
		}
	}
	if err := s.validateUpdates(updates); err != nil {
		return err
	}

	copy(s.window[s.nextWindowPos%s.windowSize], projectedKV)
	s.nextWindowPos++
	s.nextCompressRow++
	s.publish(updates)
	return nil
}

// PublishCandidates records the latest source-published candidate mask, as the
// reference candidate_source_layer does. The mask is copied.
func (s *V41AttentionState) PublishCandidates(ratio int, mask []bool) error {
	if s == nil {
		return fmt.Errorf("model: nil V41 attention state")
	}
	if ratio < 0 || ratio > s.ratioCap {
		return fmt.Errorf("model: V41 attention state candidate ratio %d outside [0,%d]", ratio, s.ratioCap)
	}
	s.candidates = append([]bool(nil), mask...)
	s.candidateRatio = ratio
	s.candidatesSet = true
	return nil
}

// PublishTopK records the latest source-published top-k selection, as the
// reference index_source does. Each row is copied.
func (s *V41AttentionState) PublishTopK(ratio int, rows [][]int32) error {
	if s == nil {
		return fmt.Errorf("model: nil V41 attention state")
	}
	if ratio < 0 || ratio > s.ratioCap {
		return fmt.Errorf("model: V41 attention state top-k ratio %d outside [0,%d]", ratio, s.ratioCap)
	}
	out := make([][]int32, len(rows))
	for i, row := range rows {
		out[i] = append([]int32(nil), row...)
	}
	s.topk = out
	s.topkRatio = ratio
	s.topkSet = true
	return nil
}

// Reset clears every row and publication, returning the state to its freshly
// constructed form with the same geometry.
func (s *V41AttentionState) Reset() {
	if s == nil {
		return
	}
	for i := range s.window {
		for j := range s.window[i] {
			s.window[i][j] = 0
		}
	}
	s.nextWindowPos = 0
	s.nextCompressRow = 0
	s.kvRows = make(map[int][][]float32)
	s.indexKeys = make(map[int][][]float32)
	s.candidates = nil
	s.candidatesSet = false
	s.candidateRatio = 0
	s.topk = nil
	s.topkSet = false
	s.topkRatio = 0
}

// WindowKV returns a copy of the current logical window, oldest slot first.
func (s *V41AttentionState) WindowKV() [][]float32 {
	if s == nil {
		return nil
	}
	out := make([][]float32, s.windowSize)
	for i := 0; i < s.windowSize; i++ {
		slot := (s.nextWindowPos + i) % s.windowSize
		out[i] = append([]float32(nil), s.window[slot]...)
	}
	return out
}

// KVSourceRows returns a copy of the completed compressed rows published by the
// given source layer, in group order.
func (s *V41AttentionState) KVSourceRows(sourceLayerID int) ([][]float32, bool) {
	if s == nil {
		return nil, false
	}
	rows, ok := s.kvRows[sourceLayerID]
	if !ok {
		return nil, false
	}
	out := make([][]float32, len(rows))
	for i, row := range rows {
		out[i] = append([]float32(nil), row...)
	}
	return out, true
}

// IndexKeys returns a copy of the index-key cache published by the given source
// layer, in group order.
func (s *V41AttentionState) IndexKeys(sourceLayerID int) ([][]float32, bool) {
	if s == nil {
		return nil, false
	}
	keys, ok := s.indexKeys[sourceLayerID]
	if !ok {
		return nil, false
	}
	out := make([][]float32, len(keys))
	for i, key := range keys {
		out[i] = append([]float32(nil), key...)
	}
	return out, true
}

// Candidates returns a copy of the latest source-published candidate mask, and
// the ratio of the layer that published it.
func (s *V41AttentionState) Candidates() ([]bool, int, bool) {
	if s == nil || !s.candidatesSet {
		return nil, 0, false
	}
	return append([]bool(nil), s.candidates...), s.candidateRatio, true
}

// TopK returns a copy of the latest source-published top-k selection, and the
// ratio of the layer that published it.
func (s *V41AttentionState) TopK() ([][]int32, int, bool) {
	if s == nil || !s.topkSet {
		return nil, 0, false
	}
	out := make([][]int32, len(s.topk))
	for i, row := range s.topk {
		out[i] = append([]int32(nil), row...)
	}
	return out, s.topkRatio, true
}

func (s *V41AttentionState) validateKV(projectedKV [][]float32) error {
	for pos, row := range projectedKV {
		if len(row) != s.headDim {
			return fmt.Errorf("model: V41 attention state KV[%d] width %d, want %d", pos, len(row), s.headDim)
		}
		for i, value := range row {
			if !model.Finite32(value) {
				return fmt.Errorf("model: V41 attention state KV[%d][%d] is non-finite", pos, i)
			}
		}
	}
	return nil
}

// validateUpdates enforces non-decreasing, zero-based layer IDs, a valid ratio,
// and a well-formed latent/index pair before mutation. A prefill chunk may
// complete several groups for one source layer, so a layer may appear more than
// once as long as its publications stay contiguous in stack order. A compressed
// row requires a positive ratio; a ratio-0 layer may not publish one.
func (s *V41AttentionState) validateUpdates(updates []V41AttentionStateUpdate) error {
	lastLayer := -1
	for i, up := range updates {
		ref := up.Ref
		if ref.LayerID < 0 {
			return fmt.Errorf("model: V41 attention state update[%d] layer %d is negative", i, ref.LayerID)
		}
		if ref.LayerID < lastLayer {
			return fmt.Errorf("model: V41 attention state update[%d] layer %d is out of stack order after %d", i, ref.LayerID, lastLayer)
		}
		if ref.Ratio < 0 || ref.Ratio > s.ratioCap {
			return fmt.Errorf("model: V41 attention state update[%d] ratio %d outside [0,%d]", i, ref.Ratio, s.ratioCap)
		}
		if up.Latent != nil {
			if !ref.IsKVSource {
				return fmt.Errorf("model: V41 attention state update[%d] layer %d is not a KV source", i, ref.LayerID)
			}
			if ref.Ratio <= 0 {
				return fmt.Errorf("model: V41 attention state update[%d] layer %d has ratio 0 but supplied a latent", i, ref.LayerID)
			}
			if len(up.Latent) != s.headDim {
				return fmt.Errorf("model: V41 attention state update[%d] latent width %d, want %d", i, len(up.Latent), s.headDim)
			}
			if err := s.finiteRow(up.Latent, "latent"); err != nil {
				return fmt.Errorf("model: V41 attention state update[%d]: %w", i, err)
			}
		}
		if up.IndexKey != nil {
			if !ref.IsIndexSource {
				return fmt.Errorf("model: V41 attention state update[%d] layer %d is not an index source", i, ref.LayerID)
			}
			if ref.Ratio <= 0 {
				return fmt.Errorf("model: V41 attention state update[%d] layer %d has ratio 0 but supplied an index key", i, ref.LayerID)
			}
			if len(up.IndexKey) != s.headDim {
				return fmt.Errorf("model: V41 attention state update[%d] index key width %d, want %d", i, len(up.IndexKey), s.headDim)
			}
			if err := s.finiteRow(up.IndexKey, "index key"); err != nil {
				return fmt.Errorf("model: V41 attention state update[%d]: %w", i, err)
			}
		}
		lastLayer = ref.LayerID
	}
	return nil
}

func (s *V41AttentionState) finiteRow(row []float32, what string) error {
	for i, value := range row {
		if !model.Finite32(value) {
			return fmt.Errorf("%s[%d] is non-finite", what, i)
		}
	}
	return nil
}

// publish appends supplied rows in stack order. It is called only after
// validateUpdates, so every append is well-formed.
func (s *V41AttentionState) publish(updates []V41AttentionStateUpdate) {
	for _, up := range updates {
		if up.Latent != nil {
			s.kvRows[up.Ref.LayerID] = append(s.kvRows[up.Ref.LayerID], append([]float32(nil), up.Latent...))
		}
		if up.IndexKey != nil {
			s.indexKeys[up.Ref.LayerID] = append(s.indexKeys[up.Ref.LayerID], append([]float32(nil), up.IndexKey...))
		}
	}
}

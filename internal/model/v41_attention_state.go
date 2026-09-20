package model

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

import "fmt"

// v41WindowSize is the pinned sliding-window width from the reference ModelArgs
// default (inference/model.py:79). Window KV lives in a ring of this many rows.
const v41WindowSize = 128

// v41PublishedAttentionGeometry is the published DeepSeek-V4.1-Flash decoder
// attention envelope, transcribed from the pinned testdata config
// (deepseek_v41_flash_config.json @ dba1be0a; inference/model.py ModelArgs). It
// is the whole-envelope value admitDeepSeekV41Published requires
// DeepSeekV41Config.Attention to hold on parse. The reduced forward fixture
// narrows only the flat Config and deliberately retains this envelope whole, so
// the forward's drift guard must distinguish exact retention from lone-axis drift
// rather than comparing Attention against the narrowed flat axes.
var v41PublishedAttentionGeometry = DeepSeekV41AttentionGeometry{
	NumLayers:           40,
	HiddenSize:          5120,
	NumHeads:            64,
	NumKVHeads:          1,
	HeadDim:             512,
	QKRopeHeadDim:       64,
	QLoraRank:           1280,
	OLoraRank:           1024,
	OGroups:             8,
	NumExperts:          384,
	NSharedExperts:      1,
	NumExpertsPerTok:    6,
	MoEIntermediateSize: 2304,
}

// v41AttentionGeometryForwardAdmitted fails closed when a V4.1 config's retained
// Attention envelope is neither empty nor the published one. Attention is a
// metadata envelope a later native leaf trusts instead of re-reading the flat
// Config, so an envelope that is populated but does not match the published axes
// is lone-axis drift: the assembly would execute the flat geometry while a reader
// trusts a disagreeing envelope. That must refuse with the typed
// ErrV41ForwardStage rather than emit logits.
//
// A zero envelope stays admitted: it is the legitimate "no typed metadata
// retained" state for hand-built fixtures, and the forward already reads the flat
// axes directly. An envelope equal to the published one also stays admitted: the
// reduced fixture narrows the flat geometry but retains the published envelope
// whole, so exact-envelope retention is not drift.
func v41AttentionGeometryForwardAdmitted(cfg Config) error {
	m := cfg.DeepSeekV41
	if m == nil {
		return nil
	}
	if m.Attention == (DeepSeekV41AttentionGeometry{}) {
		return nil
	}
	if m.Attention != v41PublishedAttentionGeometry {
		return v41StageErr(v41StageAttention, -1,
			fmt.Errorf("%w: retained Attention envelope %+v does not match the published V4.1 decoder attention axes", ErrV41ForwardStage, m.Attention))
	}
	return nil
}

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

// v41AttentionPublicationKey binds a completed compressed publication to its
// source identity and absolute half-open compressed-row range.
type v41AttentionPublicationKey struct {
	sourceLayer int
	start       int
	end         int
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

	nextWindowPos      int
	nextCompressRow    int
	retainedWindowRows int
	partialKV          [][]float32
	retainedCopies     int

	window            [][]float32
	kvPublications    map[v41AttentionPublicationKey][]float32
	indexPublications map[v41AttentionPublicationKey][]float32
	kvPublishedEnd    map[int]int
	indexPublishedEnd map[int]int

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
		windowSize:        v41WindowSize,
		headDim:           headDim,
		ratioCap:          ratioCap,
		window:            make([][]float32, v41WindowSize),
		kvPublications:    make(map[v41AttentionPublicationKey][]float32),
		indexPublications: make(map[v41AttentionPublicationKey][]float32),
		kvPublishedEnd:    make(map[int]int),
		indexPublishedEnd: make(map[int]int),
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

// seedTemporal replaces the receiver with the bounded temporal subset needed by
// a later incremental layer step. Completed compressed rows remain in the
// per-forward publication registry and are not copied into session state.
func (s *V41AttentionState) seedTemporal(projectedKV [][]float32, ratio, configuredWindow int) error {
	if s == nil {
		return fmt.Errorf("model: nil V41 attention state")
	}
	if len(projectedKV) == 0 {
		return fmt.Errorf("model: V41 temporal seed requires at least one position")
	}
	if ratio < 0 || ratio > s.ratioCap {
		return fmt.Errorf("model: V41 temporal seed ratio %d outside [0,%d]", ratio, s.ratioCap)
	}
	if configuredWindow == 0 || configuredWindow < -1 {
		return fmt.Errorf("model: V41 temporal seed window %d must be -1 or positive", configuredWindow)
	}
	if err := s.validateKV(projectedKV); err != nil {
		return err
	}

	retain := min(len(projectedKV), s.windowSize)
	if configuredWindow > 0 {
		retain = min(retain, configuredWindow)
	}
	start := len(projectedKV) - retain
	window := make([][]float32, s.windowSize)
	for i := range window {
		window[i] = make([]float32, s.headDim)
	}
	for pos := start; pos < len(projectedKV); pos++ {
		copy(window[pos%s.windowSize], projectedKV[pos])
	}
	var partialKV [][]float32
	retainedCopies := retain

	if ratio > 1 {
		partial := len(projectedKV) % ratio
		if partial > 0 {
			partialKV = make([][]float32, partial)
			for i, row := range projectedKV[len(projectedKV)-partial:] {
				partialKV[i] = append([]float32(nil), row...)
			}
			retainedCopies += partial
		}
	}
	// Commit only after every validation and allocation above succeeds.
	s.window = window
	s.nextWindowPos = len(projectedKV)
	s.nextCompressRow = len(projectedKV)
	s.retainedWindowRows = retain
	s.partialKV = partialKV
	s.retainedCopies = retainedCopies
	s.kvPublications = make(map[v41AttentionPublicationKey][]float32)
	s.indexPublications = make(map[v41AttentionPublicationKey][]float32)
	s.kvPublishedEnd = make(map[int]int)
	s.indexPublishedEnd = make(map[int]int)
	s.candidates = nil
	s.candidatesSet = false
	s.topk = nil
	s.topkSet = false
	return nil
}

// retainedWindowKV returns the populated logical window tail, oldest row first.
func (s *V41AttentionState) retainedWindowKV() [][]float32 {
	if s == nil || s.retainedWindowRows == 0 {
		return nil
	}
	out := make([][]float32, s.retainedWindowRows)
	start := s.nextWindowPos - s.retainedWindowRows
	for i := range out {
		out[i] = append([]float32(nil), s.window[(start+i)%s.windowSize]...)
	}
	return out
}

// partialGroupKV returns a copy of the incomplete compressor group.
func (s *V41AttentionState) partialGroupKV() [][]float32 {
	if s == nil {
		return nil
	}
	out := make([][]float32, len(s.partialKV))
	for i, row := range s.partialKV {
		out[i] = append([]float32(nil), row...)
	}
	return out
}

// publishUpdates atomically adds completed source publications to the
// per-forward registry without conflating them with temporal layer state.
func (s *V41AttentionState) publishUpdates(updates []V41AttentionStateUpdate) error {
	if s == nil {
		return fmt.Errorf("model: nil V41 attention state")
	}
	if err := s.validateUpdates(updates); err != nil {
		return err
	}
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
		if !finite32(value) {
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
	s.retainedWindowRows = 0
	s.partialKV = nil
	s.retainedCopies = 0
	s.kvPublications = make(map[v41AttentionPublicationKey][]float32)
	s.indexPublications = make(map[v41AttentionPublicationKey][]float32)
	s.kvPublishedEnd = make(map[int]int)
	s.indexPublishedEnd = make(map[int]int)
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
	end := s.kvPublishedEnd[sourceLayerID]
	if end == 0 {
		return nil, false
	}
	out := make([][]float32, 0, end)
	for start := 0; start < end; start++ {
		key := v41AttentionPublicationKey{sourceLayer: sourceLayerID, start: start, end: start + 1}
		row, ok := s.kvPublications[key]
		if !ok {
			return nil, false
		}
		out = append(out, append([]float32(nil), row...))
	}
	return out, true
}

// IndexKeys returns a copy of the index-key cache published by the given source
// layer, in group order.
func (s *V41AttentionState) IndexKeys(sourceLayerID int) ([][]float32, bool) {
	if s == nil {
		return nil, false
	}
	end := s.indexPublishedEnd[sourceLayerID]
	if end == 0 {
		return nil, false
	}
	out := make([][]float32, 0, end)
	for start := 0; start < end; start++ {
		key := v41AttentionPublicationKey{sourceLayer: sourceLayerID, start: start, end: start + 1}
		row, ok := s.indexPublications[key]
		if !ok {
			return nil, false
		}
		out = append(out, append([]float32(nil), row...))
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
			if !finite32(value) {
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
		if !finite32(value) {
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
			start := s.kvPublishedEnd[up.Ref.LayerID]
			key := v41AttentionPublicationKey{sourceLayer: up.Ref.LayerID, start: start, end: start + 1}
			s.kvPublications[key] = append([]float32(nil), up.Latent...)
			s.kvPublishedEnd[up.Ref.LayerID] = start + 1
		}
		if up.IndexKey != nil {
			start := s.indexPublishedEnd[up.Ref.LayerID]
			key := v41AttentionPublicationKey{sourceLayer: up.Ref.LayerID, start: start, end: start + 1}
			s.indexPublications[key] = append([]float32(nil), up.IndexKey...)
			s.indexPublishedEnd[up.Ref.LayerID] = start + 1
		}
	}
}

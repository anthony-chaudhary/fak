package model

import (
	"fmt"
	"slices"
	"strconv"
)

// v41RatioOneStep carries explicitly resolved ownership. An index-query source
// is not necessarily an index-key owner: only IndexKeySourceLayer==Layer may
// project a fresh key, and that layer must also own compressed KV. Reader layers
// never call producer callbacks. Selection remains in consume, which receives
// owned proposed rows and may use v41CombinedAttention before anything commits.
// Callbacks are invocation-local and are never retained by the state/snapshot.
type v41RatioOneStep struct {
	Layer, KVSourceLayer, IndexKeySourceLayer int
	Pos, Window, Heads                        int
	Input, WindowKV, NormWeight               []float32
	Epsilon                                   float32
	ProjectKV                                 func([]float32) ([]float32, error)
	ProjectIndex                              func([]float32) ([]float32, error)
	Finalize                                  func(pos int, latent, indexKey []float32) error
	Consume                                   func(window, compressed, indexKeys [][]float32) ([]float32, error)
}

// stepRatioOne stages the producer's projection -> learned RMSNorm -> optional
// unrotated-latent index projection -> finalization, then consumes the combined
// rows. A reader only resolves the declared shared source. Successful calls
// commit this layer's window and both cursors exactly once, plus any fresh owner
// and registry publications. Returned errors and callback panics occur before
// mutation; no partial-group or gate path exists for ratio one.
//
// This is internal composition source, not public architecture admission. The
// public ratio-one and candidate-source fences remain intact. Exact FP8/FP4
// activation/cache arithmetic is still separate, unqualified work.
func (s *V41AttentionState) stepRatioOne(registry *V41AttentionState, request v41RatioOneStep) ([]float32, error) {
	fail := func(message string) ([]float32, error) {
		return nil, v41StageErr(v41StageAttention, request.Layer, fmt.Errorf("%w: ratio-one composition %s", ErrV41ForwardStage, message))
	}
	if s == nil || registry == nil || s == registry || request.Layer < 0 || request.KVSourceLayer < 0 || request.KVSourceLayer > request.Layer ||
		request.IndexKeySourceLayer < -1 || (request.IndexKeySourceLayer >= 0 && request.IndexKeySourceLayer != request.KVSourceLayer) ||
		request.Pos < 0 || request.Pos == int(^uint(0)>>1) {
		return fail("has invalid state, position or source ownership")
	}
	if s.ratioCap < 1 || registry.ratioCap < 1 || s.headDim <= 0 || s.indexWidth() <= 0 || s.headDim != registry.headDim || s.indexWidth() != registry.indexWidth() ||
		s.windowSize <= 0 || len(s.window) != s.windowSize || request.Window <= 0 || request.Window > s.windowSize ||
		request.Pos != s.nextWindowPos || request.Pos != s.nextCompressRow ||
		s.retainedWindowRows < 0 || s.retainedWindowRows > min(request.Pos, s.windowSize) ||
		s.retainedCopies != s.retainedWindowRows ||
		len(s.partialInputs) != 0 || len(s.partialPositions) != 0 || len(s.partialKV) != 0 || request.Consume == nil {
		return fail("has inconsistent geometry, history or callbacks")
	}
	outputWidth, outputOK := checkedMulInt(request.Heads, s.headDim)
	_, outputBytesOK := checkedMulInt(outputWidth, 4)
	if request.Heads <= 0 || !outputOK || !outputBytesOK {
		return fail("has invalid output geometry")
	}
	if s.kvPublications == nil || s.indexPublications == nil || s.kvPublishedEnd == nil || s.indexPublishedEnd == nil ||
		registry.kvPublications == nil || registry.indexPublications == nil || registry.kvPublishedEnd == nil || registry.indexPublishedEnd == nil {
		return fail("has uninitialized publication stores")
	}
	if len(request.WindowKV) != s.headDim || len(s.window[request.Pos%s.windowSize]) != s.headDim {
		return fail("has invalid window row width")
	}
	if err := finiteRow32(request.WindowKV, "ratio-one window row"); err != nil {
		return nil, v41StageErr(v41StageAttention, request.Layer, err)
	}
	currentWindow := append([]float32(nil), request.WindowKV...)
	start := max(0, request.Pos-request.Window+1)
	if start < request.Pos-s.retainedWindowRows {
		return fail("configured window is not fully retained")
	}
	window := make([][]float32, 0, request.Pos-start+1)
	for pos := start; pos < request.Pos; pos++ {
		row := s.window[pos%s.windowSize]
		if len(row) != s.headDim {
			return fail("retained window row has invalid width")
		}
		window = append(window, append([]float32(nil), row...))
	}
	window = append(window, append([]float32(nil), currentWindow...))

	ownsKV := request.KVSourceLayer == request.Layer
	ownsKeys := request.IndexKeySourceLayer == request.Layer
	if ownsKeys && !ownsKV {
		return fail("index-key ownership requires KV ownership")
	}
	expectedRows := request.Pos + 1
	if ownsKV {
		expectedRows--
	}
	// Source readers allocate a row-descriptor slice before copying payloads.
	// Refuse corrupt counters and overflowing byte geometry before calling them.
	historyFits := func(rows, width int) bool {
		count, countOK := checkedMulInt(rows, width)
		_, bytesOK := checkedMulInt(count, 4)
		_, descriptorsOK := checkedMulInt(rows, 3*(strconv.IntSize/8))
		return countOK && bytesOK && descriptorsOK
	}
	if registry.kvPublishedEnd[request.KVSourceLayer] != expectedRows ||
		(ownsKV && s.kvPublishedEnd[request.Layer] != expectedRows) || !historyFits(request.Pos+1, s.headDim) {
		return fail("shared KV source cursor or byte count is invalid")
	}
	compressed, present := registry.KVSourceRows(request.KVSourceLayer)
	if len(compressed) != expectedRows || (expectedRows > 0 && !present) {
		return fail("shared source is missing or not current")
	}
	if ownsKV {
		owned, ok := s.KVSourceRows(request.Layer)
		if len(owned) != len(compressed) || (expectedRows > 0 && !ok) {
			return fail("owner and registry source histories disagree")
		}
		for i := range owned {
			if !slices.Equal(owned[i], compressed[i]) {
				return fail("owner and registry source rows disagree")
			}
		}
	}
	var keys [][]float32
	if request.IndexKeySourceLayer >= 0 {
		expectedKeys := request.Pos + 1
		if ownsKeys {
			expectedKeys--
		}
		if registry.indexPublishedEnd[request.IndexKeySourceLayer] != expectedKeys ||
			(ownsKeys && s.indexPublishedEnd[request.Layer] != expectedKeys) || !historyFits(request.Pos+1, s.indexWidth()) {
			return fail("shared index-key source cursor or byte count is invalid")
		}
		keys, present = registry.IndexKeys(request.IndexKeySourceLayer)
		if len(keys) != expectedKeys || (expectedKeys > 0 && !present) {
			return fail("shared index-key source is missing or not current")
		}
		if ownsKeys {
			owned, ok := s.IndexKeys(request.Layer)
			if len(owned) != len(keys) || (expectedKeys > 0 && !ok) {
				return fail("owner and registry index histories disagree")
			}
			for i := range owned {
				if !slices.Equal(owned[i], keys[i]) {
					return fail("owner and registry index rows disagree")
				}
			}
		}
	}
	for _, history := range []struct {
		rows  [][]float32
		width int
	}{{window, s.headDim}, {compressed, s.headDim}, {keys, s.indexWidth()}} {
		for _, row := range history.rows {
			if len(row) != history.width {
				return fail("retained source row has invalid width")
			}
			if err := finiteRow32(row, "ratio-one retained row"); err != nil {
				return nil, v41StageErr(v41StageAttention, request.Layer, err)
			}
		}
	}
	var updates []V41AttentionStateUpdate
	if ownsKV {
		if request.ProjectKV == nil || request.Finalize == nil || (ownsKeys && request.ProjectIndex == nil) {
			return fail("producer callbacks or publication cursors disagree")
		}
		if len(request.Input) == 0 || len(request.NormWeight) != s.headDim || !finite32(request.Epsilon) || request.Epsilon <= 0 {
			return fail("producer input or normalization geometry is invalid")
		}
		if err := finiteRow32(request.Input, "ratio-one producer input"); err != nil {
			return nil, v41StageErr(v41StageCompress, request.Layer, err)
		}
		projected, err := request.ProjectKV(append([]float32(nil), request.Input...))
		if err != nil {
			return nil, err
		}
		pool, err := NewV41CompressorPool(1, s.headDim)
		if err != nil {
			return nil, err
		}
		latent, emitted, err := pool.PushNormalized(0, projected, nil, request.NormWeight, request.Epsilon)
		if err != nil || !emitted {
			if err == nil {
				err = fmt.Errorf("%w: ratio-one producer emitted no latent", ErrV41ForwardStage)
			}
			return nil, v41StageErr(v41StageCompress, request.Layer, err)
		}
		if len(latent) != s.headDim {
			return fail("normalized latent has invalid width")
		}
		if err := finiteRow32(latent, "ratio-one normalized latent"); err != nil {
			return nil, v41StageErr(v41StageCompress, request.Layer, err)
		}
		var key []float32
		if ownsKeys {
			key, err = request.ProjectIndex(append([]float32(nil), latent...))
			if err != nil {
				return nil, err
			}
			if len(key) != s.indexWidth() {
				return fail("projected index key has invalid width")
			}
			if err := finiteRow32(key, "ratio-one projected index key"); err != nil {
				return nil, v41StageErr(v41StageAttention, request.Layer, err)
			}
			key = append([]float32(nil), key...)
		}
		if err := request.Finalize(request.Pos, latent, key); err != nil {
			return nil, err
		}
		updates = []V41AttentionStateUpdate{{
			// IsIndexSource denotes key publication ownership at this boundary,
			// not an index-only query/selection source later in the layer stack.
			Ref:    V41AttentionStateRef{LayerID: request.Layer, Ratio: 1, IsKVSource: true, IsIndexSource: ownsKeys},
			Latent: append([]float32(nil), latent...), IndexKey: append([]float32(nil), key...),
		}}
		if err := s.validateUpdates(updates); err != nil {
			return nil, err
		}
		if err := registry.validateUpdates(updates); err != nil {
			return nil, err
		}
		compressed = append(compressed, append([]float32(nil), latent...))
		if ownsKeys {
			keys = append(keys, append([]float32(nil), key...))
		}
	}
	// The consumer owns its copies, so even a callback that mutates them cannot
	// change staged publication bytes or a caller's original window input.
	result, err := request.Consume(window, compressed, keys)
	if err != nil {
		return nil, err
	}
	if len(result) != outputWidth {
		return fail("consumer returned an invalid output width")
	}
	if err := finiteRow32(result, "ratio-one attention output"); err != nil {
		return nil, v41StageErr(v41StageAttention, request.Layer, err)
	}
	result = append([]float32(nil), result...)

	// No error-returning operation remains. Never call State.Step here: it would
	// combine a source append with a second increment of the compressor cursor.
	s.publish(updates)
	registry.publish(updates)
	copy(s.window[request.Pos%s.windowSize], currentWindow)
	s.nextWindowPos, s.nextCompressRow = request.Pos+1, request.Pos+1
	retained := min(s.retainedWindowRows+1, s.windowSize)
	s.retainedCopies += retained - s.retainedWindowRows
	s.retainedWindowRows = retained
	return result, nil
}

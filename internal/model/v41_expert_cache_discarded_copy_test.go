package model

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
// Source-only estimate, not a measured duration: two argument shapes, eight
// AllocsPerRun measurements of 32 calls plus one warmup each (264 calls total),
// and tiny serial cache controls. Each f32 block has 16 elements (64 bytes).
// No model loading, dequantization, backend, device, or global test hook is used.
func TestV41CacheExpertTripleDiscardedCopies(t *testing.T) {
	const (
		layer      = 0
		stem       = "ffn.experts.0"
		blockLen   = 16
		blockBytes = int64(blockLen * 4)
		allocRuns  = 32
	)
	var m Model
	names := [3]string{
		layerName(layer, stem+".w1.weight"),
		layerName(layer, stem+".w3.weight"),
		layerName(layer, stem+".w2.weight"),
	}
	var weights [3][]float32
	for i := range weights {
		weights[i] = make([]float32, blockLen)
		for j := range weights[i] {
			weights[i][j] = float32(100*i + j + 1)
		}
	}

	t.Run("disabled_misses_do_not_copy", func(t *testing.T) {
		for _, shape := range []struct {
			name string
			w    [3][]float32
		}{
			{"triple", weights},
			{"down_only", [3][]float32{nil, nil, weights[2]}},
		} {
			t.Run(shape.name, func(t *testing.T) {
				// The same production helper and keys provide the allocation
				// control. Hits cannot copy f32 blocks. Their existing LRU
				// queue reorders in place, so it does not grow in this loop.
				// Do not assert zero allocations: tensor-name construction
				// may allocate independently of the unwanted f32 copies.
				hits := &v41ProjScratch{expertLayerCacheBytes: 3 * blockBytes}
				for i, w := range shape.w {
					if len(w) != 0 {
						hits.v41LayerCachePut(names[i], w)
					}
				}
				hitAllocs := testing.AllocsPerRun(allocRuns, func() {
					m.v41CacheExpertTriple(layer, stem, hits, shape.w[0], shape.w[1], shape.w[2])
				})
				for _, disabled := range []struct {
					name    string
					scratch *v41ProjScratch
				}{
					{"nil_scratch", nil},
					{"zero_budget", &v41ProjScratch{}},
					{"negative_budget", &v41ProjScratch{expertLayerCacheBytes: -1}},
				} {
					t.Run(disabled.name, func(t *testing.T) {
						got := testing.AllocsPerRun(allocRuns, func() {
							m.v41CacheExpertTriple(layer, stem, disabled.scratch, shape.w[0], shape.w[1], shape.w[2])
						})
						if got > hitAllocs {
							t.Fatalf("discarded f32 copy allocations: disabled miss=%g, resident-hit control=%g; want no excess allocations", got, hitAllocs)
						}
						if s := disabled.scratch; s != nil && (len(s.layerExperts) != 0 || len(s.layerExpertFIFO) != 0 || s.layerExpertBytes != 0) {
							t.Fatalf("disabled miss retained state: keys=%d queue=%v bytes=%d", len(s.layerExperts), s.layerExpertFIFO, s.layerExpertBytes)
						}
					})
				}
			})
		}
	})

	t.Run("positive_budget_copies_and_evicts_lru", func(t *testing.T) {
		s := &v41ProjScratch{expertLayerCacheBytes: 3 * blockBytes}
		var live [3][]float32
		for i := range live {
			live[i] = append([]float32(nil), weights[i]...)
		}
		m.v41CacheExpertTriple(layer, stem, s, live[0], live[1], live[2])
		if len(s.layerExperts) != 3 || s.layerExpertBytes != 3*blockBytes || !reflect.DeepEqual(s.layerExpertFIFO, names[:]) {
			t.Fatalf("initial cache: keys=%d queue=%v bytes=%d", len(s.layerExperts), s.layerExpertFIFO, s.layerExpertBytes)
		}
		for i, name := range names {
			cached := s.layerExperts[name]
			if !reflect.DeepEqual(cached, weights[i]) {
				t.Fatalf("%s: retained values=%v, want %v", name, cached, weights[i])
			}
			if &cached[0] == &live[i][0] {
				t.Fatalf("%s: cache aliases live scratch", name)
			}
			live[i][0] = -live[i][0]
			if !reflect.DeepEqual(cached, weights[i]) {
				t.Fatalf("%s: scratch reuse changed retained values", name)
			}
		}
		// An existing key keeps its retained bytes and refreshes recency;
		// it must not be replaced by the caller's now-mutated scratch.
		m.v41CacheExpertTriple(layer, stem, s, live[0], nil, nil)
		if !reflect.DeepEqual(s.layerExperts[names[0]], weights[0]) || !reflect.DeepEqual(s.layerExpertFIFO, []string{names[1], names[2], names[0]}) || s.layerExpertBytes != 3*blockBytes {
			t.Fatalf("hit changed retention or recency: queue=%v bytes=%d", s.layerExpertFIFO, s.layerExpertBytes)
		}
		otherStem := "ffn.experts.1"
		otherName := layerName(layer, otherStem+".w2.weight")
		m.v41CacheExpertTriple(layer, otherStem, s, nil, nil, live[2])
		if _, present := s.layerExperts[names[1]]; present {
			t.Fatal("insertion failed to evict the least-recently-used w3")
		}
		if len(s.layerExperts) != 3 || s.layerExpertBytes != 3*blockBytes || !reflect.DeepEqual(s.layerExpertFIFO, []string{names[2], names[0], otherName}) {
			t.Fatalf("eviction cache: keys=%d queue=%v bytes=%d", len(s.layerExperts), s.layerExpertFIFO, s.layerExpertBytes)
		}
		if !reflect.DeepEqual(s.layerExperts[otherName], live[2]) || &s.layerExperts[otherName][0] == &live[2][0] {
			t.Fatal("replacement projection was not independently copied")
		}
	})

	t.Run("empty_inputs_and_positive_oversize_preserve_cache", func(t *testing.T) {
		s := &v41ProjScratch{expertLayerCacheBytes: blockBytes - 1}
		coldName := layerName(layer, "ffn.experts.1.w1.weight")
		cold := weights[0][:blockLen/2]
		s.v41LayerCachePut(coldName, cold)
		checkCold := func() {
			t.Helper()
			if len(s.layerExperts) != 1 || s.layerExpertBytes != blockBytes/2 || !reflect.DeepEqual(s.layerExpertFIFO, []string{coldName}) || !reflect.DeepEqual(s.layerExperts[coldName], cold) {
				t.Fatalf("empty or oversized input changed cache: keys=%d queue=%v bytes=%d", len(s.layerExperts), s.layerExpertFIFO, s.layerExpertBytes)
			}
		}
		// Empty includes both nil and non-nil zero-length slices; neither
		// requires a live scratch pointer nor changes a populated cache.
		m.v41CacheExpertTriple(layer, stem, nil, nil, []float32{}, nil)
		m.v41CacheExpertTriple(layer, stem, s, nil, []float32{}, nil)
		checkCold()
		// A positive 63-byte budget cannot retain a 64-byte projection.
		// Each refusal must leave the existing 32-byte cold block intact.
		m.v41CacheExpertTriple(layer, stem, s, weights[0], weights[1], weights[2])
		checkCold()
	})

	t.Run("disabled_budget_miss_still_touches_later_hit", func(t *testing.T) {
		for _, disabled := range []struct {
			name   string
			budget int64
		}{
			{"zero", 0},
			{"negative", -1},
		} {
			t.Run(disabled.name, func(t *testing.T) {
				s := &v41ProjScratch{expertLayerCacheBytes: 2 * blockBytes}
				coldName := layerName(layer, "ffn.experts.1.w2.weight")
				s.v41LayerCachePut(names[1], weights[1])
				s.v41LayerCachePut(coldName, weights[2])
				s.expertLayerCacheBytes = disabled.budget
				// w1 misses before w3 hits; w2 also misses. An early return before
				// Get, or on the first disabled miss, would silently lose the w3
				// recency refresh. Existing blocks remain readable at nonpositive budgets.
				m.v41CacheExpertTriple(layer, stem, s, weights[0], weights[1], weights[2])
				if len(s.layerExperts) != 2 || s.layerExpertBytes != 2*blockBytes || !reflect.DeepEqual(s.layerExpertFIFO, []string{coldName, names[1]}) {
					t.Fatalf("disabled miss lost later-hit LRU: keys=%d queue=%v bytes=%d", len(s.layerExperts), s.layerExpertFIFO, s.layerExpertBytes)
				}
				if !reflect.DeepEqual(s.layerExperts[names[1]], weights[1]) || !reflect.DeepEqual(s.layerExperts[coldName], weights[2]) {
					t.Fatal("disabled-budget call changed retained values")
				}
				if _, present := s.layerExperts[names[0]]; present {
					t.Fatal("disabled-budget call cached the earlier w1 miss")
				}
				if _, present := s.layerExperts[names[2]]; present {
					t.Fatal("disabled-budget call cached the later w2 miss")
				}
				// Re-enabling admission makes the preserved recency observable in
				// real eviction, beyond merely inspecting the queue representation.
				s.expertLayerCacheBytes = 2 * blockBytes
				m.v41CacheExpertTriple(layer, stem, s, weights[0], nil, nil)
				if _, present := s.layerExperts[coldName]; present {
					t.Fatal("resumed insertion evicted the recent hit instead of the cold key")
				}
				if len(s.layerExperts) != 2 || s.layerExpertBytes != 2*blockBytes || !reflect.DeepEqual(s.layerExpertFIFO, []string{names[1], names[0]}) || !reflect.DeepEqual(s.layerExperts[names[1]], weights[1]) || !reflect.DeepEqual(s.layerExperts[names[0]], weights[0]) {
					t.Fatalf("resumed cache: keys=%d queue=%v bytes=%d", len(s.layerExperts), s.layerExpertFIFO, s.layerExpertBytes)
				}
			})
		}
	})
}

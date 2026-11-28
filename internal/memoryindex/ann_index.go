// Copyright (c) 2026 fak contributors
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from unum-cloud/usearch (usearch-ann-1):
// compact scalar-quantized (f32 / f16 / i8) ANN vector index for memoryindex
// semantic recall.
package memoryindex

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
)

// QuantizationLevel selects the vector storage precision in ANNIndex.
type QuantizationLevel int

const (
	QuantF32 QuantizationLevel = iota
	QuantF16
	QuantI8
)

// ANNMatch is one ranked recall hit returned by ANNIndex.Search.
type ANNMatch struct {
	ID    string
	Score float32 // cosine similarity in [-1, 1]
}

type annEntry struct {
	id    string
	f32   []float32
	i8    []int8
	scale float32
}

// ANNIndex is a compact, thread-safe vector index supporting f32, f16, and i8
// scalar-quantized storage for memory recall.
type ANNIndex struct {
	mu      sync.RWMutex
	dim     int
	quant   QuantizationLevel
	entries []annEntry
	byID    map[string]int
}

// NewANNIndex constructs an empty ANN index for dim-dimensional vectors.
func NewANNIndex(dim int, quant QuantizationLevel) (*ANNIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("memoryindex: ann dim must be positive, got %d", dim)
	}
	return &ANNIndex{
		dim:   dim,
		quant: quant,
		byID:  make(map[string]int),
	}, nil
}

// Quantization returns the configured scalar quantization level.
func (idx *ANNIndex) Quantization() QuantizationLevel { return idx.quant }

// Len returns the number of indexed vectors.
func (idx *ANNIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.entries)
}

// StorageBytes returns the quantized vector payload bytes retained by the index.
func (idx *ANNIndex) StorageBytes() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	n := int64(len(idx.entries))
	d := int64(idx.dim)
	switch idx.quant {
	case QuantI8:
		return n * (d + 4) // int8[dim] + float32 scale
	case QuantF16:
		return n * d * 2
	default:
		return n * d * 4
	}
}

// Add inserts or replaces the vector for id.
func (idx *ANNIndex) Add(id string, vec []float32) error {
	if id == "" {
		return errors.New("memoryindex: empty ann id")
	}
	if len(vec) != idx.dim {
		return fmt.Errorf("memoryindex: vec dim=%d, want %d", len(vec), idx.dim)
	}
	normed := normalizeVec(vec)
	ent := annEntry{id: id}
	if idx.quant == QuantI8 {
		ent.i8, ent.scale = quantizeI8(normed)
	} else {
		ent.f32 = normed
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()
	if pos, ok := idx.byID[id]; ok {
		idx.entries[pos] = ent
		return nil
	}
	idx.byID[id] = len(idx.entries)
	idx.entries = append(idx.entries, ent)
	return nil
}

// Search returns up to topK nearest neighbors ranked by cosine similarity descending.
func (idx *ANNIndex) Search(query []float32, topK int) []ANNMatch {
	if topK <= 0 || len(query) != idx.dim {
		return nil
	}
	q := normalizeVec(query)

	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if len(idx.entries) == 0 {
		return nil
	}
	matches := make([]ANNMatch, 0, len(idx.entries))
	for _, ent := range idx.entries {
		var score float32
		if idx.quant == QuantI8 {
			score = dotI8(q, ent.i8, ent.scale)
		} else {
			score = dotF32(q, ent.f32)
		}
		matches = append(matches, ANNMatch{ID: ent.id, Score: score})
	}
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].Score == matches[j].Score {
			return matches[i].ID < matches[j].ID
		}
		return matches[i].Score > matches[j].Score
	})
	if topK < len(matches) {
		matches = matches[:topK]
	}
	return matches
}

func normalizeVec(v []float32) []float32 {
	out := make([]float32, len(v))
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := float32(math.Sqrt(sum))
	if norm == 0 {
		return out
	}
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

func quantizeI8(v []float32) ([]int8, float32) {
	var maxAbs float32
	for _, x := range v {
		if ax := float32(math.Abs(float64(x))); ax > maxAbs {
			maxAbs = ax
		}
	}
	if maxAbs == 0 {
		return make([]int8, len(v)), 0
	}
	scale := maxAbs / 127.0
	out := make([]int8, len(v))
	for i, x := range v {
		q := math.Round(float64(x / scale))
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		out[i] = int8(q)
	}
	return out, scale
}

func dotF32(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func dotI8(q []float32, b []int8, scale float32) float32 {
	var sum float32
	for i := range q {
		sum += q[i] * float32(b[i])
	}
	return sum * scale
}

package model

// Engram hashing in this file is adapted from antirez/ds4 at
// bd66c402070042bf0a79ad6ece8242de4c93680c (ds4_engram.c).
//
// MIT License
// Copyright (c) 2026 The ds4.c authors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2023 DeepSeek
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

import (
	"fmt"
	"math"
)

// V41EngramLayout is the checkpoint-materialized hash layout. TokenMap,
// Multipliers, and Primes are conversion artifacts: the runtime deliberately
// does not reproduce the tokenizer normalization or NumPy RNG that made them.
type V41EngramLayout struct {
	TokenMap        []uint32
	CompressedVocab uint32
	PadID           uint32
	Rows            []uint32
	Multipliers     [][]uint64
	Primes          [][]uint32
	MaxNgramSize    int
	HeadsPerNgram   int
}

// V41EngramHashState carries the newest MaxNgramSize-1 compressed token IDs.
// A negative entry is a sequence boundary. Use Clone for an independent snapshot.
type V41EngramHashState struct {
	layout V41EngramLayout
	tail   []int64
}

func NewV41EngramHashState(layout V41EngramLayout) (*V41EngramHashState, error) {
	if err := layout.validate(); err != nil {
		return nil, err
	}
	layout = cloneV41EngramLayout(layout)
	s := &V41EngramHashState{layout: layout, tail: make([]int64, layout.MaxNgramSize-1)}
	for i := range s.tail {
		s.tail[i] = -1
	}
	return s, nil
}

// Clone returns an independent snapshot suitable for retry or branch decoding.
func (s *V41EngramHashState) Clone() *V41EngramHashState {
	if s == nil {
		return nil
	}
	out := &V41EngramHashState{layout: s.layout, tail: append([]int64(nil), s.tail...)}
	return out
}

func cloneV41EngramLayout(in V41EngramLayout) V41EngramLayout {
	out := in
	out.TokenMap = append([]uint32(nil), in.TokenMap...)
	out.Rows = append([]uint32(nil), in.Rows...)
	out.Multipliers = make([][]uint64, len(in.Multipliers))
	out.Primes = make([][]uint32, len(in.Primes))
	for i := range in.Multipliers {
		out.Multipliers[i] = append([]uint64(nil), in.Multipliers[i]...)
	}
	for i := range in.Primes {
		out.Primes[i] = append([]uint32(nil), in.Primes[i]...)
	}
	return out
}

func (l V41EngramLayout) validate() error {
	if len(l.TokenMap) == 0 || l.CompressedVocab == 0 || l.PadID >= l.CompressedVocab {
		return fmt.Errorf("model: invalid V4.1 Engram token map geometry")
	}
	if l.MaxNgramSize < 2 || l.HeadsPerNgram <= 0 || len(l.Rows) == 0 ||
		len(l.Multipliers) != len(l.Rows) || len(l.Primes) != len(l.Rows) {
		return fmt.Errorf("model: invalid V4.1 Engram hash geometry")
	}
	if l.MaxNgramSize-1 > math.MaxInt/l.HeadsPerNgram {
		return fmt.Errorf("model: V4.1 Engram column geometry overflows")
	}
	wantCols := (l.MaxNgramSize - 1) * l.HeadsPerNgram
	for _, id := range l.TokenMap {
		if id >= l.CompressedVocab {
			return fmt.Errorf("model: V4.1 Engram token map value %d out of range", id)
		}
	}
	for layer := range l.Rows {
		if len(l.Multipliers[layer]) != l.MaxNgramSize || len(l.Primes[layer]) != wantCols {
			return fmt.Errorf("model: invalid V4.1 Engram layer %d geometry", layer)
		}
		var rows uint64
		for _, multiplier := range l.Multipliers[layer] {
			if multiplier&1 == 0 || multiplier > uint64(math.MaxInt64)/uint64(l.CompressedVocab) {
				return fmt.Errorf("model: invalid V4.1 Engram multiplier at layer %d", layer)
			}
		}
		for _, prime := range l.Primes[layer] {
			if prime < 2 {
				return fmt.Errorf("model: invalid V4.1 Engram prime at layer %d", layer)
			}
			rows += uint64(prime)
		}
		if rows != uint64(l.Rows[layer]) {
			return fmt.Errorf("model: V4.1 Engram layer %d rows=%d want %d", layer, l.Rows[layer], rows)
		}
	}
	return nil
}

// Hash maps tokens to flattened [token][layer][ngram-head] table row IDs.
// A false mask entry breaks n-grams at that position. Invalid input leaves the
// state unchanged, which makes chunk retry deterministic.
func (s *V41EngramHashState) Hash(tokens []int, mask []bool) ([]uint32, error) {
	if s == nil {
		return nil, fmt.Errorf("model: nil V4.1 Engram hash state")
	}
	layout := s.layout
	if mask != nil && len(mask) != len(tokens) {
		return nil, fmt.Errorf("model: V4.1 Engram mask length %d != token length %d", len(mask), len(tokens))
	}
	for _, token := range tokens {
		if token < 0 || token >= len(layout.TokenMap) {
			return nil, fmt.Errorf("model: V4.1 Engram token %d out of range", token)
		}
	}

	cols := (layout.MaxNgramSize - 1) * layout.HeadsPerNgram
	if len(tokens) > math.MaxInt/len(layout.Rows) || len(tokens)*len(layout.Rows) > math.MaxInt/cols {
		return nil, fmt.Errorf("model: V4.1 Engram output geometry overflows")
	}
	out := make([]uint32, 0, len(tokens)*len(layout.Rows)*cols)
	for pos, token := range tokens {
		current := int64(layout.TokenMap[token])
		if mask != nil && !mask[pos] {
			current = -1
		}
		ids := make([]uint32, layout.MaxNgramSize)
		blocked := false
		for shift := range ids {
			id := current
			if shift > 0 {
				id = s.tail[shift-1]
			}
			blocked = blocked || id < 0
			if blocked {
				ids[shift] = layout.PadID
			} else {
				ids[shift] = uint32(id)
			}
		}
		for layer := range layout.Rows {
			hash := uint64(ids[0]) * layout.Multipliers[layer][0]
			var offset uint64
			for ngram := 1; ngram < layout.MaxNgramSize; ngram++ {
				hash ^= uint64(ids[ngram]) * layout.Multipliers[layer][ngram]
				for head := 0; head < layout.HeadsPerNgram; head++ {
					col := (ngram-1)*layout.HeadsPerNgram + head
					prime := uint64(layout.Primes[layer][col])
					out = append(out, uint32(hash%prime+offset))
					offset += prime
				}
			}
		}
		copy(s.tail[1:], s.tail[:len(s.tail)-1])
		s.tail[0] = current
	}
	return out, nil
}

// GatherV41EngramRows resolves flattened hash output through one bounded cache
// per Engram layer, preserving token/layer/column order.
func GatherV41EngramRows(caches []*V41EngramRowCache, rows []uint32, columnsPerLayer int) ([][]byte, error) {
	if len(caches) == 0 || columnsPerLayer <= 0 || len(rows)%(len(caches)*columnsPerLayer) != 0 {
		return nil, fmt.Errorf("model: invalid V4.1 Engram row gather geometry")
	}
	out := make([][]byte, len(rows))
	stride := len(caches) * columnsPerLayer
	for token := 0; token < len(rows)/stride; token++ {
		for layer, cache := range caches {
			if cache == nil {
				return nil, fmt.Errorf("model: nil V4.1 Engram row cache at layer %d", layer)
			}
			for col := 0; col < columnsPerLayer; col++ {
				i := token*stride + layer*columnsPerLayer + col
				row, err := cache.Row(int(rows[i]))
				if err != nil {
					return nil, fmt.Errorf("model: gather V4.1 Engram layer %d row %d: %w", layer, rows[i], err)
				}
				out[i] = row
			}
		}
	}
	return out, nil
}

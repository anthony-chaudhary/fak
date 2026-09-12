package model

// Portions adapted from antirez/ds4 at
// bd66c402070042bf0a79ad6ece8242de4c93680c (tests/test_engram.c).
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
	"reflect"
	"testing"
)

// Oracle adapted from antirez/ds4 tests/test_engram.c:14-103 at
// bd66c402070042bf0a79ad6ece8242de4c93680c under the MIT notice retained in
// v41_engram.go.
func TestV41EngramHashMatchesFullHistoryAcrossChunks(t *testing.T) {
	layout := toyV41EngramLayout()
	tokens := make([]int, 513)
	mask := make([]bool, len(tokens))
	for i := range tokens {
		tokens[i] = (i*97 + i/3) % 256
		mask[i] = i%17 != 0 && (i < 125 || i > 131)
	}
	for _, testMask := range [][]bool{nil, mask} {
		want := v41EngramReference(layout, tokens, testMask)
		for _, chunk := range []int{1, 2, 7, 64, 513} {
			state, err := NewV41EngramHashState(layout)
			if err != nil {
				t.Fatal(err)
			}
			var got []uint32
			for start := 0; start < len(tokens); start += chunk {
				end := min(start+chunk, len(tokens))
				var partMask []bool
				if testMask != nil {
					partMask = testMask[start:end]
				}
				part, err := state.Hash(tokens[start:end], partMask)
				if err != nil {
					t.Fatal(err)
				}
				got = append(got, part...)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("chunk=%d hash differs from independent full-history oracle", chunk)
			}
		}
	}
}

func toyV41EngramLayout() V41EngramLayout {
	mapIDs := make([]uint32, 256)
	for i := range mapIDs {
		mapIDs[i] = uint32(i / 2)
	}
	l := V41EngramLayout{TokenMap: mapIDs, CompressedVocab: 128, PadID: 1,
		Rows: make([]uint32, 2), Multipliers: make([][]uint64, 2), Primes: make([][]uint32, 2),
		MaxNgramSize: 4, HeadsPerNgram: 8}
	for layer := range l.Rows {
		l.Multipliers[layer] = make([]uint64, 4)
		for j := range l.Multipliers[layer] {
			l.Multipliers[layer][j] = 35184372088831 - uint64(2*(j+4*layer))
		}
		l.Primes[layer] = make([]uint32, 24)
		for j := range l.Primes[layer] {
			l.Primes[layer][j] = 16000057
			l.Rows[layer] += 16000057
		}
	}
	return l
}

func v41EngramReference(l V41EngramLayout, tokens []int, mask []bool) []uint32 {
	cols := (l.MaxNgramSize - 1) * l.HeadsPerNgram
	out := make([]uint32, 0, len(tokens)*len(l.Rows)*cols)
	for pos := range tokens {
		for layer := range l.Rows {
			var offset uint64
			for col := 0; col < cols; col++ {
				var hash uint64
				blocked := false
				for shift := 0; shift < col/l.HeadsPerNgram+2; shift++ {
					p := pos - shift
					blocked = blocked || p < 0 || (p >= 0 && mask != nil && !mask[p])
					id := l.PadID
					if !blocked {
						id = l.TokenMap[tokens[p]]
					}
					hash ^= uint64(id) * l.Multipliers[layer][shift]
				}
				prime := uint64(l.Primes[layer][col])
				out = append(out, uint32(hash%prime+offset))
				offset += prime
			}
		}
	}
	return out
}

func TestV41EngramHashRejectsInputWithoutAdvancingState(t *testing.T) {
	l := toyV41EngramLayout()
	s, _ := NewV41EngramHashState(l)
	if _, err := s.Hash([]int{7, len(l.TokenMap)}, nil); err == nil {
		t.Fatal("expected invalid token error")
	}
	got, err := s.Hash([]int{7}, nil)
	if err != nil {
		t.Fatal(err)
	}
	fresh, _ := NewV41EngramHashState(l)
	want, _ := fresh.Hash([]int{7}, nil)
	if !reflect.DeepEqual(got, want) {
		t.Fatal("rejected input advanced hash state")
	}
}

func TestV41EngramHashCloneAndBoundedGather(t *testing.T) {
	l := toyV41EngramLayout()
	s, _ := NewV41EngramHashState(l)
	if _, err := s.Hash([]int{11, 13}, nil); err != nil {
		t.Fatal(err)
	}
	fork := s.Clone()
	control := s.Clone()
	if _, err := s.Hash([]int{19}, nil); err != nil {
		t.Fatal(err)
	}
	got, _ := fork.Hash([]int{17}, nil)
	want, _ := control.Hash([]int{17}, nil)
	if !reflect.DeepEqual(got, want) {
		t.Fatal("clone aliases original tail state")
	}

	const rowBytes = 8
	caches := make([]*V41EngramRowCache, 2)
	for i := range caches {
		var err error
		caches[i], err = NewV41EngramRowCache(&v41CacheFakeEngram{rows: 64, rowBytes: rowBytes}, V41EngramRowCacheOptions{
			TableRows: 64, RowBytes: rowBytes, BudgetBytes: 2 * rowBytes,
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	rows, err := GatherV41EngramRows(caches, []uint32{1, 2, 3, 4}, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range rows {
		if len(row) != rowBytes || row[0] != byte(i+1) {
			t.Fatalf("gather row %d wrong: %v", i, row)
		}
	}
	if _, err := GatherV41EngramRows(caches, []uint32{1, 2, 65, 4}, 2); err == nil {
		t.Fatal("expected out-of-range gathered row to fail")
	}
}

// Copyright (c) 2026 DeepSeek
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
// Oracle adapted independently from deepseek-ai/DeepSeek-V4.1-Flash
// inference/model.py:561-610 at dba1be0a40aa45a94ad051997016db3960a90277
// (MIT), cross-checked against PipeNetwork/deepseek-v41-mlx indexer.py:38-64,
// 129-144 at a9567ae7196e596355b6fdaa7de7af4a2c92ba91 (Apache-2.0).

package v41

import (
	"math"
	"reflect"
	"testing"
)

func TestV41IndexerCandidateSelection(t *testing.T) {
	// Four blocks of two. Only positions 0..4 are visible. Block 2 is newest
	// and must be pinned despite its low score; blocks 3 is unreachable.
	logits := []float32{5, 1, 5, 4, -20, 100, 200, 300}
	mask, err := V41SelectCandidateBlocks(logits, 5, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Blocks 0 and 1 tie at five, so the explicit lower-index tie rule keeps
	// block 0 alongside pinned block 2.
	wantMask := []bool{true, true, false, false, true, true, false, false}
	if !reflect.DeepEqual(mask, wantMask) {
		t.Fatalf("candidate mask = %v, want %v", mask, wantMask)
	}

	rows, err := V41SelectIndexRows(logits, 5, 3, 11, mask)
	if err != nil {
		t.Fatal(err)
	}
	// The two visible block-0 rows and visible row 4 win, then are returned in
	// position order with the KV-layout offset.
	wantRows := []int32{11, 12, 15}
	if !reflect.DeepEqual(rows, wantRows) {
		t.Fatalf("rows = %v, want %v", rows, wantRows)
	}

	rows, err = V41SelectIndexRows(logits, 5, 6, 11, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantRows = []int32{11, 12, 13, 14, 15, -1}
	if !reflect.DeepEqual(rows, wantRows) {
		t.Fatalf("causal rows = %v, want %v", rows, wantRows)
	}

	rows, err = V41SelectIndexRows([]float32{9, 8, 7}, 3, 3, 20, []bool{false, false, true})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{-1, -1, 22}; !reflect.DeepEqual(rows, want) {
		t.Fatalf("masked rows = %v, want %v", rows, want)
	}
	rows, err = V41SelectIndexRows([]float32{5, 5, 1}, 3, 1, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{0}; !reflect.DeepEqual(rows, want) {
		t.Fatalf("tied rows = %v, want %v", rows, want)
	}

	badCalls := []func() error{
		func() error { _, err := V41SelectCandidateBlocks(logits, -1, 2, 2); return err },
		func() error { _, err := V41SelectCandidateBlocks(logits, 5, 2, 0); return err },
		func() error { _, err := V41SelectIndexRows(logits, 5, -1, 0, nil); return err },
		func() error { _, err := V41SelectIndexRows(logits, 5, 2, 0, []bool{true}); return err },
		func() error { _, err := V41SelectIndexRows(logits, 5, 2, math.MaxInt32, nil); return err },
	}
	for i, call := range badCalls {
		if err := call(); err == nil {
			t.Fatalf("bad call %d returned nil error", i)
		}
	}
}

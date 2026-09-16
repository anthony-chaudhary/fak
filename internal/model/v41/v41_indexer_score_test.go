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
// inference/model.py:488-580 at dba1be0a40aa45a94ad051997016db3960a90277
// (MIT). The reductions here are hand-transcribed per position and head; they
// do not call the production helper.

package v41

import (
	"math"
	"reflect"
	"testing"
)

// v41RefIndexerScore is a deliberately independent scalar transcription of the
// reference score reduction: per-head dot, per-element ReLU, then weighted head
// sum. It shares no code with V41IndexerScore.
func v41RefIndexerScore(q, keys, weights []float32, nHeads, headDim, compressLen int) []float32 {
	out := make([]float32, compressLen)
	for t := 0; t < compressLen; t++ {
		var acc float64
		for h := 0; h < nHeads; h++ {
			var dot float64
			for d := 0; d < headDim; d++ {
				dot += float64(q[h*headDim+d]) * float64(keys[t*headDim+d])
			}
			if dot < 0 {
				dot = 0
			}
			acc += dot * float64(weights[h])
		}
		out[t] = float32(acc)
	}
	return out
}

func v41Near(t *testing.T, got, want []float32, tol float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if d := float32(math.Abs(float64(got[i] - want[i]))); d > tol {
			t.Fatalf("%s: element %d = %g, want %g (delta %.3e > tol %.0e)", label, i, got[i], want[i], d, tol)
		}
	}
}

// TestV41IndexerScoreAndReuse is the leaf witness: it distinguishes
// ReLU-before-reduction from post-reduction ReLU, verifies selector composition
// against the accepted candidates API, and proves source publication reused by
// a reader without recomputation.
func TestV41IndexerScoreAndReuse(t *testing.T) {
	// Two heads of width two, so every dot is a hand-computable signed value.
	q := []float32{
		1, 1, // head 0: dot = k0 + k1
		1, -1, // head 1: dot = k0 - k1
	}
	// Four compressed positions of width two.
	keys := []float32{
		1, 1, // t0: head0=+2, head1=0
		1, -1, // t1: head0=0,  head1=+2
		2, 2, // t2: head0=+4, head1=0
		3, -1, // t3: head0=+2, head1=+4
	}
	weights := []float32{0.5, 2.0}
	const nHeads, headDim, compressLen = 2, 2, 4

	// Independent expectation, computed by the transcription above.
	want := v41RefIndexerScore(q, keys, weights, nHeads, headDim, compressLen)
	// Hand check: t0 = 0.5*2 + 2*0 = 1; t1 = 0 + 2*2 = 4;
	// t2 = 0.5*4 + 0 = 2; t3 = 0.5*2 + 2*4 = 9.
	if expect := []float32{1, 4, 2, 9}; !reflect.DeepEqual(want, expect) {
		t.Fatalf("independent transcription = %v, want %v", want, expect)
	}

	got, err := V41IndexerScore(q, keys, weights, nHeads, headDim, compressLen)
	if err != nil {
		t.Fatal(err)
	}
	v41Near(t, got, want, 1e-6, "indexer score")
	if want := []float32{1, 4, 2, 9}; !reflect.DeepEqual(got, want) {
		t.Fatalf("score = %v, want %v", got, want)
	}

	// Must-move control: post-reduction ReLU would use relu of the weighted sum
	// of an entire position. Flip head 1's weight negative on a position where
	// head 0 is positive and head 1 is positive too, so the two orderings
	// diverge. With q=(1,1)/(1,-1), t3 gives head dots (2, 4): with weights
	// (1, -2), ReLU-first = 2 - 8 = -6, while post-reduction ReLU gives
	// relu(2*1 + 4*(-2)) = relu(-6) = 0.
	flip := []float32{1.0, -2.0}
	flipScore, err := V41IndexerScore(q, keys, flip, nHeads, headDim, compressLen)
	if err != nil {
		t.Fatal(err)
	}
	if flipScore[3] != -6 {
		t.Fatalf("ReLU-before-reduction control: t3 = %g, want -6 (post-reduction ReLU would give 0)", flipScore[3])
	}

	// Source layer publishes candidate blocks plus final rows over the same
	// scores. Two blocks of two, keeping one block: block 1 holds t3=9 and is
	// the newest visible block, so it is pinned and kept as positions {2,3}.
	pub, err := v41IndexerPublish(0, q, keys, weights, nHeads, headDim, compressLen, 1, 2, 3, 100)
	if err != nil {
		t.Fatal(err)
	}
	if pub.layer != 0 || pub.compressLen != compressLen {
		t.Fatalf("publication identity = layer %d len %d", pub.layer, pub.compressLen)
	}
	wantMask := []bool{false, false, true, true}
	if gotMask := pub.Candidates(); !reflect.DeepEqual(gotMask, wantMask) {
		t.Fatalf("published candidates = %v, want %v", gotMask, wantMask)
	}
	// Rows come from the accepted selector over the same scores: the mask makes
	// t0 and t1 causal-invisible, so they occupy a top-k slot as -1; t2=2 and
	// t3=9 are selected, returned in position order and shifted by the offset.
	if wantRows := []int32{-1, 102, 103}; !reflect.DeepEqual(pub.Rows(), wantRows) {
		t.Fatalf("published rows = %v, want %v", pub.Rows(), wantRows)
	}

	// Selector composition: the published rows must equal a direct call to the
	// accepted candidates API on the same score vector.
	directMask, err := V41SelectCandidateBlocks(got, compressLen, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	directRows, err := V41SelectIndexRows(got, compressLen, 3, 100, directMask)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pub.Rows(), directRows) || !reflect.DeepEqual(pub.Candidates(), directMask) {
		t.Fatalf("publication (%v, %v) diverges from direct selector (%v, %v)",
			pub.Candidates(), pub.Rows(), directMask, directRows)
	}

	// Prefill/step reuse equality: a reader with identical inputs reproduces the
	// published rows without recomputing the source scoring, and repeated reuse
	// is stable.
	for i := 0; i < 2; i++ {
		reuseRows, err := pub.Reuse(q, keys, weights, nHeads, headDim, compressLen, 3, 100)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(reuseRows, pub.Rows()) {
			t.Fatalf("reuse %d rows = %v, want published %v", i, reuseRows, pub.Rows())
		}
	}

	// Definitive mutability: a reader gets copies; mutating one cannot change
	// the publication or a later reader.
	reader := pub.Candidates()
	reader[0] = true
	if pub.Candidates()[0] {
		t.Fatalf("reader mutated the published candidate mask")
	}
	rowsCopy := pub.Rows()
	rowsCopy[2] = -99
	if pub.Rows()[2] != 103 {
		t.Fatalf("reader mutated the published rows")
	}

	// Non-source layer publishes no candidate mask but still publishes rows.
	readerPub, err := v41IndexerPublish(1, q, keys, weights, nHeads, headDim, compressLen, 0, 0, 2, 50)
	if err != nil {
		t.Fatal(err)
	}
	if readerPub.Candidates() != nil {
		t.Fatalf("non-source layer published candidates: %v", readerPub.Candidates())
	}
	if wantRows := []int32{51, 53}; !reflect.DeepEqual(readerPub.Rows(), wantRows) {
		t.Fatalf("non-source rows = %v, want %v", readerPub.Rows(), wantRows)
	}

	// Fail-closed geometry: a wrong key length or a non-finite weight must be
	// refused, not silently truncated.
	badCalls := []func() error{
		func() error {
			_, err := V41IndexerScore(q, keys[:3], weights, nHeads, headDim, compressLen)
			return err
		},
		func() error {
			_, err := V41IndexerScore(q, keys, weights[:1], nHeads, headDim, compressLen)
			return err
		},
		func() error {
			_, err := V41IndexerScore(q, keys, []float32{1, float32(math.Inf(1))}, nHeads, headDim, compressLen)
			return err
		},
		func() error {
			_, err := V41IndexerScore(q[:3], keys, weights, nHeads, headDim, compressLen)
			return err
		},
		func() error { _, err := V41IndexerScore(q, keys, weights, 0, headDim, compressLen); return err },
		func() error { _, err := V41IndexerScore(q, keys, weights, nHeads, headDim, -1); return err },
		func() error {
			_, err := v41IndexerPublish(0, q, keys, weights, nHeads, headDim, compressLen, 1, 2, -1, 0)
			return err
		},
		func() error { _, err := pub.Reuse(q, keys, weights, nHeads, headDim, compressLen+1, 3, 0); return err },
	}
	for i, call := range badCalls {
		if err := call(); err == nil {
			t.Fatalf("bad call %d returned nil error", i)
		}
	}
	if _, err := (*v41IndexerPublication)(nil).Reuse(q, keys, weights, nHeads, headDim, compressLen, 3, 0); err == nil {
		t.Fatalf("reuse on nil publication returned nil error")
	}
}

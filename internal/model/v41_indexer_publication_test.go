package model

import (
	"reflect"
	"testing"
)

// v41_indexer_publication_test.go - the leaf witness for the V4.1 indexer
// publication/reuse seam on the single reachable V41IndexerScore implementation
// (fak#13792). The retired internal/model/v41 fork asserted the deep-copy
// ownership, the non-source-layer mask absence, the reuse equality, the
// selector composition and the fail-closed geometry; those assertions moved here
// when the duplicate was collapsed, so the one remaining implementation keeps
// its leaf contract covered.
//
// Hand-computable fixture, no device, no checkpoint, no full-model generation.
// fak-test:runtime fast est=1ms lane=default
func TestV41IndexerPublicationReuseAndOwnership(t *testing.T) {
	// Two heads of width two over four compressed positions, so every per-head
	// dot is a small signed integer.
	q := []float32{
		1, 1, // head 0: dot = k0 + k1
		1, -1, // head 1: dot = k0 - k1
	}
	keys := []float32{
		1, 1, // t0: head0=+2, head1=0
		1, -1, // t1: head0=0,  head1=+2
		2, 2, // t2: head0=+4, head1=0
		3, -1, // t3: head0=+2, head1=+4
	}
	weights := []float32{0.5, 2.0}
	const nHeads, headDim, compressLen = 2, 2, 4

	got, err := V41IndexerScore(q, keys, weights, nHeads, headDim, compressLen)
	if err != nil {
		t.Fatal(err)
	}
	// Independent float64 transcription of the reference reduction. Hand check:
	// t0 = 0.5*2 + 2*0 = 1; t1 = 0 + 2*2 = 4; t2 = 0.5*4 + 0 = 2; t3 = 0.5*2 + 2*4 = 9.
	want := v41RefIndexerScoreExec(q, keys, weights, nHeads, headDim, compressLen)
	if expect := []float32{1, 4, 2, 9}; !reflect.DeepEqual(want, expect) || !reflect.DeepEqual(got, expect) {
		t.Fatalf("score = %v, independent transcription %v, want %v", got, want, expect)
	}

	// Must-move control: post-reduction ReLU would use relu of the weighted sum
	// of an entire position. Flip head 1's weight negative on a position where
	// both head dots are positive, so the two orderings diverge: with weights
	// (1, -2), ReLU-first = 2 - 8 = -6, while post-reduction ReLU gives 0.
	flipScore, err := V41IndexerScore(q, keys, []float32{1.0, -2.0}, nHeads, headDim, compressLen)
	if err != nil {
		t.Fatal(err)
	}
	if flipScore[3] != -6 {
		t.Fatalf("ReLU-before-reduction control: t3 = %g, want -6 (post-reduction ReLU would give 0)", flipScore[3])
	}

	// Source layer publishes candidate blocks plus final rows over the same
	// scores. Two blocks of two, keeping one block: block 1 holds t3=9 and is the
	// newest visible block, so it is pinned and kept as positions {2,3}.
	pub, err := NewV41IndexerPublication(0, q, keys, weights, nHeads, headDim, compressLen, 1, 2, 3, 100)
	if err != nil {
		t.Fatal(err)
	}
	wantMask := []bool{false, false, true, true}
	if gotMask := pub.Candidates(); !v41RefMaskEqual(gotMask, wantMask) {
		t.Fatalf("published candidates = %v, want %v", gotMask, wantMask)
	}
	// Rows come from the accepted selector over the same scores: the mask makes
	// t0 and t1 causal-invisible, so they occupy a top-k slot as -1; t2=2 and
	// t3=9 are selected, returned in position order and shifted by the offset.
	if wantRows := []int32{-1, 102, 103}; !v41RefRowsEqual(pub.Rows(), wantRows) {
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
	if !v41RefRowsEqual(pub.Rows(), directRows) || !v41RefMaskEqual(pub.Candidates(), directMask) {
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
		if !v41RefRowsEqual(reuseRows, pub.Rows()) {
			t.Fatalf("reuse %d rows = %v, want published %v", i, reuseRows, pub.Rows())
		}
	}

	// Definitive mutability: a reader gets copies; mutating one cannot change the
	// publication or a later reader.
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
	readerPub, err := NewV41IndexerPublication(1, q, keys, weights, nHeads, headDim, compressLen, 0, 0, 2, 50)
	if err != nil {
		t.Fatal(err)
	}
	if readerPub.Candidates() != nil {
		t.Fatalf("non-source layer published candidates: %v", readerPub.Candidates())
	}
	if wantRows := []int32{51, 53}; !v41RefRowsEqual(readerPub.Rows(), wantRows) {
		t.Fatalf("non-source rows = %v, want %v", readerPub.Rows(), wantRows)
	}

	// Fail-closed geometry: a negative top-k, a wrong reuse compress length, and
	// a nil publication must be refused, not silently accepted.
	if _, err := NewV41IndexerPublication(0, q, keys, weights, nHeads, headDim, compressLen, 1, 2, -1, 0); err == nil {
		t.Fatalf("negative top-k publication returned nil error")
	}
	if _, err := pub.Reuse(q, keys, weights, nHeads, headDim, compressLen+1, 3, 0); err == nil {
		t.Fatalf("reuse with a mismatched compress length returned nil error")
	}
	if _, err := (*V41IndexerPublication)(nil).Reuse(q, keys, weights, nHeads, headDim, compressLen, 3, 0); err == nil {
		t.Fatalf("reuse on nil publication returned nil error")
	}
}

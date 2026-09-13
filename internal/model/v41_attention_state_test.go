package model

import (
	"math"
	"testing"
)

// makeKV returns a deterministic projected KV row tagged by absolute position
// and dimension, so any state divergence is visible in the comparison.
func makeKV(dim, pos int) []float32 {
	row := make([]float32, dim)
	for d := range row {
		row[d] = float32(pos*10 + d + 1)
	}
	return row
}

// TestV41AttentionStatePrefillStep is the leaf witness: a full-sequence Prefill
// and an equivalent token-by-token Step sequence must leave identical logical
// window, compressed-row, index-key, and selection state.
func TestV41AttentionStatePrefillStep(t *testing.T) {
	const (
		dim   = 4
		ratio = 2
		total = 6
	)
	ref := V41AttentionStateRef{LayerID: 2, Ratio: ratio, IsKVSource: true, IsIndexSource: true}

	// Prefill path: one chunk of `total` positions, publishing a latent and an
	// index key for every completed group (positions ratio, 2*ratio, ...).
	prefillState, err := NewV41AttentionState(dim, ratio)
	if err != nil {
		t.Fatal(err)
	}
	chunk := make([][]float32, total)
	for i := range chunk {
		chunk[i] = makeKV(dim, i)
	}
	var prefillUpdates []V41AttentionStateUpdate
	for group := 1; group*ratio <= total; group++ {
		up := V41AttentionStateUpdate{Ref: ref}
		up.Latent = makeKV(dim, group*ratio-1)
		up.IndexKey = makeKV(dim, group*ratio-1)
		prefillUpdates = append(prefillUpdates, up)
	}
	if err := prefillState.Prefill(chunk, prefillUpdates); err != nil {
		t.Fatalf("prefill: %v", err)
	}

	// Step path: prefill position 0, then step through positions 1..total-1,
	// supplying each group's latent/index key at the step that completes it.
	stepState, err := NewV41AttentionState(dim, ratio)
	if err != nil {
		t.Fatal(err)
	}
	if err := stepState.Prefill(chunk[:1], nil); err != nil {
		t.Fatalf("step prefill: %v", err)
	}
	for pos := 1; pos < total; pos++ {
		var updates []V41AttentionStateUpdate
		if (pos+1)%ratio == 0 {
			up := V41AttentionStateUpdate{Ref: ref}
			up.Latent = makeKV(dim, pos)
			up.IndexKey = makeKV(dim, pos)
			updates = append(updates, up)
		}
		if err := stepState.Step(makeKV(dim, pos), updates); err != nil {
			t.Fatalf("step %d: %v", pos, err)
		}
	}

	// Logical window must match position-for-position.
	pw := prefillState.WindowKV()
	sw := stepState.WindowKV()
	for i := range pw {
		for d := range pw[i] {
			if pw[i][d] != sw[i][d] {
				t.Fatalf("window[%d][%d]: prefill=%g step=%g", i, d, pw[i][d], sw[i][d])
			}
		}
	}

	// Compressed rows and index keys must match in count and value.
	pr, ok := prefillState.KVSourceRows(ref.LayerID)
	if !ok {
		t.Fatal("prefill published no KV rows")
	}
	sr, ok := stepState.KVSourceRows(ref.LayerID)
	if !ok {
		t.Fatal("step published no KV rows")
	}
	if len(pr) != len(sr) {
		t.Fatalf("KV row count: prefill=%d step=%d", len(pr), len(sr))
	}
	for g := range pr {
		for d := range pr[g] {
			if pr[g][d] != sr[g][d] {
				t.Fatalf("KV row %d dim %d: prefill=%g step=%g", g, d, pr[g][d], sr[g][d])
			}
		}
	}
	pk, _ := prefillState.IndexKeys(ref.LayerID)
	sk, _ := stepState.IndexKeys(ref.LayerID)
	if len(pk) != len(sk) {
		t.Fatalf("index key count: prefill=%d step=%d", len(pk), len(sk))
	}
	for g := range pk {
		for d := range pk[g] {
			if pk[g][d] != sk[g][d] {
				t.Fatalf("index key %d dim %d: prefill=%g step=%g", g, d, pk[g][d], sk[g][d])
			}
		}
	}

	// The number of completed groups must equal floor(total/ratio).
	if want := total / ratio; len(pr) != want {
		t.Fatalf("completed groups=%d want %d", len(pr), want)
	}
}

// TestV41AttentionStateReadersSeeLatestPublication covers candidate/top-k
// publication and reuse, plus the readers returning the publisher ratio.
func TestV41AttentionStateReadersSeeLatestPublication(t *testing.T) {
	s, err := NewV41AttentionState(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, ok := s.Candidates(); ok {
		t.Fatal("empty state reported candidates")
	}
	if _, _, ok := s.TopK(); ok {
		t.Fatal("empty state reported top-k")
	}
	if err := s.PublishCandidates(2, []bool{true, false, true}); err != nil {
		t.Fatal(err)
	}
	first, ratio, ok := s.Candidates()
	if !ok || ratio != 2 || len(first) != 3 || !first[0] || first[1] || !first[2] {
		t.Fatalf("candidates=(%v,%d,%v)", first, ratio, ok)
	}
	// A second publication replaces the first; readers observe the latest.
	if err := s.PublishCandidates(1, []bool{false, true}); err != nil {
		t.Fatal(err)
	}
	second, ratio, ok := s.Candidates()
	if !ok || ratio != 1 || len(second) != 2 || second[0] || !second[1] {
		t.Fatalf("updated candidates=(%v,%d,%v)", second, ratio, ok)
	}
	// The returned slice is a copy: mutating it must not change state.
	second[0] = true
	third, _, _ := s.Candidates()
	if third[0] {
		t.Fatal("candidates reader aliased internal state")
	}

	rows := [][]int32{{3, 1}, {4, 2}}
	if err := s.PublishTopK(2, rows); err != nil {
		t.Fatal(err)
	}
	got, trov, ok := s.TopK()
	if !ok || trov != 2 || len(got) != 2 {
		t.Fatalf("topk=(%v,%d,%v)", got, trov, ok)
	}
	rows[0][0] = 99
	got2, _, _ := s.TopK()
	if got2[0][0] == 99 {
		t.Fatal("top-k reader aliased caller input")
	}
}

// TestV41AttentionStateMalformedUpdatesAreAtomic proves a validation failure
// leaves the receiver unchanged, matching the leaf fail-atomic requirement.
func TestV41AttentionStateMalformedUpdatesAreAtomic(t *testing.T) {
	const dim = 4
	s, err := NewV41AttentionState(dim, 4)
	if err != nil {
		t.Fatal(err)
	}
	chunk := [][]float32{makeKV(dim, 0), makeKV(dim, 1)}
	good := V41AttentionStateUpdate{
		Ref:    V41AttentionStateRef{LayerID: 1, Ratio: 2, IsKVSource: true},
		Latent: makeKV(dim, 1),
	}
	if err := s.Prefill(chunk, []V41AttentionStateUpdate{good}); err != nil {
		t.Fatal(err)
	}
	before := s.WindowKV()

	cases := []struct {
		name    string
		updates []V41AttentionStateUpdate
		pos     int
	}{
		{"non-source supplies latent", []V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 3, Ratio: 2}, Latent: makeKV(dim, 0)}}, 2},
		{"ratio zero supplies latent", []V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 3, Ratio: 0, IsKVSource: true}, Latent: makeKV(dim, 0)}}, 2},
		{"non-index supplies key", []V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 4, Ratio: 2, IsKVSource: true}, Latent: makeKV(dim, 0), IndexKey: makeKV(dim, 0)}}, 2},
		{"short latent", []V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 5, Ratio: 2, IsKVSource: true}, Latent: []float32{1, 2}}}, 2},
		{"non-finite latent", []V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 6, Ratio: 2, IsKVSource: true}, Latent: []float32{1, float32(math.Inf(1)), 3, 4}}}, 2},
		{"out-of-order layer", []V41AttentionStateUpdate{
			{Ref: V41AttentionStateRef{LayerID: 9, Ratio: 2, IsKVSource: true}, Latent: makeKV(dim, 0)},
			{Ref: V41AttentionStateRef{LayerID: 7, Ratio: 2, IsKVSource: true}, Latent: makeKV(dim, 0)},
		}, 2},
	}
	for _, tc := range cases {
		err := s.Step(makeKV(dim, tc.pos), tc.updates)
		if err == nil {
			t.Fatalf("%s: accepted malformed update", tc.name)
		}
		after := s.WindowKV()
		for i := range before {
			for d := range before[i] {
				if before[i][d] != after[i][d] {
					t.Fatalf("%s: window mutated on failure", tc.name)
				}
			}
		}
	}
	if _, ok := s.KVSourceRows(3); ok {
		t.Fatal("failed update published a row")
	}
	if _, _, ok := s.Candidates(); ok {
		t.Fatal("failed update published candidates")
	}
}

// TestV41AttentionStateResetClearsOwnership proves Reset removes every row and
// publication so a reused state cannot leak a prior session.
func TestV41AttentionStateResetClearsOwnership(t *testing.T) {
	const dim = 4
	s, err := NewV41AttentionState(dim, 4)
	if err != nil {
		t.Fatal(err)
	}
	ref := V41AttentionStateRef{LayerID: 2, Ratio: 2, IsKVSource: true, IsIndexSource: true}
	chunk := [][]float32{makeKV(dim, 0), makeKV(dim, 1)}
	up := V41AttentionStateUpdate{Ref: ref, Latent: makeKV(dim, 1), IndexKey: makeKV(dim, 1)}
	if err := s.Prefill(chunk, []V41AttentionStateUpdate{up}); err != nil {
		t.Fatal(err)
	}
	if err := s.PublishCandidates(2, []bool{true}); err != nil {
		t.Fatal(err)
	}
	if err := s.PublishTopK(2, [][]int32{{0}}); err != nil {
		t.Fatal(err)
	}

	s.Reset()

	if _, ok := s.KVSourceRows(ref.LayerID); ok {
		t.Fatal("Reset left KV rows")
	}
	if _, ok := s.IndexKeys(ref.LayerID); ok {
		t.Fatal("Reset left index keys")
	}
	if _, _, ok := s.Candidates(); ok {
		t.Fatal("Reset left candidates")
	}
	if _, _, ok := s.TopK(); ok {
		t.Fatal("Reset left top-k")
	}
	w := s.WindowKV()
	for i := range w {
		for d := range w[i] {
			if w[i][d] != 0 {
				t.Fatalf("Reset left window[%d][%d]=%g", i, d, w[i][d])
			}
		}
	}
	// A reset state accepts a fresh prefill of the same geometry.
	if err := s.Prefill(chunk, nil); err != nil {
		t.Fatalf("post-reset prefill: %v", err)
	}
}

// TestV41AttentionStateWindowIsBoundedRing proves the ring overwrites older
// slots at the pinned window size rather than growing without bound.
func TestV41AttentionStateWindowIsBoundedRing(t *testing.T) {
	const dim = 2
	s, err := NewV41AttentionState(dim, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Prefill([][]float32{makeKV(dim, 0)}, nil); err != nil {
		t.Fatal(err)
	}
	// Advance one full window past the prefill so every original slot is
	// overwritten, then read the logical window oldest-first.
	for pos := 1; pos <= v41WindowSize; pos++ {
		if err := s.Step(makeKV(dim, pos), nil); err != nil {
			t.Fatalf("step %d: %v", pos, err)
		}
	}
	w := s.WindowKV()
	if len(w) != v41WindowSize {
		t.Fatalf("window length=%d want %d", len(w), v41WindowSize)
	}
	if w[v41WindowSize-1][0] != float32(v41WindowSize*10+1) {
		t.Fatalf("newest window row=%v want position %d", w[v41WindowSize-1], v41WindowSize)
	}
	if w[0][0] != float32(1*10+1) {
		t.Fatalf("oldest window row=%v want position 1", w[0])
	}
}

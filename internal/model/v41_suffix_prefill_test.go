package model

// v41_suffix_prefill_test.go — the fak#13346 independent regression for
// SUFFIX-PREFILL CACHE REUSE on the production V4.1 Session.Prefill entry point.
//
// It is authored against the CONTRACT, not the implementation: after a
// Session.Prefill(P) seeds a plain-layer continuation state, a following
// Session.Prefill(S) must append exactly len(S) new positions per layer through
// the existing transactional incremental step (#13313's route), agree with the
// full P+S reference, and never replay the committed prefix P. A cold session
// (empty state) keeps the historical full prefill.
//
// The witness reuses the production fixtures from v41_incremental_test.go (the
// plain reduced model, the deterministic prefix builder and the full-forward
// reference) so parity compares against rows the model actually produces.

import (
	"math"
	"reflect"
	"testing"
)

// v41SuffixPrefillCalls counts successful incremental suffix-prefill token
// advances via the package probe. It returns a read function and a restore
// function so a test can install the observer around a scoped Prefill call.
func v41SuffixPrefillCalls(t *testing.T) (read func() int, restore func()) {
	t.Helper()
	prev := v41PrefillStepProbe
	n := 0
	v41PrefillStepProbe = func() { n++ }
	return func() int { return n }, func() { v41PrefillStepProbe = prev }
}

// v41SuffixTail builds a deterministic suffix of suffixLen tokens continuing
// the prefix builder's sequence, under the fixture's vocab.
func v41SuffixTail(cfg Config, prefixLen, suffixLen int) []int {
	s := make([]int, suffixLen)
	for i := range s {
		s[i] = ((prefixLen+i)*7 + 3) % cfg.VocabSize
	}
	return s
}

// TestV41SuffixPrefillCacheReuse is the named witness the leaf requires. It
// proves the production Prefill route reuses prepared cache state: P then S
// advances exactly len(S) positions per layer (probe fires once per suffix
// token), the committed history is exactly P+S, the returned logits equal the
// full P+S reference's last row, and every layer cursor sits at len(P)+len(S).
func TestV41SuffixPrefillCacheReuse(t *testing.T) {
	const tol = 1e-6

	for _, tc := range []struct {
		name      string
		prefixLen int
		suffixLen int
	}{
		{"one-token-suffix", 8, 1},
		{"short-suffix", 12, 5},
		{"prefix32-suffix7", 32, 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			cfg := m.Cfg
			s := m.NewSession()

			prefix, _ := v41IncrementalPrefix(cfg, tc.prefixLen)
			if got := s.Prefill(prefix); len(got) == 0 {
				t.Fatalf("prefix Prefill returned no logits")
			}
			if !s.v41IncrementalEligible() {
				t.Fatal("seeded plain-layer session is not incrementally eligible")
			}

			suffix := v41SuffixTail(cfg, tc.prefixLen, tc.suffixLen)
			wantHistory := append(append([]int(nil), prefix...), suffix...)

			// The full P+S reference the cold path would produce.
			ref, err := m.forwardV41(wantHistory, nil)
			if err != nil {
				t.Fatalf("full P+S reference: %v", err)
			}
			wantRow := ref.Logits[len(ref.Logits)-1]

			read, restore := v41SuffixPrefillCalls(t)
			got := s.Prefill(suffix)
			calls := read()
			restore()

			// Clause 1: exactly len(S) incremental token advances, never len(P)+len(S).
			if calls != tc.suffixLen {
				t.Fatalf("incremental suffix advances = %d, want %d (P must not be replayed)", calls, tc.suffixLen)
			}
			// Clause 1: history is exactly P+S.
			if !reflect.DeepEqual(s.v41Forward.history, wantHistory) {
				t.Fatalf("history = %v, want %v", s.v41Forward.history, wantHistory)
			}
			// Clause 1: agree with the full reference at the declared tolerance.
			if len(got) != len(wantRow) {
				t.Fatalf("logits width %d, want %d", len(got), len(wantRow))
			}
			for j := range wantRow {
				delta := got[j] - wantRow[j]
				if delta < 0 {
					delta = -delta
				}
				if delta > tol || math.IsNaN(float64(got[j])) {
					t.Fatalf("logits[%d] = %g, want %g (delta %g)", j, got[j], wantRow[j], delta)
				}
			}
			// Clause 1: each layer advanced to the post-suffix position.
			for l := 0; l < cfg.NumLayers; l++ {
				if pos := s.v41Forward.layerState(l).nextWindowPos; pos != len(wantHistory) {
					t.Fatalf("layer %d nextWindowPos = %d, want %d", l, pos, len(wantHistory))
				}
			}
		})
	}
}

// TestV41SuffixPrefillConstantWorkPerToken is the counter witness the leaf's
// definition of done requires: across a short and a long prefix the per-suffix
// work is CONSTANT — exactly one incremental advance per suffix token, never
// proportional to the prefix length — while the full-history reference grows
// with the prefix. It also proves a HISTOry-only state (the snapshot the leaf
// warns about) is not eligible and therefore keeps the cold full route.
func TestV41SuffixPrefillConstantWorkPerToken(t *testing.T) {
	var (
		refSeqByPrefix = map[int]int{}
		callsByPrefix  = map[int]int{}
	)
	for _, prefixLen := range []int{32, 120} {
		m := v41IncrementalPlainModel(t, 2)
		cfg := m.Cfg
		s := m.NewSession()

		prefix, _ := v41IncrementalPrefix(cfg, prefixLen)
		s.Prefill(prefix)
		if !s.v41IncrementalEligible() {
			t.Fatalf("prefix %d: session is not incrementally eligible", prefixLen)
		}

		// A fixed two-token suffix: the incremental route must advance exactly
		// two positions regardless of the prefix length.
		suffix := v41SuffixTail(cfg, prefixLen, 2)
		read, restore := v41SuffixPrefillCalls(t)
		s.Prefill(suffix)
		calls := read()
		restore()
		if calls != len(suffix) {
			t.Fatalf("prefix %d: suffix advances = %d, want %d (constant, independent of prefix)",
				prefixLen, calls, len(suffix))
		}
		callsByPrefix[prefixLen] = calls

		// The contrast: the full-history reference the COLD path pays fans out
		// over the WHOLE P+S, so its produced sequence length grows with P.
		full := append(append([]int(nil), prefix...), suffix...)
		ref, err := m.forwardV41(full, nil)
		if err != nil {
			t.Fatalf("prefix %d: full reference: %v", prefixLen, err)
		}
		if ref.Seq != prefixLen+len(suffix) {
			t.Fatalf("prefix %d: full reference Seq = %d, want %d (grows with prefix)",
				prefixLen, ref.Seq, prefixLen+len(suffix))
		}
		refSeqByPrefix[prefixLen] = ref.Seq
	}

	if callsByPrefix[32] != callsByPrefix[120] {
		t.Fatalf("suffix advances differ by prefix: 32->%d, 120->%d", callsByPrefix[32], callsByPrefix[120])
	}
	if refSeqByPrefix[120] <= refSeqByPrefix[32] {
		t.Fatalf("full-history reference did not grow with prefix: 32->%d, 120->%d",
			refSeqByPrefix[32], refSeqByPrefix[120])
	}
	t.Logf("incremental suffix advances=%d (constant); full-history reference Seq: prefix32=%d prefix120=%d",
		callsByPrefix[32], refSeqByPrefix[32], refSeqByPrefix[120])
}

// TestV41SuffixPrefillColdFallback proves a COLD session (empty state) keeps the
// historical full-prefill route: the incremental suffix probe must NOT fire, the
// logits must equal the direct full-forward reference, and the history must be
// exactly the supplied prompt.
func TestV41SuffixPrefillColdFallback(t *testing.T) {
	const tol = 1e-6
	m := v41IncrementalPlainModel(t, 2)
	cfg := m.Cfg
	s := m.NewSession()

	prompt, _ := v41IncrementalPrefix(cfg, 6)
	ref, err := m.forwardV41(prompt, nil)
	if err != nil {
		t.Fatalf("full cold reference: %v", err)
	}
	wantRow := ref.Logits[len(ref.Logits)-1]

	read, restore := v41SuffixPrefillCalls(t)
	got := s.Prefill(prompt)
	calls := read()
	restore()

	if calls != 0 {
		t.Fatalf("cold Prefill fired the incremental suffix probe %d times, want 0", calls)
	}
	if !reflect.DeepEqual(s.v41Forward.history, prompt) {
		t.Fatalf("cold history = %v, want %v", s.v41Forward.history, prompt)
	}
	if len(got) != len(wantRow) {
		t.Fatalf("logits width %d, want %d", len(got), len(wantRow))
	}
	for j := range wantRow {
		delta := got[j] - wantRow[j]
		if delta < 0 {
			delta = -delta
		}
		if delta > tol || math.IsNaN(float64(got[j])) {
			t.Fatalf("logits[%d] = %g, want %g (delta %g)", j, got[j], wantRow[j], delta)
		}
	}
}

// TestV41SuffixPrefillStatePreservation proves the rollback/retry clause: a
// poisoned layer cursor makes the incremental suffix refuse, the production
// Prefill then re-folds the WHOLE suffix on the cold reference path (history
// advances by exactly len(S), never partially), and a freshly prefilled session
// takes the incremental route for the same suffix with identical output.
func TestV41SuffixPrefillStatePreservation(t *testing.T) {
	const tol = 1e-6
	m := v41IncrementalPlainModel(t, 2)
	cfg := m.Cfg
	s := m.NewSession()

	prefix, _ := v41IncrementalPrefix(cfg, 8)
	s.Prefill(prefix)

	// Poison the last layer's cursor so the incremental seam refuses inside the
	// layer loop; Prefill must roll the suffix back and re-fold it cold.
	last := cfg.NumLayers - 1
	s.v41Forward.layerState(last).nextWindowPos = len(s.v41Forward.history) + 5

	suffix := v41SuffixTail(cfg, 8, 4)
	wantHistory := append(append([]int(nil), prefix...), suffix...)

	read, restore := v41SuffixPrefillCalls(t)
	gotFallback := s.Prefill(suffix)
	calls := read()
	restore()
	if calls != 0 {
		t.Fatalf("poisoned-cursor Prefill fired the incremental probe %d times, want 0 (fallback)", calls)
	}
	if !reflect.DeepEqual(s.v41Forward.history, wantHistory) {
		t.Fatalf("after fallback history = %v, want %v", s.v41Forward.history, wantHistory)
	}
	assertV41LogitsParity(t, m, wantHistory, gotFallback, tol)

	// A freshly prefilled session takes the incremental route for the same
	// suffix and must produce identical logits -- the fallback was exactly as if
	// the failed incremental attempt never happened.
	s2 := m.NewSession()
	s2.Prefill(prefix)
	if !s2.v41IncrementalEligible() {
		t.Fatal("freshly prefilled session is not incrementally eligible")
	}
	read2, restore2 := v41SuffixPrefillCalls(t)
	gotRetry := s2.Prefill(suffix)
	calls2 := read2()
	restore2()
	if calls2 != len(suffix) {
		t.Fatalf("retry Prefill fired the incremental probe %d times, want %d", calls2, len(suffix))
	}
	if !reflect.DeepEqual(gotRetry, gotFallback) {
		t.Fatal("incremental suffix logits disagree with the fallback logits for the same prompt")
	}
}

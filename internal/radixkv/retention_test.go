package radixkv

import "testing"

// seqTokens builds an n-token sequence whose first token is unique to id, so distinct
// ids attach as separate leaves under the root (no shared prefix to collapse).
func seqTokens(id, n int) []int {
	s := make([]int, n)
	for j := range s {
		s[j] = id*1000 + j
	}
	return s
}

// insertUnleased inserts a full sequence and immediately releases its lease, so the
// leaf is a normal LRU-evictable candidate (nothing pins it against the budget).
func insertUnleased(t *Tree, tokens []int) {
	b, matched := t.Lookup(tokens)
	leaf := t.Insert(b, tokens[matched:], nil)
	t.Done(leaf)
}

// TestRetentionSignedDial is the #4039 witness: radixkv exposes ONE signed retention
// dial with a single zero-point -- <0 keep-all, 0 keep-none, >0 evict-to-N -- matching
// ctxresidency's grammar. Four sub-tests: the three signs, plus a regression that the
// legacy New() default preserves current behavior byte-for-byte.
func TestRetentionSignedDial(t *testing.T) {
	// >0 evict-to-N: bound the tree to 8 tokens; inserting 5x4=20 tokens over time keeps
	// the cache within budget and records evictions.
	t.Run("positive_evicts_to_N", func(t *testing.T) {
		tr := New(0)
		tr.SetRetention(8)
		for i := 0; i < 5; i++ {
			insertUnleased(tr, seqTokens(i, 4))
		}
		st := tr.Stats()
		if st.Tokens > 8 {
			t.Errorf("evict-to-8: Tokens=%d, want <= 8", st.Tokens)
		}
		if st.Evictions == 0 {
			t.Errorf("evict-to-8: Evictions=0, want > 0 (budget pressure should evict)")
		}
		if st.MaxTokens != 8 {
			t.Errorf("evict-to-8: Stats.MaxTokens=%d, want 8", st.MaxTokens)
		}
	})

	// 0 keep-none: retain nothing. Build an unbounded tree, fill it, then flip the dial
	// to 0 -- every unleased leaf is evicted back toward empty.
	t.Run("zero_keeps_none", func(t *testing.T) {
		tr := New(0) // unbounded to start, so both inserts are retained
		insertUnleased(tr, seqTokens(0, 4))
		insertUnleased(tr, seqTokens(1, 4))
		if tr.Stats().Tokens == 0 {
			t.Fatal("precondition: expected cached tokens before keep-none")
		}
		tr.SetRetention(0) // keep-none drains the now-unleased tree
		if got := tr.Stats().Tokens; got != 0 {
			t.Errorf("keep-none: Tokens=%d, want 0 (retain nothing)", got)
		}
		// A subsequent insert is admitted (leased) but does not accumulate: once released
		// the next budget pass reclaims it, so keep-none holds steady at empty.
		insertUnleased(tr, seqTokens(2, 4))
		tr.SetRetention(0)
		if got := tr.Stats().Tokens; got != 0 {
			t.Errorf("keep-none after reinsert: Tokens=%d, want 0", got)
		}
	})

	// <0 keep-all: disable eviction even though a positive legacy budget was configured.
	t.Run("negative_keeps_all", func(t *testing.T) {
		tr := New(4)        // legacy bound of 4 tokens...
		tr.SetRetention(-1) // ...overridden by keep-all
		for i := 0; i < 5; i++ {
			insertUnleased(tr, seqTokens(i, 4))
		}
		st := tr.Stats()
		if st.Tokens != 20 {
			t.Errorf("keep-all: Tokens=%d, want 20 (no eviction)", st.Tokens)
		}
		if st.Evictions != 0 {
			t.Errorf("keep-all: Evictions=%d, want 0 (eviction disabled)", st.Evictions)
		}
	})

	// Regression: the legacy New() dial is byte-identical to before the signed field
	// existed -- New(0) is unbounded, New(N) evicts to N, and Stats.MaxTokens is unchanged.
	t.Run("legacy_default_unchanged", func(t *testing.T) {
		unbounded := New(0)
		for i := 0; i < 5; i++ {
			insertUnleased(unbounded, seqTokens(i, 4))
		}
		st := unbounded.Stats()
		if st.Tokens != 20 || st.Evictions != 0 || st.MaxTokens != 0 {
			t.Errorf("New(0): Tokens=%d Evictions=%d MaxTokens=%d, want 20/0/0 (unbounded)",
				st.Tokens, st.Evictions, st.MaxTokens)
		}

		bounded := New(8)
		for i := 0; i < 5; i++ {
			insertUnleased(bounded, seqTokens(i, 4))
		}
		bst := bounded.Stats()
		if bst.Tokens > 8 || bst.Evictions == 0 || bst.MaxTokens != 8 {
			t.Errorf("New(8): Tokens=%d Evictions=%d MaxTokens=%d, want <=8 / >0 / 8",
				bst.Tokens, bst.Evictions, bst.MaxTokens)
		}
	})
}

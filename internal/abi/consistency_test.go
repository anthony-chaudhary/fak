package abi

import "testing"

// TestConsistencyRoundTrip witnesses the closed vocabulary: every level renders to its
// token and parses back, and the four tokens are exactly the declared set.
func TestConsistencyRoundTrip(t *testing.T) {
	cases := []struct {
		lvl   ConsistencyLevel
		token string
	}{
		{ConsistencyStrict, "STRICT"},
		{ConsistencyBoundedStale, "BOUNDED_STALE"},
		{ConsistencyBestEffort, "BEST_EFFORT"},
		{ConsistencySpeculative, "SPECULATIVE"},
	}
	for _, c := range cases {
		if got := c.lvl.String(); got != c.token {
			t.Errorf("%d.String() = %q, want %q", c.lvl, got, c.token)
		}
		got, ok := ParseConsistency(c.token)
		if !ok || got != c.lvl {
			t.Errorf("ParseConsistency(%q) = (%d,%v), want (%d,true)", c.token, got, ok, c.lvl)
		}
	}
}

// TestConsistencyDefaultsStrict is the load-bearing default: an absent, empty, or
// unrecognized value resolves to STRICT — the fail-safe strictest contract — so the
// field is purely additive and a non-aware caller is never silently relaxed.
func TestConsistencyDefaultsStrict(t *testing.T) {
	if got := ConsistencyOf(nil); got != ConsistencyStrict {
		t.Errorf("ConsistencyOf(nil) = %v, want STRICT", got)
	}
	if got := ConsistencyOf(&ToolCall{}); got != ConsistencyStrict {
		t.Errorf("ConsistencyOf(no meta) = %v, want STRICT", got)
	}
	if got := ConsistencyOf(&ToolCall{Meta: map[string]string{MetaConsistency: ""}}); got != ConsistencyStrict {
		t.Errorf("ConsistencyOf(empty) = %v, want STRICT", got)
	}
	if got := ConsistencyOf(&ToolCall{Meta: map[string]string{MetaConsistency: "garbage"}}); got != ConsistencyStrict {
		t.Errorf("ConsistencyOf(unknown) = %v, want STRICT (fail-safe)", got)
	}
	if _, ok := ParseConsistency("garbage"); ok {
		t.Errorf("ParseConsistency(garbage) ok=true, want false (so a validator can tell explicit-STRICT from fallback)")
	}
	// A set level is read back, case-insensitively.
	if got := ConsistencyOf(&ToolCall{Meta: map[string]string{MetaConsistency: "best_effort"}}); got != ConsistencyBestEffort {
		t.Errorf("ConsistencyOf(best_effort) = %v, want BEST_EFFORT", got)
	}
}

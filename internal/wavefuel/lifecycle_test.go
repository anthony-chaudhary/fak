package wavefuel

import (
	"testing"
)

// Invariant: Wave fuel string parsing must reliably extract bounded substrings between declared delimiters.
// Guard: between returns empty strings when either start or end delimiters are missing.

func TestWaveFuelLifecycle(t *testing.T) {
	t.Parallel()

	s := "prefix ## Phase 1 — Start content ## Phase 2 — End suffix"
	sub := between(s, "## Phase 1 —", "## Phase 2 —")
	if sub != "## Phase 1 — Start content " {
		t.Fatalf("unexpected extracted substring: %q", sub)
	}

	if missing := between(s, "missing", "End"); missing != "" {
		t.Fatalf("expected empty substring on missing delimiter, got %q", missing)
	}
}

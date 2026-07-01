package relay

import "testing"

// Issue #1878 done condition: a diverged baton yields RELAY_BATON_STALE with the culprit
// claim; a fresh baton does not. These are that witness (run: `go test ./internal/relay
// -run BatonStale`). fakeResolver is shared from reload_test.go (same package).

// TestBatonStaleOnDivergedCursor asserts the core outcome: when the cursor's anchor no
// longer resolves in git, CheckBatonStale emits RELAY_BATON_STALE naming the culprit
// anchor and carrying git evidence.
func TestBatonStaleOnDivergedCursor(t *testing.T) {
	const anchor = "0123456789abcdef0123456789abcdef01234567"
	b := Baton{Schema: Schema, RelayID: "RLY-1878", ProgressCursor: ProgressCursor{StartSHA: anchor}}
	diverged := fakeResolver{verified: map[string]bool{}} // anchor no longer resolves

	got := CheckBatonStale(b, diverged)
	if !got.Stale {
		t.Fatalf("a diverged baton must be stale: %+v", got)
	}
	if got.Reason != ReasonBatonStale {
		t.Errorf("reason = %q, want %s", got.Reason, ReasonBatonStale)
	}
	if got.Culprit != "start_sha" {
		t.Errorf("culprit = %q, want start_sha (the deciding claim)", got.Culprit)
	}
	if got.Evidence == "" {
		t.Error("a stale outcome must carry git evidence")
	}
}

// TestBatonStaleFreshCursorIsNotStale pins the negative: a cursor whose anchor still
// resolves yields the zero outcome — not stale, no reason token.
func TestBatonStaleFreshCursorIsNotStale(t *testing.T) {
	const anchor = "0123456789abcdef0123456789abcdef01234567"
	b := Baton{Schema: Schema, RelayID: "RLY-1878", ProgressCursor: ProgressCursor{StartSHA: anchor}}
	matching := fakeResolver{verified: map[string]bool{anchor: true}}

	got := CheckBatonStale(b, matching)
	if got.Stale {
		t.Errorf("a fresh baton must not be stale: %+v", got)
	}
	if got.Reason != "" {
		t.Errorf("a non-stale outcome must carry no reason token, got %q", got.Reason)
	}
}

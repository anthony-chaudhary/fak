package relay

import (
	"encoding/json"
	"testing"
)

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

// TestTombstoneRotRederivesFromDurableStore is the #1904 adversarial witness: the
// cursor still resolves, but the tombstone header names a commit that no longer exists.
// Reload must flag RELAY_BATON_STALE on tombstone.at_sha and still re-read durable
// artifact pointers from the store instead of trusting the rotten handoff note.
func TestTombstoneRotRederivesFromDurableStore(t *testing.T) {
	const cursorSHA = "1111111111111111111111111111111111111111"
	const rottenSHA = "2222222222222222222222222222222222222222"
	const artifactSHA = "3333333333333333333333333333333333333333"
	b := Baton{
		Schema:  Schema,
		RelayID: "RLY-1904",
		ProgressCursor: ProgressCursor{
			StartSHA:   cursorSHA,
			HeldRegion: []string{"internal/relay/**"},
		},
		Artifacts: []Artifact{
			{Kind: string(ArtifactCommit), Ref: artifactSHA},
		},
		Tombstone: Tombstone{
			Reason: "RELAY_ROTATED",
			AtSHA:  rottenSHA,
			Note:   "display-only note from a stale handoff",
		},
	}
	store := fakeResolver{verified: map[string]bool{
		cursorSHA:   true,
		artifactSHA: true,
	}}

	got := CheckBatonStale(b, store)
	if !got.Stale {
		t.Fatalf("tombstone rot must be stale: %+v", got)
	}
	if got.Reason != ReasonBatonStale {
		t.Errorf("reason = %q, want %s", got.Reason, ReasonBatonStale)
	}
	if got.Culprit != "tombstone.at_sha" {
		t.Errorf("culprit = %q, want tombstone.at_sha", got.Culprit)
	}
	if got.Evidence == "" {
		t.Fatal("tombstone-rot stale outcome must carry git evidence")
	}

	var plan reloadPlan
	if err := json.Unmarshal(projectReload(t, b, store), &plan); err != nil {
		t.Fatalf("unmarshal reload plan: %v", err)
	}
	if plan.Cursor.Verdict != ReloadFresh {
		t.Fatalf("cursor should still re-verify fresh; verdict = %q reason=%s", plan.Cursor.Verdict, plan.Cursor.Reason)
	}
	if !plan.Stale.Stale || plan.Stale.Culprit != "tombstone.at_sha" {
		t.Fatalf("reload plan stale outcome = %+v, want tombstone.at_sha stale", plan.Stale)
	}
	if len(plan.Resolutions) != 1 {
		t.Fatalf("reload plan resolutions = %d, want 1 durable artifact read", len(plan.Resolutions))
	}
	if plan.Resolutions[0].Artifact.Ref != artifactSHA || plan.Resolutions[0].Verdict != ResolveVerified {
		t.Fatalf("durable artifact resolution = %+v, want verified %s", plan.Resolutions[0], artifactSHA)
	}
}

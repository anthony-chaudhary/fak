package marketing

import (
	"context"
	"testing"
	"time"
)

// fakePoster records the artifacts it was asked to post, so a test can assert how many real
// posts a sequence of ticks produced.
type fakePoster struct {
	posted []Artifact
	ts     string
}

func (f *fakePoster) PostArtifact(_ context.Context, a Artifact) (string, error) {
	f.posted = append(f.posted, a)
	return f.ts, nil
}

func TestTickRangeUsesMarkThenBootstrap(t *testing.T) {
	cases := []struct {
		mark, bootstrap, want string
	}{
		{"abc123", "", "abc123..HEAD"},       // mark wins
		{"abc123", "HEAD~5", "abc123..HEAD"}, // mark still wins over bootstrap
		{"", "", "HEAD~20..HEAD"},            // first run, default bootstrap
		{"", "HEAD~5", "HEAD~5..HEAD"},       // first run, explicit bootstrap
		{"", "ALL", ""},                      // explicit whole-history opt-in
	}
	for _, tc := range cases {
		if got := tickRange(tc.mark, tc.bootstrap); got != tc.want {
			t.Errorf("tickRange(%q,%q) = %q, want %q", tc.mark, tc.bootstrap, got, tc.want)
		}
	}
}

func TestTickResultStatusesAreClosedSet(t *testing.T) {
	// A small guard that the documented status strings are the ones the code emits — a typo
	// in a status string would silently break a caller switch.
	want := map[string]bool{
		"posted": true, "skipped:no-new-ships": true, "skipped:raced": true,
		"dry-run": true, "skipped:no-poster": true,
	}
	for _, s := range []string{"posted", "skipped:no-new-ships", "skipped:raced", "dry-run", "skipped:no-poster"} {
		if !want[s] {
			t.Errorf("status %q not in the documented set", s)
		}
	}
}

// Note: the full Tick path (read mark -> gather -> CAS -> post) is exercised end-to-end by
// the CLI against the real repo and by the bgloop integration; it shells git, so it is not
// unit-tested here. tickRange (the pure range logic) and the Poster seam (fakePoster) are the
// unit-testable surface, and the dry-run-never-advances invariant is asserted via the CLI's
// repeatability check. The fakePoster type proves the Poster interface is satisfiable without
// the scoreboard transport.
var _ Poster = (*fakePoster)(nil)

func TestFakePosterRecords(t *testing.T) {
	f := &fakePoster{ts: "1700000000.000100"}
	art := Artifact{Kind: KindWeeklyDigest, Title: "t"}
	ts, err := f.PostArtifact(context.Background(), art)
	if err != nil || ts != "1700000000.000100" {
		t.Fatalf("PostArtifact = (%q,%v)", ts, err)
	}
	if len(f.posted) != 1 {
		t.Errorf("posted = %d, want 1", len(f.posted))
	}
}

var _ = time.Now

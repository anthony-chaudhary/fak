package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

// TestSelfUpdateShouldBuild pins the proceed decision, and in particular the case binstamp
// alone gets WRONG: a clean local binary that is AHEAD of origin/main. Under the old
// `verdict == binstamp.Stale` rule that case (rev differs => Stale) rebuilt origin/main OVER
// the newer binary; keying SELF mode off versionskew.Skewed makes Ahead a no-op. This is the
// "previously-collapsed case now drives a distinct decision" the wiring exists to produce.
func TestSelfUpdateShouldBuild(t *testing.T) {
	cases := []struct {
		name  string
		force bool
		fleet bool
		bin   binstamp.Freshness
		skew  versionskew.Verdict
		want  bool
	}{
		// SELF mode: ONLY a provably-behind skew rebuilds.
		{"self behind rebuilds", false, false, binstamp.Stale, versionskew.Skewed, true},
		{"self ahead does NOT rebuild (the fix)", false, false, binstamp.Stale, versionskew.Ahead, false},
		{"self diverged does NOT rebuild", false, false, binstamp.Stale, versionskew.Diverged, false},
		{"self fresh no-op", false, false, binstamp.Fresh, versionskew.Fresh, false},
		{"self dirty no-op", false, false, binstamp.Unknown, versionskew.Dirty, false},
		{"self unstamped no-op", false, false, binstamp.Unknown, versionskew.Unstamped, false},
		{"self unknown no-op", false, false, binstamp.Unknown, versionskew.Unknown, false},
		{"self force overrides a fresh binary", true, false, binstamp.Fresh, versionskew.Fresh, true},
		// FLEET mode: rebuild unless binstamp proves Fresh — regardless of the skew token.
		{"fleet not-fresh rebuilds", false, true, binstamp.Unknown, versionskew.Unknown, true},
		{"fleet behind rebuilds", false, true, binstamp.Stale, versionskew.Skewed, true},
		{"fleet fresh no-op", false, true, binstamp.Fresh, versionskew.Fresh, false},
		{"fleet fresh + force rebuilds", true, true, binstamp.Fresh, versionskew.Fresh, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := selfUpdateShouldBuild(c.force, c.fleet, c.bin, c.skew); got != c.want {
				t.Fatalf("selfUpdateShouldBuild(force=%v fleet=%v bin=%v skew=%v) = %v, want %v",
					c.force, c.fleet, c.bin, c.skew, got, c.want)
			}
		})
	}
}

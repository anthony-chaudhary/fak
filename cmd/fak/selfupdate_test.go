package main

import (
	"os"
	"path/filepath"
	"strings"
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

// TestSelfUpdateSiblingsIncludesInTreeFleetBinary pins the fix for the stale-fleet-binary lag:
// `self-update --target X` converged X and nothing else, while every dispatcher-launched worker
// runs `<root>/tools/.bin/fak[.exe] guard -- claude …` — the path
// tools/dispatch_worker.py resolve_fak_bin prefers AHEAD of PATH. Because that in-tree file
// existed, PATH was never consulted, so the fleet ran a binary no updater targeted and the tick
// still exited 0. The sibling set must therefore contain the in-tree fleet binary.
func TestSelfUpdateSiblingsIncludesInTreeFleetBinary(t *testing.T) {
	root := t.TempDir()
	binDir := filepath.Join(root, "tools", ".bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	fleetBin := filepath.Join(binDir, "fak"+exeSuffix())
	if err := os.WriteFile(fleetBin, []byte("stale"), 0o755); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "installed"+exeSuffix())
	if err := os.WriteFile(target, []byte("fresh"), 0o755); err != nil {
		t.Fatal(err)
	}

	got := selfUpdateSiblings(root, target)
	found := false
	for _, p := range got {
		if strings.EqualFold(p, fleetBin) {
			found = true
		}
		if strings.EqualFold(p, target) {
			t.Errorf("sibling set must not repeat the primary --target %q: %v", target, got)
		}
	}
	if !found {
		t.Errorf("selfUpdateSiblings(%q, %q) = %v; want it to include the in-tree fleet binary %q",
			root, target, got, fleetBin)
	}
}

// TestSelfUpdateSiblingsSkipsMissingPaths — we converge binaries that already exist; a path that
// is absent is not an install location we should create. With no tools/.bin on disk the only
// sibling is the running test binary itself, never a phantom <root>/tools/.bin entry.
func TestSelfUpdateSiblingsSkipsMissingPaths(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "installed"+exeSuffix())
	if err := os.WriteFile(target, []byte("fresh"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range selfUpdateSiblings(root, target) {
		if strings.Contains(strings.ToLower(p), filepath.Join("tools", ".bin")) {
			t.Errorf("selfUpdateSiblings returned a non-existent path %q", p)
		}
		if st, err := os.Stat(p); err != nil || st.IsDir() {
			t.Errorf("selfUpdateSiblings returned %q which is not an existing file", p)
		}
	}
}

// TestConvergeSiblings pins that skipping requires PROOF of freshness: only a binstamp.Fresh
// invoker is left alone. An Unknown stamp — a binary that cannot self-report which commit it is —
// converges, because that is exactly the un-attestable fleet binary the lag hid.
func TestConvergeSiblings(t *testing.T) {
	head := "2c52df490c53d5689fe6f42dfb829d27b5f160bb"
	cases := []struct {
		name  string
		stamp binstamp.Stamp
		want  bool
	}{
		{"fresh invoker is left alone", binstamp.Stamp{Revision: head, HasVCS: true}, false},
		{"stale invoker converges", binstamp.Stamp{Revision: "b225bb1ca20f0000000000000000000000000000", HasVCS: true}, true},
		{"unstamped invoker converges", binstamp.Stamp{}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := convergeSiblings(c.stamp, head); got != c.want {
				t.Errorf("convergeSiblings(%+v, head) = %v; want %v", c.stamp, got, c.want)
			}
		})
	}
}

// TestSelfUpdateSkipOutcome pins the closed outcome vocabulary against the message switch it
// mirrors. The scheduler sees only an exit code, and rc=0 is identical for "installed",
// "already current", "busy" and "--check"; the named outcome is what re-couples the success
// code to whether an update actually happened.
func TestSelfUpdateSkipOutcome(t *testing.T) {
	cases := []struct {
		fleet bool
		skew  versionskew.Verdict
		want  selfUpdateOutcome
	}{
		{true, versionskew.Fresh, outcomeTargetCurrent},
		{true, versionskew.Unknown, outcomeTargetCurrent},
		{false, versionskew.Fresh, outcomeSelfFresh},
		{false, versionskew.Ahead, outcomeSelfAhead},
		{false, versionskew.Dirty, outcomeSelfLocal},
		{false, versionskew.Unstamped, outcomeSelfLocal},
		{false, versionskew.Diverged, outcomeSelfLocal},
		{false, versionskew.Unknown, outcomeSelfUnknown},
	}
	for _, c := range cases {
		if got := selfUpdateSkipOutcome(c.fleet, c.skew); got != c.want {
			t.Errorf("selfUpdateSkipOutcome(fleet=%v, %v) = %q; want %q", c.fleet, c.skew, got, c.want)
		}
	}
}

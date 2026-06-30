package opttarget

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// matchingTarget returns a valid OptTarget whose Site IS the worktree harness's
// hard-wired tunable, so the worktree-int factory's honesty guard is satisfied.
// It is hermetic: Resolve only BUILDS the measurer (no git, no worktree, no run).
func matchingTarget() OptTarget {
	return OptTarget{
		Name:        "lru",
		Metric:      "lru_hit_rate",
		Direction:   HigherBetter,
		BaselineRef: "main",
		Site:        Site{Path: worktreeIntSitePath, Const: rsiloop.TunableConstName},
		Grammar:     Grammar{Kind: GrammarIntSweep, Ints: []int{4, 6, 8}},
		Measurer:    "worktree-int",
	}
}

func TestResolveMatchingTargetBuildsMeasurer(t *testing.T) {
	m, err := Resolve(matchingTarget(), "/some/repo")
	if err != nil {
		t.Fatalf("Resolve(matching) returned error: %v", err)
	}
	if m == nil {
		t.Fatal("Resolve(matching) returned a nil Measurer")
	}
}

func TestResolveSiteMismatchRefused(t *testing.T) {
	tgt := matchingTarget()
	tgt.Site.Path = "internal/other/x.go"
	_, err := Resolve(tgt, "/some/repo")
	if err == nil {
		t.Fatal("Resolve with mismatched Site.Path should error, got nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, worktreeIntSitePath) && !strings.Contains(msg, "Phase 0.1") {
		t.Fatalf("Site-mismatch error should mention the tunable path or Phase 0.1, got: %q", msg)
	}
}

func TestResolveUnknownMeasurerRefused(t *testing.T) {
	tgt := matchingTarget()
	tgt.Measurer = "nope"
	_, err := Resolve(tgt, "/some/repo")
	if err == nil {
		t.Fatal("Resolve with unknown measurer should error, got nil")
	}
	if !strings.Contains(err.Error(), "nope") {
		t.Fatalf("unknown-measurer error should mention the bad key, got: %q", err.Error())
	}
}

func TestKnownMeasurersContainsWorktreeInt(t *testing.T) {
	known := KnownMeasurers()
	found := false
	for _, k := range known {
		if k == "worktree-int" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("KnownMeasurers() should contain %q, got %v", "worktree-int", known)
	}
}

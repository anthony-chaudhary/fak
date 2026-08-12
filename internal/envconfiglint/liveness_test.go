package envconfiglint

import (
	"strings"
	"testing"
)

// TestLivenessCore pins the liveness judgment on synthetic ages — no git, no tree, the
// package's own "verify the verifier" idiom. This is the only way to exercise the RED path
// at all once the trunk is green, and the red path is the whole point: a liveness check that
// can only be observed passing is exactly the unwatched silence it exists to end.
func TestLivenessCore(t *testing.T) {
	fresh := Offense{Name: "FAK_NEW_KNOB", File: "cmd/fak/new.go"}
	stale := Offense{Name: "FAK_OLD_KNOB", File: "internal/old/old.go"}
	ages := func(m map[string]int) AdvancesSince {
		return func(o Offense) int { return m[o.Name] }
	}

	// A clean tree is watched, and says so with a count of zero.
	if v := ClassifyLiveness(nil, ages(nil)); !v.Watched() || v.Offenses != 0 || v.Advances() != 0 {
		t.Fatalf("clean tree: %+v, want watched with 0 offenses", v)
	}

	// A read introduced by HEAD itself is the gate WORKING, not the gate failing: the author
	// reds their own run. Same at exactly the tolerance — one advance is the grace period.
	for _, age := range []int{0, UnwatchedTolerance} {
		v := ClassifyLiveness([]Offense{fresh}, ages(map[string]int{"FAK_NEW_KNOB": age}))
		if !v.Watched() {
			t.Errorf("age %d: %+v, want watched (a fresh red is the ratchet doing its job)", age, v)
		}
		if v.Offenses != 1 {
			t.Errorf("age %d: Offenses = %d, want 1 (a watched verdict still counts the debt)", age, v.Offenses)
		}
	}

	// One advance past the tolerance and the ratchet is no longer gating: the read landed on
	// top of a red somebody already had the chance to see.
	v := ClassifyLiveness([]Offense{fresh}, ages(map[string]int{"FAK_NEW_KNOB": UnwatchedTolerance + 1}))
	if v.Watched() || len(v.Unwatched) != 1 || v.Advances() != UnwatchedTolerance+1 {
		t.Fatalf("age %d: %+v, want exactly one unwatched read", UnwatchedTolerance+1, v)
	}

	// The verdict leads with the OLDEST read — the age is the finding, and the worst age is
	// the one that measures how long the gate was dead.
	v = ClassifyLiveness([]Offense{fresh, stale}, ages(map[string]int{"FAK_NEW_KNOB": 2, "FAK_OLD_KNOB": 435}))
	if len(v.Unwatched) != 2 || v.Advances() != 435 || v.Unwatched[0].Name != "FAK_OLD_KNOB" {
		t.Fatalf("mixed ages: %+v, want both unwatched, oldest (FAK_OLD_KNOB, 435) first", v)
	}
	msg := v.String()
	for _, want := range []string{ReasonRatchetUnwatched, "435", "FAK_OLD_KNOB", "internal/old/old.go", "2 of 2"} {
		if !strings.Contains(msg, want) {
			t.Errorf("verdict %q must name %q", msg, want)
		}
	}

	// An UNKNOWN age (a rename, a squash, a shallow clone) is never treated as unwatched:
	// fewer refusals is the safe direction, the same one IsSecretName takes. A nil lookup is
	// the degenerate case of that — it must not manufacture a red out of missing evidence.
	if v := ClassifyLiveness([]Offense{stale}, ages(map[string]int{"FAK_OLD_KNOB": -1})); !v.Watched() {
		t.Errorf("unknown age: %+v, want watched (an unlocatable introduction is not evidence)", v)
	}
	if v := ClassifyLiveness([]Offense{stale}, nil); !v.Watched() {
		t.Errorf("nil lookup: %+v, want watched", v)
	}
}

// TestRatchetIsStillGating is the LIVE liveness gate — the check doc.go asked for and this
// issue (#6215) finally builds. TestNoNewNonSecretEnvReads reds when the tree is dirty; this
// reds when the ratchet ITSELF has stopped enforcing, i.e. when a non-secret read has been
// outstanding for more than one trunk advance. The two failures look identical to a rule
// that can only count offenses, and they call for opposite responses: fix the read, versus
// fix the process that let 435 commits ship past a gate that was already failing.
func TestRatchetIsStillGating(t *testing.T) {
	root := repoRoot(t)
	v, err := TreeLiveness(root)
	if err != nil {
		t.Skipf("git unavailable (%v); the liveness gate needs a git checkout", err)
	}
	if !v.Watched() {
		t.Error(v.String())
	}
}

// TestLivenessAgeLookupIsNotBlind is the negative control for the gate above, mirroring
// TestTreeScannerIsNotVacuous. TreeLiveness is green in two very different ways: because no
// read is stale, or because the pickaxe silently returns nothing and every age comes back
// unknown. The second is a dead check that would pass forever, so prove the age lookup can
// actually date a read the tree really carries.
func TestLivenessAgeLookupIsNotBlind(t *testing.T) {
	root := repoRoot(t)
	all, err := scanTree(root, nil)
	if err != nil {
		t.Skipf("git grep unavailable (%v); the liveness gate needs a git checkout", err)
	}
	if len(all) == 0 {
		t.Skip("tree carries no non-secret env reads; nothing to date")
	}
	// The victim is derived from the scan, never hardcoded, so this cannot rot against the tree.
	victim := all[0]
	if age := GitAdvancesSince(root)(victim); age < 0 {
		t.Fatalf("could not date %s (%s); the pickaxe lookup is blind, which would make the "+
			"liveness gate pass vacuously forever", victim.Name, victim.File)
	}
}

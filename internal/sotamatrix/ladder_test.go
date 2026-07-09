package sotamatrix

import (
	"strings"
	"testing"
)

// TestLaddersWellFormed enforces the structural invariants that keep the milestone
// ladders honest: contiguous rungs from 0, every rung fully populated with a real
// reference, unique kebab-case axes, and a conservative in-range FakRung.
func TestLaddersWellFormed(t *testing.T) {
	all := Ladders()
	if len(all) == 0 {
		t.Fatal("no ladders defined")
	}

	seen := map[string]bool{}
	for _, l := range all {
		if l.Axis == "" {
			t.Errorf("ladder %q: empty axis", l.Title)
		}
		if seen[l.Axis] {
			t.Errorf("duplicate axis %q", l.Axis)
		}
		seen[l.Axis] = true
		if l.Axis != strings.ToLower(l.Axis) || strings.ContainsAny(l.Axis, " _") {
			t.Errorf("axis %q is not lower-kebab-case", l.Axis)
		}
		if strings.TrimSpace(l.Title) == "" {
			t.Errorf("ladder %q: empty title", l.Axis)
		}
		if strings.TrimSpace(l.Summary) == "" {
			t.Errorf("ladder %q: empty summary", l.Axis)
		}
		if len(l.Rungs) < 2 {
			t.Errorf("ladder %q: a ladder needs >=2 rungs, got %d", l.Axis, len(l.Rungs))
		}

		// Rungs are contiguous from 0 (0,1,2,…) so FakRung+1 is "the next rung".
		for i, r := range l.Rungs {
			if r.Level != i {
				t.Errorf("ladder %q rung %d: Level=%d, want contiguous %d", l.Axis, i, r.Level, i)
			}
			if strings.TrimSpace(r.Name) == "" {
				t.Errorf("ladder %q rung %d: empty Name", l.Axis, i)
			}
			if strings.TrimSpace(r.Year) == "" {
				t.Errorf("ladder %q rung %q: empty Year", l.Axis, r.Name)
			}
			if strings.TrimSpace(r.Ref) == "" {
				t.Errorf("ladder %q rung %q: empty Ref (a rung must cite real prior art)", l.Axis, r.Name)
			}
			if strings.TrimSpace(r.Adds) == "" {
				t.Errorf("ladder %q rung %q: empty Adds", l.Axis, r.Name)
			}
		}

		// FakRung is -1 (not implemented) or a real rung index; never past the top.
		top := len(l.Rungs) - 1
		if l.FakRung < -1 || l.FakRung > top {
			t.Errorf("ladder %q: FakRung=%d out of range [-1,%d]", l.Axis, l.FakRung, top)
		}
	}
}

// TestLadderOpSlugResolves is the ladder's honesty anchor: any ladder that claims
// to map to a kernel Op MUST name a slug that exists in the matrix, so a ladder
// cannot cite a fak position that the tree-verified matrix does not back.
func TestLadderOpSlugResolves(t *testing.T) {
	for _, l := range Ladders() {
		if l.OpSlug == "" {
			continue // serving-level axis with no single kernel row — allowed.
		}
		if _, ok := BySlug(l.OpSlug); !ok {
			t.Errorf("ladder %q maps to OpSlug %q, which is not in the matrix", l.Axis, l.OpSlug)
		}
		// The bridge must be walkable both ways.
		found := false
		for _, back := range LaddersForOp(l.OpSlug) {
			if back.Axis == l.Axis {
				found = true
			}
		}
		if !found {
			t.Errorf("LaddersForOp(%q) did not return ladder %q", l.OpSlug, l.Axis)
		}
	}
}

// TestNextRung checks the "next baseline milestone to target" helper: it returns
// the rung above FakRung, and reports none when fak is on the top rung or the axis
// is not implemented.
func TestNextRung(t *testing.T) {
	for _, l := range Ladders() {
		next, ok := l.NextRung()
		switch {
		case l.FakRung < 0:
			if ok {
				t.Errorf("ladder %q: FakRung<0 but NextRung returned %q", l.Axis, next.Name)
			}
		case l.FakRung == len(l.Rungs)-1:
			if ok {
				t.Errorf("ladder %q: on top rung but NextRung returned %q", l.Axis, next.Name)
			}
		default:
			if !ok {
				t.Errorf("ladder %q: FakRung=%d has a rung above but NextRung reported none", l.Axis, l.FakRung)
			} else if next.Level != l.FakRung+1 {
				t.Errorf("ladder %q: NextRung Level=%d, want %d", l.Axis, next.Level, l.FakRung+1)
			}
		}
	}
}

// TestLadderCopyIsolation confirms accessors hand back copies — mutating a returned
// ladder's rungs must not corrupt the source (the matrix rows already guarantee
// this; ladders carry a nested slice, so it is worth pinning).
func TestLadderCopyIsolation(t *testing.T) {
	l, ok := LadderByAxis("attention")
	if !ok {
		t.Fatal("attention ladder missing")
	}
	if len(l.Rungs) == 0 {
		t.Fatal("attention ladder has no rungs")
	}
	l.Rungs[0].Name = "MUTATED"
	again, _ := LadderByAxis("attention")
	if again.Rungs[0].Name == "MUTATED" {
		t.Error("LadderByAxis returned a shared Rungs slice; mutation leaked into the source")
	}
}

// TestNamedAxesPresent pins the axes the milestone goal explicitly named
// (attention / batching) plus the obvious neighbors, so a future edit cannot
// quietly drop one.
func TestNamedAxesPresent(t *testing.T) {
	for _, axis := range []string{"attention", "batching", "quantization", "kv-cache", "speculative-decoding"} {
		if _, ok := LadderByAxis(axis); !ok {
			t.Errorf("expected a milestone ladder for axis %q", axis)
		}
	}
}

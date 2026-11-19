package policygoal

import (
	"encoding/json"
	"testing"
)

// TestAllGoalsClosed verifies the closed vocabulary is exactly the eight
// named goals, in canonical order.
func TestAllGoalsClosed(t *testing.T) {
	want := []Goal{
		GoalBatch, GoalInteractive, GoalReliability, GoalCapacity,
		GoalP50, GoalP99, GoalPrefillDecode, GoalContextLength,
	}
	if len(AllGoals) != len(want) {
		t.Fatalf("AllGoals len = %d, want %d", len(AllGoals), len(want))
	}
	for i, g := range AllGoals {
		if g != want[i] {
			t.Errorf("AllGoals[%d] = %q, want %q", i, g, want[i])
		}
	}
}

// TestInventoryDeterministic verifies the "run twice -> identical" witness:
// marshaling the inventory twice yields byte-identical output.
func TestInventoryDeterministic(t *testing.T) {
	b1, err := json.Marshal(DefaultInventory)
	if err != nil {
		t.Fatalf("Marshal #1: %v", err)
	}
	b2, err := json.Marshal(DefaultInventory)
	if err != nil {
		t.Fatalf("Marshal #2: %v", err)
	}
	if string(b1) != string(b2) {
		t.Fatalf("marshal not deterministic:\n%s\nvs\n%s", b1, b2)
	}
}

// TestInventoryComplete verifies every named goal has at least one knob, so
// the inventory is not vacuous for any goal.
func TestInventoryComplete(t *testing.T) {
	counts := DefaultInventory.CountByGoal()
	for _, g := range AllGoals {
		if counts[g] == 0 {
			t.Errorf("goal %q has no knobs; inventory is incomplete", g)
		}
	}
}

// TestGoalsFor verifies the query API returns only knobs that name the goal.
func TestGoalsFor(t *testing.T) {
	got := DefaultInventory.GoalsFor(GoalReliability)
	if len(got) == 0 {
		t.Fatalf("GoalsFor(reliability) returned no knobs")
	}
	for _, k := range got {
		found := false
		for _, g := range k.Goals {
			if g == GoalReliability {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("GoalsFor(reliability) returned knob %q which does not name reliability", k.Name)
		}
	}
}

// TestHotAndDefectHiding verifies the hot/cold and defect-hiding partitions
// are non-empty and mutually consistent.
func TestHotAndDefectHiding(t *testing.T) {
	hot := DefaultInventory.Hot()
	if len(hot) == 0 {
		t.Errorf("Hot() returned no knobs; expected at least one live-changeable knob")
	}
	for _, k := range hot {
		if k.Hotness != Hot {
			t.Errorf("Hot() returned knob %q with hotness %q", k.Name, k.Hotness)
		}
	}
	dh := DefaultInventory.DefectHiding()
	if len(dh) == 0 {
		t.Errorf("DefectHiding() returned no knobs; expected at least one defect-hiding knob")
	}
	for _, k := range dh {
		if !k.DefectHiding {
			t.Errorf("DefectHiding() returned knob %q which is not defect-hiding", k.Name)
		}
	}
}

// TestSortStable verifies Sort returns a deterministic ordering by Name.
func TestSortStable(t *testing.T) {
	s1 := DefaultInventory.Sort()
	s2 := DefaultInventory.Sort()
	if len(s1.Knobs) != len(s2.Knobs) {
		t.Fatalf("Sort length mismatch: %d vs %d", len(s1.Knobs), len(s2.Knobs))
	}
	for i := range s1.Knobs {
		if s1.Knobs[i].Name != s2.Knobs[i].Name {
			t.Errorf("Sort not stable at %d: %q vs %q", i, s1.Knobs[i].Name, s2.Knobs[i].Name)
		}
	}
	for i := 1; i < len(s1.Knobs); i++ {
		if s1.Knobs[i-1].Name > s1.Knobs[i].Name {
			t.Errorf("Sort not ascending at %d: %q > %q", i, s1.Knobs[i-1].Name, s1.Knobs[i].Name)
		}
	}
}

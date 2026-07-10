package milestonereport

import "testing"

func TestTrackedEpicsIncludesMLPMilestoneProgram(t *testing.T) {
	for _, epic := range TrackedEpics {
		if epic.Number == 3256 {
			if epic.Generation != "now" {
				t.Fatalf("epic #3256 generation = %q, want now", epic.Generation)
			}
			return
		}
	}
	t.Fatal("milestone report does not track MLP epic #3256")
}

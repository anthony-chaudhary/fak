package agent

import "testing"

// fak-test:runtime fast est=2ms lane=default
func TestEffortBudgetTrainedLoopsIrreversibleFloor(t *testing.T) {
	cases := []struct {
		name         string
		granted, tr  int
		irreversible bool
		want         int
	}{
		{"irreversible lifts a low grant to trained", 1, 3, true, 3},
		{"reversible keeps the grant", 1, 3, false, 1},
		{"unknown trained count keeps the grant", 1, 0, true, 1},
		{"grant already above trained is kept", 5, 3, true, 5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveIrreversibleLoopFloor(tc.granted, tc.tr, tc.irreversible); got != tc.want {
				t.Fatalf("ResolveIrreversibleLoopFloor(%d,%d,%v) = %d, want %d", tc.granted, tc.tr, tc.irreversible, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestLoopDepthTrainedLoopsEffortNeverBelowFloor(t *testing.T) {
	for _, effort := range []string{"none", "low", "medium", "high", "ultra"} {
		if got := ResolveTrainedLoopDepth(effort, 8, 3); got < 3 {
			t.Fatalf("effort %s with trained 3 resolved to %d", effort, got)
		}
	}
}

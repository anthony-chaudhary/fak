package agent

import "testing"

func TestResolveLoopDepth(t *testing.T) {
	errTurn := TurnAssessment{IsErrorRecovery: true}
	routine := TurnAssessment{ToolName: "read"}
	cases := []struct {
		name     string
		profile  string
		maxLoops int
		turns    []TurnAssessment
		want     int
	}{
		{"non-looped zero", EffortTierUltra, 0, nil, 1},
		{"non-looped one", EffortTierUltra, 1, []TurnAssessment{errTurn}, 1},
		{"negative max", EffortTierHigh, -3, nil, 1},
		{"none floors to one", EffortTierNone, 4, nil, 1},
		{"low of 4", EffortTierLow, 4, nil, 1},
		{"medium of 4", EffortTierMedium, 4, nil, 2},
		{"default profile is medium", ReasoningProfileDefault, 4, nil, 2},
		{"high of 4", EffortTierHigh, 4, nil, 3},
		{"deep-reason is high", ReasoningProfileDeepReason, 4, nil, 3},
		{"ultra of 4", EffortTierUltra, 4, nil, 4},
		{"low of 8", EffortTierLow, 8, nil, 2},
		{"medium of 8", EffortTierMedium, 8, nil, 4},
		{"high of 8", EffortTierHigh, 8, nil, 6},
		{"ultra of 8", EffortTierUltra, 8, nil, 8},
		{"low of 2 rounds up", EffortTierLow, 2, nil, 1},
		{"high of 2", EffortTierHigh, 2, nil, 2},
		{"error raises low", EffortTierLow, 4, []TurnAssessment{errTurn}, 2},
		{"error raises medium", EffortTierMedium, 8, []TurnAssessment{routine, errTurn}, 6},
		{"error raises none", EffortTierNone, 8, []TurnAssessment{errTurn}, 2},
		{"error at ultra clamps", EffortTierUltra, 4, []TurnAssessment{errTurn}, 4},
		{"error message counts", EffortTierHigh, 4, []TurnAssessment{{ErrorMessage: "exit status 1"}}, 4},
		{"routine leaves depth", EffortTierMedium, 4, []TurnAssessment{routine}, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveLoopDepth(tc.profile, tc.maxLoops, tc.turns...); got != tc.want {
				t.Fatalf("ResolveLoopDepth(%q, %d, %v) = %d, want %d", tc.profile, tc.maxLoops, tc.turns, got, tc.want)
			}
		})
	}
}

func TestResolveLoopDepthMonotone(t *testing.T) {
	tiers := []string{EffortTierNone, EffortTierLow, EffortTierMedium, EffortTierHigh, EffortTierUltra}
	for maxLoops := 1; maxLoops <= 16; maxLoops++ {
		prev := 0
		for _, tier := range tiers {
			got := ResolveLoopDepth(tier, maxLoops)
			if got < prev || got < 1 || got > maxLoops {
				t.Fatalf("maxLoops=%d tier=%s: got %d (prev %d)", maxLoops, tier, got, prev)
			}
			prev = got
		}
	}
}

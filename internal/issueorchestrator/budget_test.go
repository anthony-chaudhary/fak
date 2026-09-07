package issueorchestrator

import (
	"strings"
	"testing"
)

func TestDynamicIterationBudget_BaselineScaling(t *testing.T) {
	tests := []struct {
		name          string
		points        float64
		expectedSteps int
		wantBudget    int
	}{
		{
			name:          "Zero points, small steps (floor applies)",
			points:        0,
			expectedSteps: 3,
			wantBudget:    15, // max(15, 3*2) = 15
		},
		{
			name:          "Zero points, larger steps",
			points:        0,
			expectedSteps: 10,
			wantBudget:    20, // max(15, 10*2) = 20
		},
		{
			name:          "Zero points, exact floor boundary",
			points:        0,
			expectedSteps: 7,
			wantBudget:    15, // max(15, 7*2=14) = 15
		},
		{
			name:          "Small points below floor (floor applies)",
			points:        1.0,
			expectedSteps: 2,
			wantBudget:    15, // max(15, int(1.0*8)=8) = 15
		},
		{
			name:          "Points scaling 2.0 (exceeds floor)",
			points:        2.0,
			expectedSteps: 0,
			wantBudget:    16, // max(15, int(2.0*8)=16) = 16
		},
		{
			name:          "Points scaling 3.5",
			points:        3.5,
			expectedSteps: 0,
			wantBudget:    28, // max(15, int(3.5*8)=28) = 28
		},
		{
			name:          "Points scaling 5.0",
			points:        5.0,
			expectedSteps: 5,
			wantBudget:    40, // max(15, int(5.0*8)=40) = 40
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CalculateTurnBudget(tc.points, tc.expectedSteps)
			if got != tc.wantBudget {
				t.Errorf("CalculateTurnBudget(%v, %d) = %d; want %d", tc.points, tc.expectedSteps, got, tc.wantBudget)
			}
		})
	}
}

func TestDynamicIterationBudget_SoftExtensionOnProgress(t *testing.T) {
	monitor := &BudgetMonitor{
		BaselineBudget: 15,
		CurrentBudget:  15,
		IssueNumber:    12017,
		MaxCap:         50,
	}

	// Normal progress across turns 1..15 within baseline budget.
	for turn := 1; turn <= 15; turn++ {
		v := monitor.RecordTurn(turn, "normal progress step", 1, nil)
		if v != VerdictContinue {
			t.Fatalf("turn %d: expected VERDICT_CONTINUE, got %v", turn, v)
		}
		if monitor.CurrentBudget != 15 {
			t.Fatalf("turn %d: expected CurrentBudget 15, got %d", turn, monitor.CurrentBudget)
		}
	}

	// Turn 16: overrun with witnessed progress (file edit & milestone tag).
	output := "completed core implementation\n<!-- fak:progress milestone=\"core_impl\" delta=\"+2 files\" tests=\"pass\" -->"
	v := monitor.RecordTurn(16, output, 2, nil)

	if v != VerdictExtended {
		t.Fatalf("turn 16: expected VERDICT_EXTENDED, got %v", v)
	}
	if monitor.CurrentBudget != 20 {
		t.Fatalf("turn 16: expected CurrentBudget extended to 20, got %d", monitor.CurrentBudget)
	}
	if monitor.DeadloopDetected {
		t.Fatalf("turn 16: expected DeadloopDetected to be false")
	}

	// Verify advisory warning message format.
	if len(monitor.Warnings) == 0 {
		t.Fatalf("expected at least one advisory warning logged")
	}
	expectedWarn := "WARN [budget]: worker issue #12017 exceeded baseline budget; progress witnessed; granting dynamic extension (5 turns)"
	if !strings.Contains(monitor.Warnings[0], expectedWarn) {
		t.Fatalf("warning mismatch:\n  got:  %q\n  want: %q", monitor.Warnings[0], expectedWarn)
	}

	// Turns 17..20: inside the new extended budget (20).
	for turn := 17; turn <= 20; turn++ {
		v := monitor.RecordTurn(turn, "verifying tests", 1, nil)
		if v != VerdictContinue {
			t.Fatalf("turn %d: expected VERDICT_CONTINUE, got %v", turn, v)
		}
	}

	// Turn 21: second overrun, progress witnessed again.
	v2 := monitor.RecordTurn(21, "additional verification <!-- fak:progress milestone=\"tests_pass\" delta=\"+1 files\" tests=\"pass\" -->", 1, nil)
	if v2 != VerdictExtended {
		t.Fatalf("turn 21: expected VERDICT_EXTENDED, got %v", v2)
	}
	if monitor.CurrentBudget != 25 {
		t.Fatalf("turn 21: expected CurrentBudget extended to 25, got %d", monitor.CurrentBudget)
	}
	if monitor.DeadloopDetected {
		t.Fatalf("turn 21: expected DeadloopDetected to be false")
	}
}

func TestDynamicIterationBudget_DeadloopTermination(t *testing.T) {
	monitor := &BudgetMonitor{
		BaselineBudget: 15,
		CurrentBudget:  15,
		IssueNumber:    12017,
		MaxCap:         50,
	}

	// Turns 1..5: initial forward progress.
	for turn := 1; turn <= 5; turn++ {
		v := monitor.RecordTurn(turn, "working", 1, nil)
		if v != VerdictContinue {
			t.Fatalf("turn %d: expected VERDICT_CONTINUE, got %v", turn, v)
		}
	}

	errSigs := []string{"syntax error: unexpected token", "build failed"}

	// Turns 6..9: 4 consecutive zero-progress turns with repeating errors.
	for turn := 6; turn <= 9; turn++ {
		v := monitor.RecordTurn(turn, "trying command", 0, errSigs)
		if v != VerdictContinue {
			t.Fatalf("turn %d: expected VERDICT_CONTINUE, got %v", turn, v)
		}
		if monitor.DeadloopDetected {
			t.Fatalf("turn %d: deadloop should not be detected at turn %d (< 5 consecutive zero progress)", turn, turn)
		}
		if monitor.ConsecutiveZeroProgressTurns != turn-5 {
			t.Fatalf("turn %d: expected %d consecutive zero progress turns, got %d", turn, turn-5, monitor.ConsecutiveZeroProgressTurns)
		}
	}

	// Turn 10: 5th consecutive zero progress turn -> halts confirmed zero-progress deadloop.
	v := monitor.RecordTurn(10, "trying same command again", 0, errSigs)
	if v != VerdictDeadloopAbort {
		t.Fatalf("turn 10: expected VERDICT_DEADLOOP_ABORT, got %v", v)
	}
	if !monitor.DeadloopDetected {
		t.Fatalf("turn 10: expected DeadloopDetected to be true")
	}

	// Turn 11: subsequent turn continues to return abort.
	vNext := monitor.RecordTurn(11, "more attempts", 0, errSigs)
	if vNext != VerdictDeadloopAbort {
		t.Fatalf("turn 11: expected persistent VERDICT_DEADLOOP_ABORT, got %v", vNext)
	}
}

func TestDynamicIterationBudget_MilestoneParsing(t *testing.T) {
	output := `Starting implementation.
<!-- fak:progress milestone="repro_test" delta="+1 files" tests="fail" -->
Created test. Now implementing fix.
<!-- fak:progress milestone="core_fix" delta="+3 files" tests="pass" -->
<!-- fak:extend-turns count="10" reason="complex refactoring required" -->
All done.`

	milestones := ParseMilestones(output, 4)
	if len(milestones) != 2 {
		t.Fatalf("expected 2 milestones, got %d", len(milestones))
	}

	if milestones[0].Name != "repro_test" || milestones[0].DeltaFiles != 1 || milestones[0].TestsPass != false || milestones[0].Turn != 4 {
		t.Errorf("milestone[0] mismatch: %+v", milestones[0])
	}

	if milestones[1].Name != "core_fix" || milestones[1].DeltaFiles != 3 || milestones[1].TestsPass != true || milestones[1].Turn != 4 {
		t.Errorf("milestone[1] mismatch: %+v", milestones[1])
	}

	exts := ParseExtensionRequests(output)
	if len(exts) != 1 {
		t.Fatalf("expected 1 extension request, got %d", len(exts))
	}
	if exts[0].Count != 10 || exts[0].Reason != "complex refactoring required" {
		t.Errorf("extension mismatch: %+v", exts[0])
	}

	// Test tag format helpers
	progTag := FormatProgressTag("benchmark", 2, true)
	if !strings.Contains(progTag, `milestone="benchmark"`) || !strings.Contains(progTag, `delta="+2 files"`) || !strings.Contains(progTag, `tests="pass"`) {
		t.Errorf("FormatProgressTag generated unexpected tag: %s", progTag)
	}

	extTag := FormatExtensionTag(5, "need more time")
	if !strings.Contains(extTag, `count="5"`) || !strings.Contains(extTag, `reason="need more time"`) {
		t.Errorf("FormatExtensionTag generated unexpected tag: %s", extTag)
	}
}

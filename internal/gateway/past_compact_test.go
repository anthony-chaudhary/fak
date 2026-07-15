package gateway

import (
	"strings"
	"testing"
)

func TestPastCompactEscalationLadderAndReset(t *testing.T) {
	srv := &Server{}
	first := srv.observePastCompact("trace", true, false)
	if first.Consecutive != 1 || first.Escalated || first.Action != PastCompactActionCheckpoint {
		t.Fatalf("first = %+v", first)
	}
	second := srv.observePastCompact("trace", true, false)
	if second.Consecutive != 2 || second.Escalated {
		t.Fatalf("second = %+v", second)
	}
	third := srv.observePastCompact("trace", true, false)
	if third.Consecutive != 3 || !third.Escalated || third.Action != PastCompactActionRotate {
		t.Fatalf("third = %+v", third)
	}
	field := formatPastCompactEscalation(third)
	for _, want := range []string{"escalation=past-compact-repeat", "consecutive=3", "next_action=rotate"} {
		if !strings.Contains(field, want) {
			t.Fatalf("field %q missing %q", field, want)
		}
	}
	park := third
	for i := 0; i < 3; i++ {
		park = srv.observePastCompact("trace", true, false)
	}
	if park.Action != PastCompactActionPark {
		t.Fatalf("park = %+v", park)
	}
	reset := srv.observePastCompact("trace", true, true)
	if reset.Consecutive != 0 || reset.Escalated {
		t.Fatalf("checkpoint reset = %+v", reset)
	}
	after := srv.observePastCompact("trace", true, false)
	if after.Consecutive != 1 || after.Escalated {
		t.Fatalf("after reset = %+v", after)
	}
}

func TestPastCompactOneOffKeepsOnlyExistingNudge(t *testing.T) {
	srv := &Server{}
	line := formatTurnDebugStatsWithBudget("trace", "wire", true, "end_turn", 100, 20, 0, 0, false, 100, ResetDecision{}, false)
	escalation := srv.observePastCompact("trace", compactionBudgetPast(100, 20, 0, 0, 100), false)
	line += formatPastCompactEscalation(escalation)
	if !strings.Contains(line, "nudge=past-inversion-checkpoint-now") {
		t.Fatalf("line = %q", line)
	}
	if strings.Contains(line, "escalation=") {
		t.Fatalf("one-off escalated: %q", line)
	}
}

func TestPastCompactBelowThresholdResetsRun(t *testing.T) {
	srv := &Server{}
	_ = srv.observePastCompact("trace", true, false)
	_ = srv.observePastCompact("trace", true, false)
	reset := srv.observePastCompact("trace", false, false)
	if reset.Consecutive != 0 {
		t.Fatalf("reset = %+v", reset)
	}
	next := srv.observePastCompact("trace", true, false)
	if next.Consecutive != 1 {
		t.Fatalf("next = %+v", next)
	}
}

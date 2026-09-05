package agent

import (
	"testing"
)

func TestGoalContinuation_TripsStallDetectorOnConsecutiveZeroProgress(t *testing.T) {
	session := NewGoalContinuationSession("goal-trip", "Trips stall detector on zero progress")

	if session.BranchBlocked {
		t.Fatalf("expected session.BranchBlocked to start false")
	}
	if !session.CanContinue() {
		t.Fatalf("expected session.CanContinue() to start true")
	}
	if session.MaxConsecutiveNoProgress != 5 {
		t.Fatalf("expected MaxConsecutiveNoProgress default 5, got %d", session.MaxConsecutiveNoProgress)
	}

	// 4 consecutive turns with zero progress: must not trip BranchBlocked
	for i := 1; i <= 4; i++ {
		blocked := session.RecordTurnProgress(TurnProgress{
			HasProgress:    false,
			ToolExecutions: 0,
			StateDelta:     false,
		})
		if blocked {
			t.Fatalf("turn %d unexpectedly returned BranchBlocked=true", i)
		}
		if session.BranchBlocked {
			t.Fatalf("turn %d unexpectedly tripped session.BranchBlocked", i)
		}
		if !session.CanContinue() {
			t.Fatalf("turn %d unexpectedly suppressed continuation", i)
		}
		if session.ConsecutiveNoProgress() != i {
			t.Fatalf("turn %d expected ConsecutiveNoProgress=%d, got %d", i, i, session.ConsecutiveNoProgress())
		}
	}

	// 5th consecutive turn with zero progress: must trip BranchBlocked
	blocked := session.RecordTurnProgress(TurnProgress{
		HasProgress:    false,
		ToolExecutions: 0,
		StateDelta:     false,
	})
	if !blocked {
		t.Fatalf("5th consecutive zero-progress turn must return BranchBlocked=true")
	}
	if !session.BranchBlocked {
		t.Fatalf("expected session.BranchBlocked to be true after 5 zero-progress turns")
	}
	if session.CanContinue() {
		t.Fatalf("expected session.CanContinue() to be false (continuation suppressed)")
	}
	if session.StallDetector == nil || !session.StallDetector.IsStalled() {
		t.Fatalf("expected StallDetector.IsStalled() to be true")
	}

	// Further continuation calls must be suppressed
	nextBlocked := session.RecordTurnProgress(false)
	if !nextBlocked {
		t.Fatalf("further turn after trip must continue returning BranchBlocked=true")
	}

	env := map[string]string{"STATUS": "idle"}
	up, msg := session.FormatWorldState(env, false)
	if up.Full || up.DriftSeen {
		t.Fatalf("FormatWorldState after BranchBlocked must not emit full update or drift")
	}
	if msg.Role != RoleSystem {
		t.Fatalf("FormatWorldState after BranchBlocked expected RoleSystem, got %s", msg.Role)
	}
}

func TestGoalContinuation_ProgressResetsConsecutiveZeroProgress(t *testing.T) {
	session := NewGoalContinuationSession("goal-reset", "Reset stall detector on progress")

	// 4 zero-progress turns
	for i := 0; i < 4; i++ {
		if session.RecordTurnProgress(false) {
			t.Fatalf("turn %d should not be blocked", i)
		}
	}
	if session.ConsecutiveNoProgress() != 4 {
		t.Fatalf("expected 4 consecutive no progress, got %d", session.ConsecutiveNoProgress())
	}

	// 1 turn with progress (tool execution > 0)
	if session.RecordTurnProgress(TurnProgress{ToolExecutions: 2}) {
		t.Fatalf("progress turn should not be blocked")
	}
	if session.ConsecutiveNoProgress() != 0 {
		t.Fatalf("expected consecutive no progress to reset to 0, got %d", session.ConsecutiveNoProgress())
	}

	// 4 more zero-progress turns: still not blocked
	for i := 1; i <= 4; i++ {
		if session.RecordTurnProgress(false) {
			t.Fatalf("turn %d after reset should not be blocked", i)
		}
	}
	if session.BranchBlocked {
		t.Fatalf("session should not be blocked after 4 consecutive no progress turns")
	}

	// 5th zero-progress turn trips
	if !session.RecordTurnProgress(false) {
		t.Fatalf("5th zero-progress turn must trip BranchBlocked")
	}
	if !session.BranchBlocked {
		t.Fatalf("expected session.BranchBlocked to be true")
	}
}

func TestGoalContinuation_MaxTurnsBudget(t *testing.T) {
	session := NewGoalContinuationSession("goal-budget", "Test max turns budget", WithMaxTurns(3))

	if session.MaxTurns != 3 {
		t.Fatalf("expected MaxTurns=3, got %d", session.MaxTurns)
	}

	// Turn 1 with progress
	session.RecordTurnProgress(true)
	if session.BranchBlocked || !session.CanContinue() {
		t.Fatalf("turn 1 should not be blocked")
	}

	// Turn 2 with progress
	session.RecordTurnProgress(true)
	if session.BranchBlocked || !session.CanContinue() {
		t.Fatalf("turn 2 should not be blocked")
	}

	// Turn 3 with progress (hits budget 3)
	session.RecordTurnProgress(true)
	if !session.BranchBlocked {
		t.Fatalf("turn 3 exceeding MaxTurns must set BranchBlocked")
	}
	if session.CanContinue() {
		t.Fatalf("continuation must be suppressed when MaxTurns is reached")
	}
	if session.StallReason != StallReasonMaxTurnsExceeded {
		t.Fatalf("expected StallReason %q, got %q", StallReasonMaxTurnsExceeded, session.StallReason)
	}
}

func TestGoalContinuation_RecordDenialTracking(t *testing.T) {
	session := NewGoalContinuationSession("goal-denial", "Test tool denial tracking")

	session.RecordDenial("rm_rf", "POLICY_BLOCK")
	session.RecordDenial("DROP_TABLE", "POLICY_BLOCK")
	session.RecordDenial("EXTERNAL_NETWORK")

	if session.DenialsTotal() != 3 {
		t.Fatalf("expected 3 denials, got %d", session.DenialsTotal())
	}

	reasons := session.DenialsByReason()
	if reasons["POLICY_BLOCK"] != 2 {
		t.Fatalf("expected 2 POLICY_BLOCK denials, got %d", reasons["POLICY_BLOCK"])
	}
	if reasons["EXTERNAL_NETWORK"] != 1 {
		t.Fatalf("expected 1 EXTERNAL_NETWORK denial, got %d", reasons["EXTERNAL_NETWORK"])
	}
}

func TestGoalContinuation_FormatWorldStateConsecutiveStall(t *testing.T) {
	session := NewGoalContinuationSession("goal-ws-stall", "Test format world state stall")
	env := map[string]string{"ENV": "staging"}

	// Turn 1: initial baseline (turnsDelivered = 1, isFirst = true)
	session.FormatWorldState(env, false)
	if session.BranchBlocked {
		t.Fatalf("turn 1 should not be blocked")
	}

	// Turns 2..5: 4 consecutive turns with identical env hash
	for i := 2; i <= 5; i++ {
		session.FormatWorldState(env, false)
		if session.BranchBlocked {
			t.Fatalf("turn %d should not be blocked yet", i)
		}
	}

	// Turn 6: 5th consecutive turn with identical env hash trips MaxConsecutiveNoProgress (5)
	session.FormatWorldState(env, false)
	if !session.BranchBlocked {
		t.Fatalf("turn 6 with 5 consecutive identical world states must trip BranchBlocked")
	}
	if session.CanContinue() {
		t.Fatalf("continuation must be suppressed")
	}
}

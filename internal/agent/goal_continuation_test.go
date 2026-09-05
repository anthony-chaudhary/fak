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

func TestGoalContinuation_CombinedWorldStateAndTurnProgressLifecycle(t *testing.T) {
	session := NewGoalContinuationSession("goal-lifecycle", "Combined world state and turn progress lifecycle")
	env := map[string]string{"ENV": "production", "STEP": "active"}

	for i := 1; i <= 5; i++ {
		up, msg := session.FormatWorldState(env, false)
		if i == 1 && !up.Full {
			t.Fatalf("turn 1 expected full world state snapshot")
		} else if i > 1 && up.Full {
			t.Fatalf("turn %d with unchanged env expected partial diff, got full", i)
		}
		if msg.Role != RoleSystem {
			t.Fatalf("turn %d expected RoleSystem message, got %s", i, msg.Role)
		}
		// FormatWorldState must be idempotent context formatting and not increment turnsDelivered
		if session.TurnsDelivered() != i-1 {
			t.Fatalf("turn %d: FormatWorldState unexpectedly incremented turnsDelivered, got %d, expected %d", i, session.TurnsDelivered(), i-1)
		}

		blocked := session.RecordTurnProgress(TurnProgress{ToolExecutions: 1})
		if blocked {
			t.Fatalf("turn %d unexpectedly returned blocked=true", i)
		}
		if session.BranchBlocked {
			t.Fatalf("turn %d unexpectedly tripped BranchBlocked", i)
		}
		if !session.CanContinue() {
			t.Fatalf("turn %d unexpectedly suppressed continuation", i)
		}
		if session.TurnsDelivered() != i {
			t.Fatalf("turn %d expected TurnsDelivered() == %d, got %d", i, i, session.TurnsDelivered())
		}
	}

	if session.TurnsDelivered() != 5 {
		t.Fatalf("expected TurnsDelivered() == 5, got %d", session.TurnsDelivered())
	}
	if !session.CanContinue() {
		t.Fatalf("expected CanContinue() == true after 5 turns with progress")
	}
	if session.BranchBlocked {
		t.Fatalf("expected BranchBlocked == false")
	}
}

func TestGoalContinuation_FormatWorldStateIdempotentDoesNotStall(t *testing.T) {
	session := NewGoalContinuationSession("goal-ws-idempotent", "Test format world state idempotency")
	env := map[string]string{"ENV": "staging"}

	for i := 1; i <= 10; i++ {
		up, msg := session.FormatWorldState(env, false)
		if i == 1 && !up.Full {
			t.Fatalf("turn 1 must emit full snapshot")
		} else if i > 1 && up.Full {
			t.Fatalf("call %d with unchanged env must emit partial diff", i)
		}
		if msg.Role != RoleSystem {
			t.Fatalf("call %d expected RoleSystem, got %s", i, msg.Role)
		}
		if session.TurnsDelivered() != 0 {
			t.Fatalf("FormatWorldState must not increment TurnsDelivered, got %d", session.TurnsDelivered())
		}
		if session.BranchBlocked {
			t.Fatalf("FormatWorldState must not trip BranchBlocked on consecutive calls")
		}
		if !session.CanContinue() {
			t.Fatalf("FormatWorldState must not suppress continuation")
		}
	}
}

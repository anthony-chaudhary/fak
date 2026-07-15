package session

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestCumulativeEnvelopeReplayIntervenesByFourthSemanticDenial(t *testing.T) {
	envelope := NewCumulativeEnvelope(CumulativeEnvelopePolicy{
		MaxUncachedInputTokens: 100_000,
		MaxWallTimeNanos:       int64(10 * time.Minute),
		MaxEquivalentRefusals:  4,
		MinRefusalDensity:      0.75,
	})

	state := DefaultState("session-019f6745")
	state.Goal = Goal{ID: "publish-executive-report", Priority: 2, Budget: 200_000}
	tools := []string{"shell", "write_file", "patch_file", "powershell"}

	var decision CumulativeEnvelopeDecision
	for i := 1; i <= 4; i++ {
		// The attempted command changes on every replay, but the refusal is
		// semantically the same: SELF_MODIFY against the same guarded target for
		// the same intended effect.
		state.PendingTurn = PendingTurn{
			Attempt:           i,
			LastStatus:        403,
			StartedAtUnixNano: int64(i),
		}
		state.Rev = uint64(i)
		decision = envelope.Observe(state, CumulativeEnvelopeSample{
			InputTokens:         20_000,
			CachedInputTokens:   0,
			OutputTokens:        100,
			ModelCalls:          1,
			ToolCalls:           1,
			WallTimeNanos:       int64(30 * time.Second),
			ManualContinuations: boolInt(i > 1),
			Outcome: ToolCallOutcome{
				Tool:           tools[i-1],
				ToolCallID:     fmt.Sprintf("mutated-command-%d", i),
				Kind:           ToolCallOutcomeRejected,
				Reason:         abi.ReasonSelfModify,
				Disposition:    ToolDispositionTerminal,
				Target:         "guard/private-bridge",
				IntendedEffect: "publish-executive-report",
			},
		})

		if i < 4 && decision.Action != EnvelopeActionContinue {
			t.Fatalf("denial %d action = %s, want continue below the configured threshold", i, decision.Action)
		}
	}

	if decision.Action != EnvelopeActionCheckpointRecovery {
		t.Fatalf("fourth equivalent denial action = %s, want checkpoint recovery", decision.Action)
	}
	if decision.Reason != ReasonEnvelopeSemanticRefusal {
		t.Fatalf("fourth equivalent denial reason = %q, want %q", decision.Reason, ReasonEnvelopeSemanticRefusal)
	}
	if decision.EquivalentRefusals != 4 || decision.RefusalDensity != 1 {
		t.Fatalf("semantic refusal signal = count %d density %.2f, want 4/1.0", decision.EquivalentRefusals, decision.RefusalDensity)
	}
	if decision.Totals.UncachedInputTokens != 80_000 {
		t.Fatalf("uncached input at intervention = %d, want 80000 (before the 100k envelope)", decision.Totals.UncachedInputTokens)
	}
	if decision.Totals.ManualContinuations != 3 {
		t.Fatalf("manual continuations = %d, want 3", decision.Totals.ManualContinuations)
	}
	if decision.Outcome.Kind != ToolCallOutcomeRejected || decision.Outcome.Reason != abi.ReasonSelfModify {
		t.Fatalf("intervention weakened the latest denial: %+v", decision.Outcome)
	}
	if decision.Recovery.Reason != ReasonEnvelopeSemanticRefusal || decision.Recovery.TraceID != state.TraceID {
		t.Fatalf("typed recovery = %+v, want reason %q and trace %q", decision.Recovery, ReasonEnvelopeSemanticRefusal, state.TraceID)
	}
	if decision.Recovery.Goal != state.Goal {
		t.Fatalf("recovery goal = %+v, want latest %+v", decision.Recovery.Goal, state.Goal)
	}
	if decision.Recovery.PendingTurn != state.PendingTurn || decision.Recovery.StateRev != state.Rev {
		t.Fatalf("recovery checkpoint = %+v, want latest pending turn %+v at rev %d", decision.Recovery, state.PendingTurn, state.Rev)
	}
}

func TestCumulativeEnvelopeTripsConfiguredUncachedInputBoundary(t *testing.T) {
	envelope := NewCumulativeEnvelope(CumulativeEnvelopePolicy{
		MaxUncachedInputTokens: 90_000,
		MaxEquivalentRefusals:  4,
	})
	state := DefaultState("uncached-boundary")
	state.Goal = Goal{ID: "finish-bounded-task"}

	var decision CumulativeEnvelopeDecision
	for i := 1; i <= 3; i++ {
		decision = envelope.Observe(state, CumulativeEnvelopeSample{
			InputTokens:       35_000,
			CachedInputTokens: 5_000,
			ModelCalls:        1,
			WallTimeNanos:     int64(time.Minute),
		})
	}

	if decision.Action != EnvelopeActionCheckpointRecovery || decision.Reason != ReasonEnvelopeUncachedInput {
		t.Fatalf("uncached boundary decision = %+v, want checkpoint with %q", decision, ReasonEnvelopeUncachedInput)
	}
	if decision.Totals.InputTokens != 105_000 || decision.Totals.CachedInputTokens != 15_000 || decision.Totals.UncachedInputTokens != 90_000 {
		t.Fatalf("token totals = %+v, want input/cached/uncached 105000/15000/90000", decision.Totals)
	}
	if decision.Recovery.Goal != state.Goal {
		t.Fatalf("uncached checkpoint lost goal: got %+v want %+v", decision.Recovery.Goal, state.Goal)
	}
}

func TestCumulativeEnvelopeTripsConfiguredWallTimeBoundary(t *testing.T) {
	envelope := NewCumulativeEnvelope(CumulativeEnvelopePolicy{
		MaxWallTimeNanos: int64(90 * time.Second),
	})
	state := DefaultState("wall-boundary")
	state.Goal = Goal{ID: "recover-long-running-task"}

	first := envelope.Observe(state, CumulativeEnvelopeSample{WallTimeNanos: int64(45 * time.Second)})
	if first.Action != EnvelopeActionContinue {
		t.Fatalf("first wall sample action = %s, want continue", first.Action)
	}
	decision := envelope.Observe(state, CumulativeEnvelopeSample{WallTimeNanos: int64(45 * time.Second)})
	if decision.Action != EnvelopeActionCheckpointRecovery || decision.Reason != ReasonEnvelopeWallTime {
		t.Fatalf("wall boundary decision = %+v, want checkpoint with %q", decision, ReasonEnvelopeWallTime)
	}
	if decision.Totals.WallTimeNanos != int64(90*time.Second) || decision.Recovery.Goal != state.Goal {
		t.Fatalf("wall checkpoint = %+v, want 90s with goal %+v", decision, state.Goal)
	}
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

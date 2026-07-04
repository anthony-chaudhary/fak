package main

// session_time_observability_test.go — the wall-clock budget must be OBSERVABLE in
// `fak session status`, not just armed and enforced. `--max-duration` (and the
// managed-context wall axis) promises "Query/inspect anytime with `fak session status
// <id>`", but before #1584's projection was wired the token Budget/Pace crossed the
// wire while the TimeBudget was dropped on the floor. These tests pin the projection
// (toGatewaySessionStateAt) and the human render (formatSessionState) so the promise
// cannot silently regress.

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestToGatewaySessionTimeProjection(t *testing.T) {
	base := time.Unix(1_700_000_000, 0)

	t.Run("bounded and ticking", func(t *testing.T) {
		tb := session.NewTimeBudget().WithLimit(2 * time.Hour).Start(base)
		st := session.State{TraceID: "sess-1", Run: session.Running, Time: tb}
		got := toGatewaySessionStateAt(st, base.Add(5*time.Minute)).Time
		if !got.Bounded || got.Exceeded {
			t.Fatalf("bounded/exceeded = %v/%v, want true/false", got.Bounded, got.Exceeded)
		}
		if got.ElapsedSeconds != 300 || got.RemainingSeconds != 6900 || got.LimitSeconds != 7200 {
			t.Fatalf("elapsed/remaining/limit = %d/%d/%d, want 300/6900/7200",
				got.ElapsedSeconds, got.RemainingSeconds, got.LimitSeconds)
		}
	})

	t.Run("exceeded floors remaining at zero", func(t *testing.T) {
		tb := session.NewTimeBudget().WithLimit(2 * time.Hour).Start(base)
		got := toGatewaySessionStateAt(session.State{Time: tb}, base.Add(3*time.Hour)).Time
		if !got.Exceeded || got.RemainingSeconds != 0 {
			t.Fatalf("exceeded/remaining = %v/%d, want true/0", got.Exceeded, got.RemainingSeconds)
		}
	})

	t.Run("unbounded but started still reports elapsed", func(t *testing.T) {
		tb := session.NewTimeBudget().Start(base) // no WithLimit → unbounded
		got := toGatewaySessionStateAt(session.State{Time: tb}, base.Add(90*time.Second)).Time
		if got.Bounded {
			t.Fatalf("unbounded budget must project Bounded=false, got %+v", got)
		}
		if got.ElapsedSeconds != 90 {
			t.Fatalf("elapsed = %d, want 90 (unbounded-but-tracked still visible)", got.ElapsedSeconds)
		}
	})

	t.Run("never configured projects to zero", func(t *testing.T) {
		got := toGatewaySessionStateAt(session.State{TraceID: "sess-1", Run: session.Running}, base).Time
		if !got.IsZero() {
			t.Fatalf("a session with no time budget must project a zero SessionTime, got %+v", got)
		}
	})
}

// TestToGatewaySessionBudgetProjectsContextSignals pins the two signals the outbound-compaction
// burst gate reads on a context-budgeted-but-turn-unbounded session: the context ceiling
// (Budget.ContextTokensCap) and the last debited turn's resident window (Cost.LatestContextTokens).
// Before these were projected, a headless `fak guard -- claude` with a --context-budget-tokens but
// no turn budget crossed the wire with no way to reason about its remaining horizon. The
// un-budgeted default must still project both as zero so its wire form is unchanged.
func TestToGatewaySessionBudgetProjectsContextSignals(t *testing.T) {
	base := time.Unix(1_700_000_000, 0)

	t.Run("context-budgeted session carries cap and last resident", func(t *testing.T) {
		// Construct the ring via its exported fields (push is package-private): one debited turn
		// whose resident window was 52000 tokens, at ring head 1 / count 1.
		ring := session.CostRing{Count: 1, Head: 1}
		ring.Turns[0] = session.TurnCost{OutputTokens: 300, ContextTokens: 52000}
		st := session.State{
			TraceID: "sess-ctx",
			Run:     session.Running,
			Budget: session.Budget{
				TurnsLeft:         session.Unbounded, // no turn horizon — the common headless shape
				TokensLeft:        session.Unbounded,
				ContextTokensLeft: 12000,
				ContextTokensCap:  64000,
			},
			Cost: ring,
		}
		got := toGatewaySessionStateAt(st, base).Budget
		if got.ContextTokensCap != 64000 {
			t.Fatalf("ContextTokensCap = %d, want 64000", got.ContextTokensCap)
		}
		if got.ResidentContextTokens != 52000 {
			t.Fatalf("ResidentContextTokens = %d, want 52000 (last debited turn's resident window)", got.ResidentContextTokens)
		}
		if got.ContextTokensLeft != 12000 {
			t.Fatalf("ContextTokensLeft = %d, want 12000 (unchanged)", got.ContextTokensLeft)
		}
	})

	t.Run("un-budgeted session projects both signals as zero", func(t *testing.T) {
		got := toGatewaySessionStateAt(session.State{TraceID: "sess-plain", Run: session.Running}, base).Budget
		if got.ContextTokensCap != 0 || got.ResidentContextTokens != 0 {
			t.Fatalf("an un-budgeted, never-debited session must project zero context signals, got cap=%d resident=%d", got.ContextTokensCap, got.ResidentContextTokens)
		}
	})
}

func TestFormatSessionStateRendersTime(t *testing.T) {
	base := time.Unix(1_700_000_000, 0)

	bounded := toGatewaySessionStateAt(
		session.State{TraceID: "sess-1", Run: session.Running,
			Time: session.NewTimeBudget().WithLimit(2 * time.Hour).Start(base)},
		base.Add(5*time.Minute))
	line := formatSessionState(bounded)
	for _, want := range []string{"time(elapsed=5m0s", "remaining=1h55m0s", "limit=2h0m0s"} {
		if !strings.Contains(line, want) {
			t.Fatalf("bounded line missing %q: %s", want, line)
		}
	}
	if strings.Contains(line, "EXCEEDED") {
		t.Fatalf("a non-exhausted budget must not render EXCEEDED: %s", line)
	}

	exceeded := toGatewaySessionStateAt(
		session.State{TraceID: "sess-1", Run: session.Draining,
			Time: session.NewTimeBudget().WithLimit(2 * time.Hour).Start(base)},
		base.Add(3*time.Hour))
	if l := formatSessionState(exceeded); !strings.Contains(l, "EXCEEDED") {
		t.Fatalf("an exhausted wall-clock budget must render EXCEEDED: %s", l)
	}

	// No time budget: the line must be byte-identical to the pre-#1584 shape (no time segment).
	plain := formatSessionState(toGatewaySessionStateAt(session.State{TraceID: "sess-1", Run: session.Running}, base))
	if strings.Contains(plain, "time(") {
		t.Fatalf("a session with no wall-clock budget must not render a time segment: %s", plain)
	}
}

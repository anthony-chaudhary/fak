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

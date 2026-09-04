package loopunblock

import (
	"testing"
)

// BenchmarkLoopUnblock exercises the pure Decide and Assess decision logic across
// representative candidate lists and tick histories in a loop.
func BenchmarkLoopUnblock(b *testing.B) {
	candsBypass := []Candidate{
		{ID: "task-1", Rank: 1, Admit: Blocked(CauseLeaseLive, "peer lease active")},
		{ID: "task-2", Rank: 2, Admit: Blocked(CauseCapped, "account rate capped")},
		{ID: "task-3", Rank: 3, Admit: Admittable()},
		{ID: "task-4", Rank: 4, Admit: Admittable()},
	}
	candsClear := []Candidate{
		{ID: "task-1", Rank: 1, Admit: Blocked(CauseLeaseStale, "pid 999 gone")},
		{ID: "task-2", Rank: 2, Admit: Admittable()},
	}
	candsWait := []Candidate{
		{ID: "task-1", Rank: 1, Admit: Blocked(CauseCapped, "limit reached")},
	}
	pol := Policy{}

	history := []Tick{
		{Action: ActionWait, Head: "task-1", UnixNano: 1_000_000_000},
		{Action: ActionWait, Head: "task-1", UnixNano: 2_000_000_000},
		{Action: ActionBypass, Head: "task-1", UnixNano: 3_000_000_000},
	}
	stallPol := StallPolicy{StallAfter: 3, EscalateAfterSeconds: 300}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d1 := Decide(candsBypass, pol)
		if d1.Action != ActionBypass {
			b.Fatalf("expected bypass, got %s", d1.Action)
		}
		d2 := Decide(candsClear, pol)
		if d2.Action != ActionClearThenEnter {
			b.Fatalf("expected clear_then_enter, got %s", d2.Action)
		}
		d3 := Decide(candsWait, pol)
		if d3.Action != ActionWait {
			b.Fatalf("expected wait, got %s", d3.Action)
		}
		v := Assess(history, stallPol, 4_000_000_000)
		if v.Stalled {
			b.Fatalf("expected not stalled")
		}
	}
}

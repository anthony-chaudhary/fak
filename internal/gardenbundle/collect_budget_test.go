package gardenbundle

import (
	"fmt"
	"testing"
	"time"
)

type collectClock struct{ t time.Time }

func (c *collectClock) now() time.Time      { return c.t }
func (c *collectClock) add(d time.Duration) { c.t = c.t.Add(d) }

func okPayload() map[string]any {
	return map[string]any{"ok": true, "verdict": "OK", "reason": "clear"}
}

func TestCollectBoundedDoesNotStartAfterGlobalBudgetSpent(t *testing.T) {
	clk := &collectClock{t: time.Unix(1_700_000_000, 0)}
	calls := 0
	results, p := CollectBounded(t.TempDir(), CollectOptions{
		Budget: time.Second,
		Start:  clk.now().Add(-time.Hour),
		Now:    clk.now,
		Run: func(string, Member, string, time.Duration) (map[string]any, int, string) {
			calls++
			return okPayload(), 0, ""
		},
	})
	if calls != 0 || len(results) != 0 {
		t.Fatalf("expired budget ran %d member(s), results=%v", calls, results)
	}
	if !p.Exhausted || p.Next != Members[0].Key || len(p.Deferred) != len(Members) {
		t.Fatalf("progress = %+v", p)
	}
}

func TestCollectBoundedCapsMemberTimeoutToGlobalRemaining(t *testing.T) {
	clk := &collectClock{t: time.Unix(1_700_000_000, 0)}
	var timeouts []time.Duration
	_, p := CollectBounded(t.TempDir(), CollectOptions{
		PerMemberTimeout: 30 * time.Second,
		Budget:           5 * time.Second,
		Start:            clk.now(),
		Now:              clk.now,
		Run: func(_ string, _ Member, _ string, timeout time.Duration) (map[string]any, int, string) {
			timeouts = append(timeouts, timeout)
			clk.add(3 * time.Second)
			return okPayload(), 0, ""
		},
	})
	if len(timeouts) != 2 || timeouts[0] != 5*time.Second || timeouts[1] != 2*time.Second {
		t.Fatalf("timeouts = %v, want [5s 2s]", timeouts)
	}
	if !p.Exhausted || p.Completed != 2 || p.Next != Members[2].Key {
		t.Fatalf("progress = %+v", p)
	}
}

func TestCollectBoundedCheckpointsAndConvergesAcrossTicks(t *testing.T) {
	var prior []MemberResult
	next := ""
	seen := map[string]int{}
	for tick := 0; tick < len(Members); tick++ {
		clk := &collectClock{t: time.Unix(1_700_000_000+int64(tick), 0)}
		results, p := CollectBounded(t.TempDir(), CollectOptions{
			PerMemberTimeout: time.Minute,
			Budget:           time.Second,
			Start:            clk.now(),
			Now:              clk.now,
			Next:             next,
			Prior:            prior,
			Run: func(_ string, m Member, _ string, _ time.Duration) (map[string]any, int, string) {
				seen[m.Key]++
				clk.add(2 * time.Second)
				return okPayload(), 0, ""
			},
			Checkpoint: func(gotNext string, got []MemberResult) error {
				next = gotNext
				prior = got
				return nil
			},
		})
		prior = results
		next = p.Next
	}
	if len(prior) != len(Members) {
		t.Fatalf("completed %d/%d members", len(prior), len(Members))
	}
	for _, m := range Members {
		if seen[m.Key] != 1 {
			t.Fatalf("member %q ran %d times, want once (seen=%v)", m.Key, seen[m.Key], seen)
		}
	}
	if next != "" {
		t.Fatalf("final next = %q, want empty", next)
	}
}

func TestCollectBoundedCheckpointFailureRefusesMoreMembers(t *testing.T) {
	calls := 0
	_, p := CollectBounded(t.TempDir(), CollectOptions{
		PerMemberTimeout: time.Second,
		Run: func(string, Member, string, time.Duration) (map[string]any, int, string) {
			calls++
			return okPayload(), 0, ""
		},
		Checkpoint: func(string, []MemberResult) error { return fmt.Errorf("disk full") },
	})
	if calls != 1 {
		t.Fatalf("checkpoint failure ran %d members, want 1", calls)
	}
	if p.CheckpointError == "" || len(p.Deferred) != len(Members)-1 {
		t.Fatalf("progress = %+v", p)
	}
}

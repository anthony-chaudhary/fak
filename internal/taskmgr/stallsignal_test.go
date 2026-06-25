package taskmgr

import (
	"testing"
	"time"
)

// TestRepeatStallTripsOnThirdIdenticalCallInTurn proves the core contract: the same
// tool-input hash seen a third time within one turn flips Stalled, while the first two
// are clean and a DIFFERENT hash never trips.
func TestRepeatStallTripsOnThirdIdenticalCallInTurn(t *testing.T) {
	base := time.Unix(1700000000, 0)
	now := base
	r := NewRepeatStallMonitor(WithRepeatClock(func() time.Time { return now }))

	if r.Threshold() != DefaultRepeatStallThreshold {
		t.Fatalf("threshold = %d, want %d", r.Threshold(), DefaultRepeatStallThreshold)
	}

	// First identical call: work.
	s1 := r.ObserveToolCall("task_a", "hash_read_foo")
	if s1.Stalled || s1.Count != 1 {
		t.Fatalf("call 1 = stalled %v count %d, want false/1", s1.Stalled, s1.Count)
	}
	if s1.FirstSeenUnixNano != base.UnixNano() {
		t.Fatalf("call 1 first-seen = %d, want %d", s1.FirstSeenUnixNano, base.UnixNano())
	}

	// A different hash in the same turn is independent — never a repeat.
	if s := r.ObserveToolCall("task_a", "hash_grep_bar"); s.Stalled || s.Count != 1 {
		t.Fatalf("other hash = stalled %v count %d, want false/1", s.Stalled, s.Count)
	}

	// Second identical call: a retry, still clean.
	now = now.Add(1 * time.Second)
	if s := r.ObserveToolCall("task_a", "hash_read_foo"); s.Stalled || s.Count != 2 {
		t.Fatalf("call 2 = stalled %v count %d, want false/2", s.Stalled, s.Count)
	}

	// Third identical call inside the same turn: the wedge. First-seen is pinned to the
	// FIRST occurrence so a host can report how long the loop has spun.
	now = now.Add(1 * time.Second)
	s3 := r.ObserveToolCall("task_a", "hash_read_foo")
	if !s3.Stalled || s3.Count != 3 {
		t.Fatalf("call 3 = stalled %v count %d, want true/3", s3.Stalled, s3.Count)
	}
	if s3.FirstSeenUnixNano != base.UnixNano() {
		t.Fatalf("call 3 first-seen = %d, want pinned to base %d", s3.FirstSeenUnixNano, base.UnixNano())
	}
}

// TestRepeatStallResetsAcrossTurns proves the "within ONE turn" scope: the same hash
// re-issued in a NEW turn is normal work, not a stall, because NextTurn clears counts.
func TestRepeatStallResetsAcrossTurns(t *testing.T) {
	now := time.Unix(1700000000, 0)
	r := NewRepeatStallMonitor(WithRepeatClock(func() time.Time { return now }))

	// Trip it in turn 0.
	r.ObserveToolCall("t", "h")
	r.ObserveToolCall("t", "h")
	if s := r.ObserveToolCall("t", "h"); !s.Stalled || s.Turn != 0 {
		t.Fatalf("turn 0 third call = stalled %v turn %d, want true/0", s.Stalled, s.Turn)
	}

	// Advance the turn: counts clear, so the same hash is clean again.
	if turn := r.NextTurn("t"); turn != 1 {
		t.Fatalf("NextTurn = %d, want 1", turn)
	}
	s := r.ObserveToolCall("t", "h")
	if s.Stalled || s.Count != 1 || s.Turn != 1 {
		t.Fatalf("turn 1 first call = stalled %v count %d turn %d, want false/1/1", s.Stalled, s.Count, s.Turn)
	}
}

// TestRepeatStallThresholdOverrideAndGuards proves WithRepeatThreshold takes effect and
// that an invalid (<2) threshold is ignored, and that the empty hash is a no-op.
func TestRepeatStallThresholdOverrideAndGuards(t *testing.T) {
	r := NewRepeatStallMonitor(WithRepeatThreshold(2))
	if r.Threshold() != 2 {
		t.Fatalf("threshold = %d, want 2", r.Threshold())
	}
	r.ObserveToolCall("x", "h")
	if s := r.ObserveToolCall("x", "h"); !s.Stalled || s.Count != 2 {
		t.Fatalf("threshold-2 second call = stalled %v count %d, want true/2", s.Stalled, s.Count)
	}

	// An invalid threshold (< 2) is ignored: the default stands.
	r2 := NewRepeatStallMonitor(WithRepeatThreshold(1))
	if r2.Threshold() != DefaultRepeatStallThreshold {
		t.Fatalf("threshold = %d, want default %d (1 ignored)", r2.Threshold(), DefaultRepeatStallThreshold)
	}

	// The empty hash never counts as a repeat — it cannot wedge a task.
	for i := 0; i < 5; i++ {
		if s := r2.ObserveToolCall("y", ""); s.Stalled || s.Count != 0 {
			t.Fatalf("empty-hash observe = stalled %v count %d, want false/0", s.Stalled, s.Count)
		}
	}
}

// TestRepeatStallForgetDropsState proves Forget clears a task's per-task maps so a
// host can release them when the task ends.
func TestRepeatStallForgetDropsState(t *testing.T) {
	r := NewRepeatStallMonitor()
	r.ObserveToolCall("z", "h")
	r.ObserveToolCall("z", "h")
	r.Forget("z")
	// After Forget, the hash starts over: count 1, not 3.
	if s := r.ObserveToolCall("z", "h"); s.Stalled || s.Count != 1 || s.Turn != 0 {
		t.Fatalf("post-forget observe = stalled %v count %d turn %d, want false/1/0", s.Stalled, s.Count, s.Turn)
	}
}

// TestRepeatStallTasksAreIndependent proves two tasks repeating the same hash do not
// contaminate each other's counts.
func TestRepeatStallTasksAreIndependent(t *testing.T) {
	r := NewRepeatStallMonitor()
	r.ObserveToolCall("a", "h")
	r.ObserveToolCall("a", "h")
	// Task b's first observation of the same hash is independent.
	if s := r.ObserveToolCall("b", "h"); s.Stalled || s.Count != 1 {
		t.Fatalf("task b first call = stalled %v count %d, want false/1 (independent of task a)", s.Stalled, s.Count)
	}
	// Task a's third trips; task b is unaffected.
	if s := r.ObserveToolCall("a", "h"); !s.Stalled {
		t.Fatalf("task a third call should stall")
	}
}

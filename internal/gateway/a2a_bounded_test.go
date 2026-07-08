package gateway

import (
	"strconv"
	"testing"
	"time"
)

// TestA2AStoreBoundsTasksAndEvictsTerminalFirst proves the A2A task store no longer grows
// without bound: past a2aMaxTasks it evicts, and it evicts the oldest TERMINAL task before
// ever dropping one still in-flight, so a caller polling a live task is never displaced by a
// flood of completed send-message tasks.
func TestA2AStoreBoundsTasksAndEvictsTerminalFirst(t *testing.T) {
	s := &a2aTaskStore{tasks: make(map[string]*a2aTask)}
	base := time.Unix(1_700_000_000, 0)

	// Fill exactly to capacity with completed (terminal) tasks, oldest first.
	for i := 0; i < a2aMaxTasks; i++ {
		id := "done-" + strconv.Itoa(i)
		s.insertLocked(id, &a2aTask{TaskID: id, State: "completed", CreatedAt: base.Add(time.Duration(i) * time.Second)})
	}
	if len(s.tasks) != a2aMaxTasks {
		t.Fatalf("len = %d, want %d at capacity", len(s.tasks), a2aMaxTasks)
	}

	// One more insert must evict, keeping the map at the cap (never above).
	s.insertLocked("overflow", &a2aTask{TaskID: "overflow", State: "created", CreatedAt: base.Add(time.Hour)})
	if len(s.tasks) != a2aMaxTasks {
		t.Fatalf("len = %d after overflow insert, want it held at %d", len(s.tasks), a2aMaxTasks)
	}
	// The oldest terminal task (done-0) is the one evicted.
	if _, ok := s.tasks["done-0"]; ok {
		t.Fatal("done-0 (oldest terminal) should have been evicted")
	}
	if _, ok := s.tasks["overflow"]; !ok {
		t.Fatal("the newly inserted task should be present")
	}
}

func TestA2AStoreEvictsInFlightOnlyWhenNoTerminalExists(t *testing.T) {
	s := &a2aTaskStore{tasks: make(map[string]*a2aTask)}
	base := time.Unix(1_700_000_000, 0)

	// Capacity of ALL in-flight tasks — the pathological case with no terminal victim.
	for i := 0; i < a2aMaxTasks; i++ {
		id := "run-" + strconv.Itoa(i)
		s.insertLocked(id, &a2aTask{TaskID: id, State: "running", CreatedAt: base.Add(time.Duration(i) * time.Second)})
	}
	s.insertLocked("newest", &a2aTask{TaskID: "newest", State: "running", CreatedAt: base.Add(time.Hour)})

	if len(s.tasks) != a2aMaxTasks {
		t.Fatalf("len = %d, want held at %d", len(s.tasks), a2aMaxTasks)
	}
	// With no terminal task, the oldest overall (run-0) is evicted — never the newest.
	if _, ok := s.tasks["run-0"]; ok {
		t.Fatal("run-0 (oldest overall) should have been evicted when no terminal task exists")
	}
	if _, ok := s.tasks["newest"]; !ok {
		t.Fatal("newest task must be retained")
	}
}

func TestA2ATerminalStateClassification(t *testing.T) {
	terminal := []string{"completed", "canceled", "cancelled", "failed", "error"}
	for _, st := range terminal {
		if !a2aTerminalState(st) {
			t.Errorf("state %q should be terminal", st)
		}
	}
	for _, st := range []string{"created", "running", "pending", ""} {
		if a2aTerminalState(st) {
			t.Errorf("state %q should NOT be terminal", st)
		}
	}
}

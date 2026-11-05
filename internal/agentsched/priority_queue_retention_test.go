package agentsched

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1ms lane=default
func TestPriorityQueueRemovalClearsObsoleteTaskSlots(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, operation string
		index           int
	}{
		{"dequeue head", "dequeue", 0},
		{"remove ID head", "remove", 0},
		{"remove ID middle", "remove", 1},
		{"remove ID tail", "remove", 2},
		{"remove pointer head", "remove-task", 0},
		{"remove pointer middle", "remove-task", 1},
		{"remove pointer tail", "remove-task", 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pq := NewPriorityQueue(3)
			tasks := []*Task{
				{ID: "a", Priority: abi.ThreadPriorityP2Batch},
				{ID: "b", Priority: abi.ThreadPriorityP2Batch},
				{ID: "c", Priority: abi.ThreadPriorityP2Batch},
			}
			for _, task := range tasks {
				if err := pq.Enqueue(task); err != nil {
					t.Fatal(err)
				}
			}
			// Keep the original backing span visible so this tests clearing
			// directly, without relying on GC timing or finalizers.
			backing := pq.buckets[abi.ThreadPriorityP2Batch]
			cleared := len(backing) - 1
			var removed bool
			switch tc.operation {
			case "dequeue":
				var got *Task
				got, removed = pq.Dequeue(true)
				if got != tasks[tc.index] {
					t.Fatalf("dequeued %p, want %p", got, tasks[tc.index])
				}
				cleared = 0
			case "remove":
				removed = pq.Remove(tasks[tc.index].ID)
			case "remove-task":
				removed = pq.RemoveTask(tasks[tc.index])
			default:
				t.Fatalf("unknown fixture operation %q", tc.operation)
			}
			if !removed || pq.Len() != 2 || pq.LenPriority(abi.ThreadPriorityP2Batch) != 2 {
				t.Fatalf("removed=%t total=%d tier=%d", removed, pq.Len(), pq.LenPriority(abi.ThreadPriorityP2Batch))
			}
			want := append(append([]*Task(nil), tasks[:tc.index]...), tasks[tc.index+1:]...)
			got := pq.Candidates(true)
			if len(got) != len(want) {
				t.Fatalf("candidate count=%d, want %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("candidate %d=%p, want %p", i, got[i], want[i])
				}
			}
			if backing[cleared] != nil {
				t.Fatalf("obsolete backing slot %d still references task %q", cleared, backing[cleared].ID)
			}
			for _, task := range want {
				got, ok := pq.Dequeue(true)
				if !ok || got != task {
					t.Fatalf("drain got=%p ok=%t, want %p", got, ok, task)
				}
			}
			for i, task := range backing {
				if task != nil {
					t.Fatalf("drained backing slot %d still references task %q", i, task.ID)
				}
			}
		})
	}
}

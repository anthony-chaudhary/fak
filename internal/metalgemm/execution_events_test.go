package metalgemm

import (
	"sync"
	"testing"
)

func TestExecutionObservationUsesLocalOpaqueIDsAndClonesSnapshots(t *testing.T) {
	observation := newExecutionObservation(ExecutionQ4KGEMV, true)
	observation.record(0xdeadbeef, true, true, true)
	observation.record(0xcafebabe, true, false, false)
	observation.record(0xdeadbeef, true, true, true)

	first, err := observation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []uint64{1, 2, 3}
	for i, event := range first.Events {
		if event.CommandBufferID != wantIDs[i] {
			t.Fatalf("event[%d] ID = %d, want %d", i, event.CommandBufferID, wantIDs[i])
		}
		if event.Operation != ExecutionQ4KGEMV {
			t.Fatalf("event[%d] operation = %q", i, event.Operation)
		}
	}
	first.Events[0].CommandBufferID = 99
	first.Events = append(first.Events, ExecutionEvent{})

	second, err := observation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(second.Events) != 3 || second.Events[0].CommandBufferID != 1 {
		t.Fatalf("snapshot changed through clone: %+v", second.Events)
	}
}

func TestExecutionObservationsAreConcurrentAndIsolated(t *testing.T) {
	const count = 32
	observations := make([]*ExecutionObservation, count)
	var wg sync.WaitGroup
	for i := range observations {
		observations[i] = newExecutionObservation(ExecutionQ8GEMM, true)
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			observations[i].record(uintptr(0x1000+i), true, true, true)
		}(i)
	}
	wg.Wait()

	for i, observation := range observations {
		snapshot, err := observation.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if len(snapshot.Events) != 1 || snapshot.Events[0].CommandBufferID != 1 {
			t.Fatalf("observation[%d] = %+v, want one local ID 1", i, snapshot.Events)
		}
	}
}

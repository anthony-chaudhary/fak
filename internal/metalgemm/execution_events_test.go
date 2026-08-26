package metalgemm

import (
	"strings"
	"sync"
	"testing"
)

func TestExecutionObservationUsesLocalOpaqueIDsAndClonesSnapshots(t *testing.T) {
	observation := newExecutionObservation(ExecutionQ4KGEMV, true)
	observation.record(0xdeadbeef, true, true, true, 1, 0.2, 0.3, true)
	observation.record(0xcafebabe, true, false, false, 1, 0.2, 0.3, true)
	observation.record(0xdeadbeef, true, true, true, 1, 0.2, 0.3, true)

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
			observations[i].record(uintptr(0x1000+i), true, true, true, 1, 0.2, 0.3, true)
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

func TestMixedQ4KQ8ObservationsAreConcurrentAndIsolated(t *testing.T) {
	const eventsPerObservation = 64
	mixedOperation := ExecutionOperation("mixed-q4_k-q8-qkv")
	left := newExecutionObservation(mixedOperation, true)
	right := newExecutionObservation(mixedOperation, true)
	start := make(chan struct{})

	var wg sync.WaitGroup
	for _, tc := range []struct {
		observation *ExecutionObservation
		encoders    int
		gpuMS       float64
	}{
		{observation: left, encoders: 2, gpuMS: 0.25},
		{observation: right, encoders: 7, gpuMS: 0.75},
	} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for i := 0; i < eventsPerObservation; i++ {
				tc.observation.record(uintptr(i+1), true, true, true, tc.encoders, tc.gpuMS, 1, true)
			}
		}()
	}
	close(start)
	wg.Wait()

	for name, tc := range map[string]struct {
		observation *ExecutionObservation
		encoders    int
		gpuMS       float64
	}{
		"left":  {observation: left, encoders: 2, gpuMS: 0.25},
		"right": {observation: right, encoders: 7, gpuMS: 0.75},
	} {
		snapshot, err := tc.observation.Snapshot()
		if err != nil {
			t.Fatalf("%s snapshot: %v", name, err)
		}
		if len(snapshot.Events) != eventsPerObservation {
			t.Fatalf("%s events=%d, want %d", name, len(snapshot.Events), eventsPerObservation)
		}
		for i, event := range snapshot.Events {
			if event.Operation != mixedOperation || event.CommandBufferID != uint64(i+1) || event.Encoders != tc.encoders || event.GPUMilliseconds != tc.gpuMS {
				t.Fatalf("%s event[%d]=%+v, want isolated operation/id/encoders/gpu_ms", name, i, event)
			}
		}
	}
}

func TestExecutionSessionAggregatesAndFailsClosed(t *testing.T) {
	session := NewExecutionSession()
	session.Record(ExecutionSnapshot{Events: []ExecutionEvent{{
		Operation: ExecutionQ4KGEMM, CommandBufferID: 1, Committed: true,
		CompletedWait: true, HostReadback: true, Encoders: 2,
		GPUMilliseconds: 1.25, WaitMilliseconds: 1.5, TimingAvailable: true,
	}}}, nil)
	counters, err := session.Counters()
	if err != nil {
		t.Fatal(err)
	}
	if counters.CommandBuffers != 1 || counters.Encoders != 2 || counters.DispatchMilliseconds != 1.25 || counters.WaitMilliseconds != 1.5 {
		t.Fatalf("counters = %+v", counters)
	}

	incomplete := NewExecutionSession()
	incomplete.Record(ExecutionSnapshot{Events: []ExecutionEvent{{Operation: ExecutionQ8GEMV, Committed: true}}}, nil)
	if _, err := incomplete.Counters(); !IsExecutionCountersIncomplete(err) {
		t.Fatalf("err = %v, want typed incomplete capture", err)
	}
	empty := NewExecutionSession()
	empty.Record(ExecutionSnapshot{}, nil)
	if _, err := empty.Counters(); !IsExecutionCountersIncomplete(err) {
		t.Fatalf("empty err = %v, want typed incomplete capture", err)
	}
}

func TestExecutionReceiptRecomputesDigestAndCounters(t *testing.T) {
	session := NewExecutionSession()
	session.Record(ExecutionSnapshot{Events: []ExecutionEvent{
		{Operation: ExecutionQ4KGEMM, CommandBufferID: 1, Committed: true, CompletedWait: true, HostReadback: true, Encoders: 2, GPUMilliseconds: 1.25, WaitMilliseconds: 1.5, TimingAvailable: true},
		{Operation: ExecutionQ8GEMV, CommandBufferID: 1, Committed: true, CompletedWait: true, HostReadback: true, Encoders: 1, GPUMilliseconds: 0.5, WaitMilliseconds: 0.75, TimingAvailable: true},
	}}, nil)
	receipt, err := session.Receipt()
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateExecutionReceipt(receipt); err != nil {
		t.Fatalf("receipt did not read back: %v", err)
	}
	if receipt.Counters.CommandBuffers != 2 || receipt.Counters.Encoders != 3 || receipt.Counters.DispatchMilliseconds != 1.75 || receipt.Counters.WaitMilliseconds != 2.25 {
		t.Fatalf("receipt counters = %+v", receipt.Counters)
	}

	tamperedEvent := receipt
	tamperedEvent.Events = append([]ExecutionEvent(nil), receipt.Events...)
	tamperedEvent.Events[0].Encoders++
	if err := ValidateExecutionReceipt(tamperedEvent); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("tampered event err = %v, want digest mismatch", err)
	}

	tamperedCounters := receipt
	tamperedCounters.Counters.CommandBuffers++
	if err := ValidateExecutionReceipt(tamperedCounters); err == nil || !strings.Contains(err.Error(), "aggregate mismatch") {
		t.Fatalf("tampered counters err = %v, want aggregate mismatch", err)
	}
}

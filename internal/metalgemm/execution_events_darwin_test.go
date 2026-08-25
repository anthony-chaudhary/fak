//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

func requireCompletedExecution(t *testing.T, observation *ExecutionObservation, operation ExecutionOperation) {
	t.Helper()
	snapshot, err := observation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Events) != 1 {
		t.Fatalf("events = %+v, want exactly one native execution", snapshot.Events)
	}
	event := snapshot.Events[0]
	if event.Operation != operation {
		t.Fatalf("operation = %q, want %q", event.Operation, operation)
	}
	if event.CommandBufferID != 1 {
		t.Fatalf("local command-buffer ID = %d, want 1", event.CommandBufferID)
	}
	if !event.Committed || !event.CompletedWait || !event.HostReadback {
		t.Fatalf("incomplete native lifecycle: %+v", event)
	}
}

func TestExecutionObservationCapturesQuantizedMetalLifecycle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer ResetQ8()

	t.Run("Q4_K", func(t *testing.T) {
		const in, out = 256, 1
		// One zeroed block_q4_K is valid and deterministically produces zero.
		weight := UploadQ4K(make([]byte, 144), out, in)
		if weight == nil {
			t.Fatal("UploadQ4K returned nil")
		}
		defer weight.Release()

		x := make([]float32, in)
		for i := range x {
			x[i] = 1
		}
		y := []float32{float32(math.NaN())}
		observation := NewExecutionObservation(ExecutionQ4KGEMV)
		weight.GEMVWithEvents(x, y, observation)

		if y[0] != 0 {
			t.Fatalf("Q4_K result = %v, want 0", y[0])
		}
		requireCompletedExecution(t, observation, ExecutionQ4KGEMV)
	})

	t.Run("Q8", func(t *testing.T) {
		const in, out = 32, 1
		codes := make([]int8, in)
		xq := make([]int8, in)
		for i := range codes {
			codes[i] = 1
			xq[i] = 1
		}
		weight := UploadQ8(codes, []float32{1}, out, in)
		if weight == nil {
			t.Fatal("UploadQ8 returned nil")
		}

		y := []float32{float32(math.NaN())}
		observation := NewExecutionObservation(ExecutionQ8GEMV)
		weight.GEMVWithEvents(xq, []float32{1}, y, observation)

		if y[0] != in {
			t.Fatalf("Q8 result = %v, want %d", y[0], in)
		}
		requireCompletedExecution(t, observation, ExecutionQ8GEMV)
	})
}

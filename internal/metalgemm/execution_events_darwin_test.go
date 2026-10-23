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
	if event.Encoders <= 0 || !event.TimingAvailable || event.GPUMilliseconds <= 0 || event.WaitMilliseconds < 0 {
		t.Fatalf("missing native counters: %+v", event)
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

// fak-test:runtime integration est=2s lane=optin
func TestExecutionEncoderSpansNativeCapture(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const width = 256
	weight := UploadQ4K(make([]byte, 144), 1, width)
	if weight == nil {
		t.Fatal("Q4K upload")
	}
	defer weight.Release()
	run := func(profile bool) GraphReceipt {
		g, err := BeginProjectionGraph(make([]float32, width), nil, nil, 1, width)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Free()
		if profile {
			if err := g.RequestEncoderTiming(); err != nil {
				t.Fatal(err)
			}
		}
		first, err := g.EncodeQ4K(weight)
		if err != nil {
			t.Fatal(err)
		}
		if err := g.Flush(); err != nil {
			t.Fatal(err)
		}
		second, err := g.EncodeQ4K(weight)
		if err != nil {
			t.Fatal(err)
		}
		values, receipt, err := g.FinishRead(first, second)
		if err != nil {
			t.Fatal(err)
		}
		if len(values) != 2 || len(values[0]) != 1 || len(values[1]) != 1 || values[0][0] != 0 || values[1][0] != 0 {
			t.Fatalf("output = %v", values)
		}
		return receipt
	}
	off, on := run(false), run(true)
	if off.EncoderTiming != nil || off.CommandBuffers != on.CommandBuffers || off.Encoders != on.Encoders || off.HostReadbacks != on.HostReadbacks {
		t.Fatal("profiling changed default lifecycle")
	}
	if on.EncoderTiming == nil {
		t.Fatal("requested capture has no availability receipt")
	}
	if _, err := on.EncoderTiming.EncoderCoverage(on.Encoders); err != nil {
		t.Fatal(err)
	}
	if on.EncoderTiming.Status == "unavailable" {
		if on.EncoderTiming.Reason == "Apple GPU stage-boundary timestamp counters unsupported" {
			t.Skipf("native timestamp capture unsupported: %s", on.EncoderTiming.Reason)
		}
		t.Fatalf("native timestamp capture failed: %s", on.EncoderTiming.Reason)
	}
	if len(on.EncoderTiming.Segments) != 2 {
		t.Fatal("missing native command-buffer segments")
	}
	for _, segment := range on.EncoderTiming.Segments {
		if segment.CapturedEncoders != 1 || len(segment.Timing.Spans) != 1 || segment.Timing.Status != "partial" {
			t.Fatalf("invalid native segment: %+v", segment)
		}
	}
}

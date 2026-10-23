package metalgemm

import (
	"encoding/json"
	"math"
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

// fak-test:runtime fast est=1ms lane=default
func TestExecutionEncoderSpans(t *testing.T) {
	t.Parallel()
	capture := ExecutionEncoderTiming{Status: "complete", ClockDomain: "metal-gpu-counter", WindowEndMilliseconds: 10,
		Spans: []ExecutionEncoderSpan{{1, "norm", 1, 5}, {2, "projection", 3, 8}},
	}
	coverage, err := capture.EncoderCoverage(2)
	if err != nil || !coverage.Available || coverage.AttributedMilliseconds != 7 || coverage.UnattributedMilliseconds != 3 || len(coverage.Covered) != 1 || len(coverage.Uncovered) != 2 {
		t.Fatalf("overlap union = %+v, %v", coverage, err)
	}
	// Encoder order need not be chronological when work overlaps.
	reversed := capture
	reversed.Spans = []ExecutionEncoderSpan{{1, "projection", 3, 8}, {2, "norm", 1, 5}}
	if result, err := reversed.EncoderCoverage(2); err != nil || result.AttributedMilliseconds != 7 {
		t.Fatalf("submission order: %+v, %v", result, err)
	}
	partial := capture
	partial.Status = "partial"
	if _, err := partial.EncoderCoverage(3); err != nil {
		t.Fatal(err)
	}
	for _, status := range []string{"unavailable", "disjoint"} {
		missing := ExecutionEncoderTiming{Status: status, Reason: "no comparable GPU samples"}
		if c, err := missing.EncoderCoverage(2); err != nil || c.Available {
			t.Fatalf("%s = %+v, %v", status, c, err)
		}
	}
	for _, tc := range []struct {
		name   string
		change func(*ExecutionEncoderTiming)
	}{
		{"nonfinite", func(c *ExecutionEncoderTiming) { c.Spans[0].StartMilliseconds = math.NaN() }},
		{"infinite window", func(c *ExecutionEncoderTiming) { c.WindowEndMilliseconds = math.Inf(1) }},
		{"outside window", func(c *ExecutionEncoderTiming) { c.Spans[0].EndMilliseconds = 11 }},
		{"reversed interval", func(c *ExecutionEncoderTiming) { c.Spans[0].EndMilliseconds = 0 }},
		{"duplicate ordinal", func(c *ExecutionEncoderTiming) { c.Spans[1].Ordinal = 1 }},
		{"missing identity", func(c *ExecutionEncoderTiming) { c.Spans[0].Operation = "" }},
		{"host clock", func(c *ExecutionEncoderTiming) { c.ClockDomain = "host-monotonic" }},
		{"false unavailable", func(c *ExecutionEncoderTiming) { c.Status = "unavailable" }},
		{"false partial", func(c *ExecutionEncoderTiming) { c.Status = "partial" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := capture
			c.Spans = append([]ExecutionEncoderSpan(nil), capture.Spans...)
			tc.change(&c)
			if _, err := c.EncoderCoverage(2); err == nil {
				t.Fatal("invalid timing accepted")
			}
		})
	}
	event := ExecutionEvent{Operation: ExecutionQ8GEMV, CommandBufferID: 1, Committed: true, CompletedWait: true, HostReadback: true, Encoders: 2, GPUMilliseconds: 10, TimingAvailable: true, EncoderTiming: &capture}
	session := NewExecutionSession()
	session.Record(ExecutionSnapshot{Events: []ExecutionEvent{event}}, nil)
	capture.Spans[0].Operation = "mutated caller"
	receipt, err := session.Receipt()
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Events[0].EncoderTiming.Spans[0].Operation != "norm" {
		t.Fatal("session retained caller timing alias")
	}
	if err := ValidateExecutionReceipt(receipt); err != nil {
		t.Fatal(err)
	}
	receipt.Events[0].EncoderTiming.Spans[0].Operation = "mutated receipt"
	if err := ValidateExecutionReceipt(receipt); err == nil {
		t.Fatal("timing tampering escaped digest")
	}
	again, err := session.Receipt()
	if err != nil || again.Events[0].EncoderTiming.Spans[0].Operation != "norm" {
		t.Fatalf("receipt retained alias: %v", err)
	}
	event.EncoderTiming = nil
	data, err := json.Marshal(event)
	if err != nil || strings.Contains(string(data), "encoder_timing") {
		t.Fatalf("profile-OFF schema changed: %s, %v", data, err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestExecutionEncoderSpansNativeSegments(t *testing.T) {
	t.Parallel()
	// CPU fixture verifies native readback conversion, not measured GPU evidence.
	raw := []rawEncoderSample{
		{1, 1, 1000000, 3000000, "norm"},
		{1, 2, 2000000, 4000000, "projection"},
		{3, 1, 9000000, 11000000, "attention"},
	}
	if executionEncoderSampleLimit*2*8 > 32*1024 {
		t.Fatal("capture exceeds Apple timestamp buffer ceiling")
	}
	oversized := make([]rawEncoderSample, executionEncoderSampleLimit+1)
	if got := encoderTimingFromSamples(oversized, 1); got.Status != "unavailable" {
		t.Fatal("oversized native readback accepted")
	}
	capture := encoderTimingFromSamples(raw, 3)
	if capture.Status != "segmented" || len(capture.Segments) != 3 {
		t.Fatalf("segments = %+v", capture)
	}
	if total, err := capture.EncoderCoverage(0); err != nil || total.Available {
		t.Fatalf("cross-buffer timeline was invented: %+v, %v", total, err)
	}
	first := capture.Segments[0]
	coverage, err := first.Timing.EncoderCoverage(first.CapturedEncoders)
	if err != nil || coverage.AttributedMilliseconds != 3 || first.Timing.Status != "partial" || !first.Timing.UnmeasuredPrefix || !first.Timing.UnmeasuredSuffix || first.Timing.UncoveredEncoderScope == "" {
		t.Fatalf("partial coverage = %+v, %+v, %v", first, coverage, err)
	}
	if capture.Segments[1].Timing.Status != "unavailable" || capture.Segments[1].CapturedEncoders != 0 {
		t.Fatal("unsampled segment fabricated")
	}
	clone := capture.Clone()
	clone.Segments[0].Timing.Spans[0].Operation = "mutated"
	if capture.Segments[0].Timing.Spans[0].Operation != "norm" {
		t.Fatal("segment clone aliases original")
	}
	clone = capture.Clone()
	clone.Segments[0].Timing.UnmeasuredPrefix = false
	if _, err := clone.EncoderCoverage(0); err == nil {
		t.Fatal("unknown prefix omission accepted")
	}
	clone = capture.Clone()
	clone.Segments[1].CommandBufferID = 1
	if _, err := clone.EncoderCoverage(0); err == nil {
		t.Fatal("duplicate segment accepted")
	}
	for _, invalid := range [][]rawEncoderSample{
		{{1, 1, 0, 1, "zero"}}, {{1, 1, 2, 1, "reversed"}}, {{1, 1, 1, ^uint64(0), "error"}},
		{{2, 1, 1, 2, "a"}, {1, 1, 3, 4, "b"}}, {{1, 2, 1, 2, "ordinal"}},
	} {
		if got := encoderTimingFromSamples(invalid, 3); got.Status != "unavailable" {
			t.Fatalf("invalid samples accepted: %+v", got)
		}
	}
}

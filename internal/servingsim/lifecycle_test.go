package servingsim

import (
	"bytes"
	"container/heap"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
)

// TestEngineLifecycleSequentialRuns verifies that an Engine instance can be reused
// across multiple sequential Run invocations without state leakage between runs.
func TestEngineLifecycleSequentialRuns(t *testing.T) {
	config := SimulatorConfig{
		MaxBatchSize:     4,
		MaxTokensPerStep: 256,
		KVBlockTokens:    16,
		TotalKVBlocks:    256,
		Seed:             101,
		EnableTrace:      true,
	}
	hw := DefaultHardwareLatencyTable()

	engine, err := NewServingSimulator(config, hw)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	workloadA := []RequestState{
		{ID: "runA-1", ArrivalTimeMS: 0, PromptTokens: 64, OutputTarget: 16},
		{ID: "runA-2", ArrivalTimeMS: 10, PromptTokens: 32, OutputTarget: 8},
	}
	metricsA, err := engine.Run(workloadA)
	if err != nil {
		t.Fatalf("run A failed: %v", err)
	}
	if metricsA.TotalRequests != 2 {
		t.Errorf("expected 2 requests in run A, got %d", metricsA.TotalRequests)
	}

	workloadB := []RequestState{
		{ID: "runB-1", ArrivalTimeMS: 0, PromptTokens: 128, OutputTarget: 32},
	}
	metricsB, err := engine.Run(workloadB)
	if err != nil {
		t.Fatalf("run B failed: %v", err)
	}
	if metricsB.TotalRequests != 1 {
		t.Errorf("expected 1 request in run B, got %d", metricsB.TotalRequests)
	}
	if metricsB.CompletedRequests[0].ID != "runB-1" {
		t.Errorf("expected request runB-1, got %s", metricsB.CompletedRequests[0].ID)
	}

	// Verify peak KV blocks for run B reflected only run B's allocation, not cumulative
	expectedMaxBlocksB := (128 + 32 + 15) / 16
	if metricsB.PeakKVBlocksUsed > expectedMaxBlocksB {
		t.Errorf("run B peak KV blocks (%d) exceeded single request requirement (%d)",
			metricsB.PeakKVBlocksUsed, expectedMaxBlocksB)
	}
}

// TestEngineLifecycleUnsortedArrivals verifies that Engine.Run sorts requests
// by ArrivalTimeMS before starting the simulation loop.
func TestEngineLifecycleUnsortedArrivals(t *testing.T) {
	config := SimulatorConfig{
		MaxBatchSize:     2,
		MaxTokensPerStep: 512,
		KVBlockTokens:    16,
		TotalKVBlocks:    512,
		Seed:             42,
	}
	hw := DefaultHardwareLatencyTable()

	// Arriving out of chronological order
	requests := []RequestState{
		{ID: "req-late", ArrivalTimeMS: 50.0, PromptTokens: 32, OutputTarget: 10},
		{ID: "req-early", ArrivalTimeMS: 0.0, PromptTokens: 32, OutputTarget: 10},
		{ID: "req-mid", ArrivalTimeMS: 20.0, PromptTokens: 32, OutputTarget: 10},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("simulation failed: %v", err)
	}

	if len(metrics.CompletedRequests) != 3 {
		t.Fatalf("expected 3 completed requests, got %d", len(metrics.CompletedRequests))
	}

	// Verify completion times respect arrival ordering
	for _, req := range metrics.CompletedRequests {
		if req.FirstTokenTimeMS < req.ArrivalTimeMS {
			t.Errorf("request %s had first token %f before arrival %f",
				req.ID, req.FirstTokenTimeMS, req.ArrivalTimeMS)
		}
	}
}

// TestEngineLifecycleConcurrentArrivals verifies proper handling of requests
// arriving at the exact same millisecond timestamp.
func TestEngineLifecycleConcurrentArrivals(t *testing.T) {
	config := SimulatorConfig{
		MaxBatchSize:     4,
		MaxTokensPerStep: 1024,
		KVBlockTokens:    16,
		TotalKVBlocks:    1024,
		Seed:             777,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "c-1", ArrivalTimeMS: 15.0, PromptTokens: 48, OutputTarget: 16},
		{ID: "c-2", ArrivalTimeMS: 15.0, PromptTokens: 48, OutputTarget: 16},
		{ID: "c-3", ArrivalTimeMS: 15.0, PromptTokens: 48, OutputTarget: 16},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("concurrent arrivals run failed: %v", err)
	}

	if metrics.TotalRequests != 3 {
		t.Fatalf("expected 3 completed requests, got %d", metrics.TotalRequests)
	}

	for _, req := range metrics.CompletedRequests {
		if req.ArrivalTimeMS != 15.0 {
			t.Errorf("expected arrival 15.0ms, got %f", req.ArrivalTimeMS)
		}
		if req.CompletionTimeMS <= 15.0 {
			t.Errorf("expected completion after arrival, got %f", req.CompletionTimeMS)
		}
	}
}

// TestEngineSchedulingBatchSaturation verifies that requests arriving beyond MaxBatchSize
// are queued and wait until active requests complete.
func TestEngineSchedulingBatchSaturation(t *testing.T) {
	config := SimulatorConfig{
		MaxBatchSize:     2,
		MaxTokensPerStep: 512,
		KVBlockTokens:    16,
		TotalKVBlocks:    1024,
		Seed:             123,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "batch-1", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 30},
		{ID: "batch-2", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 30},
		{ID: "batch-3", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 10},
		{ID: "batch-4", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 10},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("batch saturation test failed: %v", err)
	}

	if metrics.TotalRequests != 4 {
		t.Fatalf("expected 4 completed requests, got %d", metrics.TotalRequests)
	}

	// At least 2 requests must have experienced non-zero queue time
	queuedCount := 0
	for _, req := range metrics.CompletedRequests {
		if req.QueueTimeMS > 0 {
			queuedCount++
		}
	}
	if queuedCount < 2 {
		t.Errorf("expected at least 2 requests to experience queue wait, got %d", queuedCount)
	}
}

// TestEngineSchedulingPreemptionUnderMemoryExhaustion verifies decode preemption
// when KV cache space is exhausted.
func TestEngineSchedulingPreemptionUnderMemoryExhaustion(t *testing.T) {
	// Total KV blocks is tightly restricted to 6 blocks (96 tokens).
	// Two requests of 48 tokens each (3 blocks each).
	// As they decode, they will demand additional blocks, triggering preemption logic.
	config := SimulatorConfig{
		MaxBatchSize:     2,
		MaxTokensPerStep: 128,
		KVBlockTokens:    16,
		TotalKVBlocks:    6,
		Seed:             42,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "preempt-1", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 32},
		{ID: "preempt-2", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 32},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("preemption run failed: %v", err)
	}

	if metrics.TotalRequests != 2 {
		t.Fatalf("expected 2 completed requests after preemption recovery, got %d", metrics.TotalRequests)
	}
	if metrics.PeakKVBlocksUsed > config.TotalKVBlocks {
		t.Errorf("peak blocks (%d) exceeded total blocks (%d)", metrics.PeakKVBlocksUsed, config.TotalKVBlocks)
	}
}

// TestEngineSchedulingTokenBudgetDepletion tests chunked prefill when MaxTokensPerStep
// limits the number of tokens scheduled per step.
func TestEngineSchedulingTokenBudgetDepletion(t *testing.T) {
	config := SimulatorConfig{
		MaxBatchSize:     2,
		MaxTokensPerStep: 64, // Small token budget
		KVBlockTokens:    16,
		TotalKVBlocks:    256,
		Seed:             42,
		EnableTrace:      true,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "budget-req", ArrivalTimeMS: 0, PromptTokens: 192, OutputTarget: 5},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("token budget run failed: %v", err)
	}

	// 192 tokens / 64 tokens per step = 3 prefill steps minimum
	prefillSteps := 0
	for _, ev := range metrics.TraceEvents {
		if ev.Name == "step_prefill" {
			prefillSteps++
		}
	}
	if prefillSteps < 3 {
		t.Errorf("expected at least 3 prefill chunks for 192 prompt tokens with budget 64, got %d", prefillSteps)
	}
}

// TestTraceSimulationCollectorLifecycle verifies concurrency, event collection, snapshotting, and reset.
func TestTraceSimulationCollectorLifecycle(t *testing.T) {
	collector := NewTraceCollector()
	if collector == nil {
		t.Fatal("expected non-nil collector from NewTraceCollector")
	}

	// Concurrent writing test
	const goroutines = 8
	const eventsPerGoroutine = 25
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(gid int) {
			defer wg.Done()
			for i := 0; i < eventsPerGoroutine; i++ {
				collector.RecordStep(
					fmt.Sprintf("step-%d-%d", gid, i),
					"test",
					float64(i)*10.0,
					5.0,
					gid+1,
					1,
					map[string]any{"index": i},
				)
				collector.RecordCounter("counter", "metric", float64(i)*10.0, gid+1, 1, map[string]any{"val": i})
				collector.RecordInstant("instant", "marker", float64(i)*10.0, gid+1, 1, nil)
			}
		}(g)
	}
	wg.Wait()

	events := collector.Events()
	expectedCount := goroutines * eventsPerGoroutine * 3
	if len(events) != expectedCount {
		t.Errorf("expected %d events, got %d", expectedCount, len(events))
	}

	// Verify Events() returns a snapshot copy that is isolated
	collector.Reset()
	if len(collector.Events()) != 0 {
		t.Errorf("expected 0 events after Reset, got %d", len(collector.Events()))
	}
	if len(events) != expectedCount {
		t.Errorf("snapshot was modified: expected %d events, got %d", expectedCount, len(events))
	}
}

// TestTraceSimulationJSONLMalformed tests fail-closed error handling on invalid JSONL workloads.
func TestTraceSimulationJSONLMalformed(t *testing.T) {
	// Test negative arrival time
	negArrival := `{"id": "req-1", "arrival_time_ms": -5.0, "prompt_tokens": 10, "output_tokens": 10}`
	if _, err := ReadTraceJSONL(strings.NewReader(negArrival)); err == nil {
		t.Error("expected error for negative arrival time")
	}

	// Test zero prompt tokens
	zeroPrompt := `{"id": "req-2", "arrival_time_ms": 0.0, "prompt_tokens": 0, "output_tokens": 10}`
	if _, err := ReadTraceJSONL(strings.NewReader(zeroPrompt)); err == nil {
		t.Error("expected error for zero prompt tokens")
	}

	// Test zero output tokens
	zeroOutput := `{"id": "req-3", "arrival_time_ms": 0.0, "prompt_tokens": 10, "output_tokens": 0}`
	if _, err := ReadTraceJSONL(strings.NewReader(zeroOutput)); err == nil {
		t.Error("expected error for zero output tokens")
	}

	// Test corrupt JSON syntax
	corruptJSON := `{"id": "req-4", "arrival_time_ms": 0.0, "prompt_tokens": `
	if _, err := ReadTraceJSONL(strings.NewReader(corruptJSON)); err == nil {
		t.Error("expected error for malformed JSON")
	}

	// Test comments and blank lines are cleanly ignored
	validWithComments := `
# Comment line
// Another comment
{"request_id": "fallback-id", "arrival_time": 10.0, "prompt_tokens": 20, "output_target": 15}

`
	parsed, err := ReadTraceJSONL(strings.NewReader(validWithComments))
	if err != nil {
		t.Fatalf("unexpected error parsing valid JSONL with comments: %v", err)
	}
	if len(parsed) != 1 {
		t.Fatalf("expected 1 parsed request, got %d", len(parsed))
	}
	if parsed[0].ID != "fallback-id" || parsed[0].PromptTokens != 20 || parsed[0].OutputTarget != 15 {
		t.Errorf("parsed request mismatch: %+v", parsed[0])
	}
}

// TestTraceSimulationExportVariations verifies Chrome trace formatting and export helpers.
func TestTraceSimulationExportVariations(t *testing.T) {
	events := []TraceEvent{
		{Cat: "gpu", PID: 1, TID: 1, TS: 1000.0, Ph: "X", Name: "step_prefill", Dur: 500.0},
		{Cat: "mem", PID: 1, TID: 1, TS: 1500.0, Ph: "C", Name: "KVBlocks", Args: map[string]any{"used": 4}},
		{Cat: "req", PID: 1, TID: 2, TS: 1000.0, Ph: "i", Name: "arrival:r1"},
	}

	var bufChrome bytes.Buffer
	if err := ExportTrace(&bufChrome, events); err != nil {
		t.Fatalf("ExportTrace failed: %v", err)
	}

	var container ChromeTraceContainer
	if err := json.Unmarshal(bufChrome.Bytes(), &container); err != nil {
		t.Fatalf("invalid Chrome trace container JSON: %v", err)
	}
	if len(container.TraceEvents) != 3 {
		t.Errorf("expected 3 trace events, got %d", len(container.TraceEvents))
	}
	if container.DisplayTimeUnit != "ms" {
		t.Errorf("expected displayTimeUnit 'ms', got %s", container.DisplayTimeUnit)
	}
}

// TestHardwareLatencyTableZeroAndNegativeBounds verifies fail-closed zero returns on non-positive dimensions.
func TestHardwareLatencyTableZeroAndNegativeBounds(t *testing.T) {
	hw := DefaultHardwareLatencyTable()

	if lat := hw.PrefillChunkLatency(0, 4); lat != 0 {
		t.Errorf("expected 0 latency for 0 tokens, got %f", lat)
	}
	if lat := hw.PrefillChunkLatency(64, -1); lat != 0 {
		t.Errorf("expected 0 latency for negative batch, got %f", lat)
	}
	if lat := hw.DecodeStepLatency(0, 2); lat != 0 {
		t.Errorf("expected 0 latency for 0 batch decode, got %f", lat)
	}
	if lat := hw.StepLatency(0, 0, 0, 0); lat != 0 {
		t.Errorf("expected 0 latency for empty step, got %f", lat)
	}
}

// TestEventQueueHeapInvariants verifies the container/heap behavior and ordering of EventQueue.
func TestEventQueueHeapInvariants(t *testing.T) {
	var eq EventQueue
	heap.Init(&eq)

	e1 := &SimEvent{TimeMS: 10.0, Seq: 1}
	e2 := &SimEvent{TimeMS: 5.0, Seq: 2}
	e3 := &SimEvent{TimeMS: 10.0, Seq: 0} // same time as e1, lower Seq -> must pop before e1

	heap.Push(&eq, e1)
	heap.Push(&eq, e2)
	heap.Push(&eq, e3)

	if eq.Len() != 3 {
		t.Fatalf("expected queue length 3, got %d", eq.Len())
	}

	p1 := heap.Pop(&eq).(*SimEvent)
	if p1.TimeMS != 5.0 {
		t.Errorf("expected lowest TimeMS (5.0), got %f", p1.TimeMS)
	}

	p2 := heap.Pop(&eq).(*SimEvent)
	if p2.TimeMS != 10.0 || p2.Seq != 0 {
		t.Errorf("expected TimeMS 10.0 with Seq 0, got TimeMS %f Seq %d", p2.TimeMS, p2.Seq)
	}

	p3 := heap.Pop(&eq).(*SimEvent)
	if p3.TimeMS != 10.0 || p3.Seq != 1 {
		t.Errorf("expected TimeMS 10.0 with Seq 1, got TimeMS %f Seq %d", p3.TimeMS, p3.Seq)
	}

	if eq.Len() != 0 {
		t.Errorf("expected empty queue, got len %d", eq.Len())
	}
}

// BenchmarkServingSimEngine benchmarks standard continuous batching simulation of 16 requests.
func BenchmarkServingSimEngine(b *testing.B) {
	config := SimulatorConfig{
		MaxBatchSize:     8,
		MaxTokensPerStep: 512,
		KVBlockTokens:    16,
		TotalKVBlocks:    2048,
		Seed:             42,
	}
	hw := DefaultHardwareLatencyTable()

	requests := make([]RequestState, 16)
	for i := range requests {
		requests[i] = RequestState{
			ID:            fmt.Sprintf("bench-req-%d", i),
			ArrivalTimeMS: float64(i * 10),
			PromptTokens:  128,
			OutputTarget:  64,
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := Run(config, requests, hw)
		if err != nil {
			b.Fatalf("benchmark iteration failed: %v", err)
		}
	}
}

// BenchmarkServingSimChunkedPrefill benchmarks simulation with heavy chunked prefill workloads.
func BenchmarkServingSimChunkedPrefill(b *testing.B) {
	config := SimulatorConfig{
		MaxBatchSize:     4,
		MaxTokensPerStep: 128, // forces fine-grained chunks
		KVBlockTokens:    16,
		TotalKVBlocks:    4096,
		Seed:             42,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "heavy-1", ArrivalTimeMS: 0, PromptTokens: 1024, OutputTarget: 32},
		{ID: "heavy-2", ArrivalTimeMS: 50, PromptTokens: 1024, OutputTarget: 32},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := Run(config, requests, hw)
		if err != nil {
			b.Fatalf("chunked prefill benchmark iteration failed: %v", err)
		}
	}
}

// BenchmarkServingSimSpeculative benchmarks simulation with speculative decoding enabled.
func BenchmarkServingSimSpeculative(b *testing.B) {
	config := SimulatorConfig{
		MaxBatchSize:       4,
		MaxTokensPerStep:   512,
		KVBlockTokens:      16,
		TotalKVBlocks:      1024,
		SpeculativeHorizon: 4,
		AcceptanceRate:     0.75,
		SpeculativeMode:    SpeculativeModeDeterministic,
		Seed:               42,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "spec-1", ArrivalTimeMS: 0, PromptTokens: 64, OutputTarget: 128},
		{ID: "spec-2", ArrivalTimeMS: 10, PromptTokens: 64, OutputTarget: 128},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := Run(config, requests, hw)
		if err != nil {
			b.Fatalf("speculative benchmark iteration failed: %v", err)
		}
	}
}

// BenchmarkServingSimTraceCollection benchmarks simulation with Perfetto/Chrome trace generation.
func BenchmarkServingSimTraceCollection(b *testing.B) {
	config := SimulatorConfig{
		MaxBatchSize:     4,
		MaxTokensPerStep: 256,
		KVBlockTokens:    16,
		TotalKVBlocks:    1024,
		Seed:             42,
		EnableTrace:      true,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "trace-1", ArrivalTimeMS: 0, PromptTokens: 128, OutputTarget: 32},
		{ID: "trace-2", ArrivalTimeMS: 15, PromptTokens: 64, OutputTarget: 32},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		metrics, err := Run(config, requests, hw)
		if err != nil {
			b.Fatalf("trace benchmark iteration failed: %v", err)
		}
		if len(metrics.TraceEvents) == 0 {
			b.Fatal("expected non-empty trace events")
		}
	}
}

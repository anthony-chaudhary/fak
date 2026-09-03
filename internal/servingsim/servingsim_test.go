package servingsim

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestBasicSimulation(t *testing.T) {
	config := SimulatorConfig{
		MaxBatchSize:     4,
		MaxTokensPerStep: 512,
		KVBlockTokens:    16,
		TotalKVBlocks:    1024,
		Seed:             42,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "req-1", ArrivalTimeMS: 0, PromptTokens: 64, OutputTarget: 32},
		{ID: "req-2", ArrivalTimeMS: 5, PromptTokens: 128, OutputTarget: 16},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("unexpected simulation error: %v", err)
	}

	if metrics.TotalRequests != 2 {
		t.Errorf("expected 2 total requests, got %d", metrics.TotalRequests)
	}
	if metrics.SimulatedDurationMS <= 0 {
		t.Errorf("expected positive simulated duration, got %f", metrics.SimulatedDurationMS)
	}
	if metrics.RequestThroughput <= 0 {
		t.Errorf("expected positive request throughput, got %f", metrics.RequestThroughput)
	}
	if metrics.TokenThroughput <= 0 {
		t.Errorf("expected positive token throughput, got %f", metrics.TokenThroughput)
	}
	if metrics.TTFT.P50 <= 0 {
		t.Errorf("expected positive TTFT P50, got %f", metrics.TTFT.P50)
	}
	if metrics.TPOT.P50 <= 0 {
		t.Errorf("expected positive TPOT P50, got %f", metrics.TPOT.P50)
	}
	if metrics.PeakKVBlocksUsed <= 0 {
		t.Errorf("expected positive peak KV blocks used, got %d", metrics.PeakKVBlocksUsed)
	}
	if metrics.KVBlockUtilization <= 0 || metrics.KVBlockUtilization > 1.0 {
		t.Errorf("expected KVBlockUtilization in (0, 1], got %f", metrics.KVBlockUtilization)
	}
}

func TestServingSimDeterministicReplay(t *testing.T) {
	config := SimulatorConfig{
		MaxBatchSize:       4,
		MaxTokensPerStep:   256,
		KVBlockTokens:      16,
		TotalKVBlocks:      512,
		SpeculativeHorizon: 4,
		AcceptanceRate:     0.75,
		SpeculativeMode:    SpeculativeModePrefix,
		Seed:               12345,
	}
	hw := DefaultHardwareLatencyTable()
	requests := []RequestState{
		{ID: "req-1", ArrivalTimeMS: 0, PromptTokens: 100, OutputTarget: 50},
		{ID: "req-2", ArrivalTimeMS: 10, PromptTokens: 150, OutputTarget: 40},
		{ID: "req-3", ArrivalTimeMS: 20, PromptTokens: 80, OutputTarget: 60},
		{ID: "req-4", ArrivalTimeMS: 30, PromptTokens: 200, OutputTarget: 30},
	}

	m1, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("first run failed: %v", err)
	}
	m2, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("second run failed: %v", err)
	}

	// Verify identical seed yields bit-identical metrics (P50/P90/P99 TTFT, TPOT)
	if m1.TTFT.P50 != m2.TTFT.P50 || m1.TTFT.P90 != m2.TTFT.P90 || m1.TTFT.P99 != m2.TTFT.P99 {
		t.Fatalf("TTFT mismatch: %+v vs %+v", m1.TTFT, m2.TTFT)
	}
	if m1.TPOT.P50 != m2.TPOT.P50 || m1.TPOT.P90 != m2.TPOT.P90 || m1.TPOT.P99 != m2.TPOT.P99 {
		t.Fatalf("TPOT mismatch: %+v vs %+v", m1.TPOT, m2.TPOT)
	}
	if m1.SimulatedDurationMS != m2.SimulatedDurationMS {
		t.Fatalf("duration mismatch: %f vs %f", m1.SimulatedDurationMS, m2.SimulatedDurationMS)
	}
	if m1.TokenThroughput != m2.TokenThroughput {
		t.Fatalf("token throughput mismatch: %f vs %f", m1.TokenThroughput, m2.TokenThroughput)
	}
	if m1.RequestThroughput != m2.RequestThroughput {
		t.Fatalf("request throughput mismatch: %f vs %f", m1.RequestThroughput, m2.RequestThroughput)
	}
	if len(m1.CompletedRequests) != len(m2.CompletedRequests) {
		t.Fatalf("completed count mismatch: %d vs %d", len(m1.CompletedRequests), len(m2.CompletedRequests))
	}
	for i := range m1.CompletedRequests {
		r1 := m1.CompletedRequests[i]
		r2 := m2.CompletedRequests[i]
		if r1.FirstTokenTimeMS != r2.FirstTokenTimeMS || r1.CompletionTimeMS != r2.CompletionTimeMS ||
			r1.TokensGenerated != r2.TokensGenerated || r1.SpecAccepted != r2.SpecAccepted ||
			r1.QueueTimeMS != r2.QueueTimeMS {
			t.Fatalf("request %d mismatch: %+v vs %+v", i, r1, r2)
		}
	}
}

func TestServingSimChunkedPrefill(t *testing.T) {
	// PromptTokens = 500, MaxTokensPerStep = 128 -> should require 4 prefill chunks
	config := SimulatorConfig{
		MaxBatchSize:     1,
		MaxTokensPerStep: 128,
		KVBlockTokens:    16,
		TotalKVBlocks:    512,
		Seed:             42,
		EnableTrace:      true,
	}

	hw := HardwareLatencyTable{
		BasePrefillMS:     2.0,
		PerTokenPrefillMS: 0.01,
		BaseDecodeMS:      5.0,
	}

	requests := []RequestState{
		{ID: "chunked-req", ArrivalTimeMS: 0, PromptTokens: 500, OutputTarget: 10},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if metrics.TotalRequests != 1 {
		t.Fatalf("expected 1 completed request, got %d", metrics.TotalRequests)
	}

	req := metrics.CompletedRequests[0]
	if req.PrefillComputed != 500 {
		t.Errorf("expected 500 prefill tokens computed, got %d", req.PrefillComputed)
	}
	if req.TokensGenerated != 10 {
		t.Errorf("expected 10 tokens generated, got %d", req.TokensGenerated)
	}

	// Verify trace contains multiple prefill steps over time
	prefillStepCount := 0
	var lastTS float64 = -1
	for _, ev := range metrics.TraceEvents {
		if ev.Name == "step_prefill" {
			prefillStepCount++
			if lastTS >= 0 && ev.TS <= lastTS {
				t.Errorf("expected monotonically increasing step timestamps, got %f then %f", lastTS, ev.TS)
			}
			lastTS = ev.TS
		}
	}
	// 500 / 128 = 3.9 -> 4 prefill steps
	if prefillStepCount != 4 {
		t.Errorf("expected 4 prefill steps for chunked prefill, got %d", prefillStepCount)
	}
}

func TestChunkedPrefill(t *testing.T) {
	TestServingSimChunkedPrefill(t)
}

func TestSpeculativeDecoding(t *testing.T) {
	baseConfig := SimulatorConfig{
		MaxBatchSize:       2,
		MaxTokensPerStep:   512,
		KVBlockTokens:      16,
		TotalKVBlocks:      1024,
		SpeculativeHorizon: 4,
		AcceptanceRate:     0.8,
		SpeculativeMode:    SpeculativeModeDeterministic,
		Seed:               123,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "spec-1", ArrivalTimeMS: 0, PromptTokens: 64, OutputTarget: 50},
	}

	metrics, err := Run(baseConfig, requests, hw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if metrics.SpeculativeYield <= 0 || metrics.SpeculativeYield > 1.0 {
		t.Errorf("expected SpeculativeYield in (0, 1], got %f", metrics.SpeculativeYield)
	}
	if metrics.SpeculativeWaste < 0 || metrics.SpeculativeWaste >= 1.0 {
		t.Errorf("expected SpeculativeWaste in [0, 1), got %f", metrics.SpeculativeWaste)
	}
	if math.Abs((metrics.SpeculativeYield+metrics.SpeculativeWaste)-1.0) > 1e-6 {
		t.Errorf("expected yield + waste == 1.0, got %f + %f", metrics.SpeculativeYield, metrics.SpeculativeWaste)
	}

	req := metrics.CompletedRequests[0]
	if req.SpecProposed <= 0 {
		t.Errorf("expected positive SpecProposed, got %d", req.SpecProposed)
	}
	if req.SpecAccepted <= 0 {
		t.Errorf("expected positive SpecAccepted, got %d", req.SpecAccepted)
	}
}

func TestSpeculativeModes(t *testing.T) {
	modes := []SpeculativeMode{
		SpeculativeModePrefix,
		SpeculativeModeBinomial,
		SpeculativeModePoisson,
		SpeculativeModeDeterministic,
	}

	for _, mode := range modes {
		t.Run(string(mode), func(t *testing.T) {
			config := SimulatorConfig{
				MaxBatchSize:       2,
				MaxTokensPerStep:   512,
				KVBlockTokens:      16,
				TotalKVBlocks:      512,
				SpeculativeHorizon: 3,
				AcceptanceRate:     0.75,
				SpeculativeMode:    mode,
				Seed:               999,
				PositionalAlphaFunc: func(pos int) float64 {
					return 0.9 - float64(pos)*0.1
				},
			}
			hw := DefaultHardwareLatencyTable()
			requests := []RequestState{
				{ID: "req-" + string(mode), ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 30},
			}

			metrics, err := Run(config, requests, hw)
			if err != nil {
				t.Fatalf("mode %s failed: %v", mode, err)
			}
			if metrics.TotalRequests != 1 {
				t.Errorf("expected 1 completed request, got %d", metrics.TotalRequests)
			}
			if metrics.SpeculativeYield <= 0 {
				t.Errorf("expected positive yield for mode %s, got %f", mode, metrics.SpeculativeYield)
			}
		})
	}
}

func TestServingSimKVCacheConstraints(t *testing.T) {
	// TotalKVBlocks only enough for ~1 request at a time
	// Prompt 64 tokens + 64 output tokens = 128 tokens -> 8 blocks
	config := SimulatorConfig{
		MaxBatchSize:     4,
		MaxTokensPerStep: 512,
		KVBlockTokens:    16,
		TotalKVBlocks:    10, // only 10 blocks available!
		Seed:             42,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "req-1", ArrivalTimeMS: 0, PromptTokens: 64, OutputTarget: 64},
		{ID: "req-2", ArrivalTimeMS: 0, PromptTokens: 64, OutputTarget: 64},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if metrics.TotalRequests != 2 {
		t.Fatalf("expected 2 requests completed, got %d", metrics.TotalRequests)
	}

	// req-2 should have been forced to wait in the queue until req-1 completed because of KV block limit!
	req1 := metrics.CompletedRequests[0]
	req2 := metrics.CompletedRequests[1]
	if req2.QueueTimeMS <= 0 {
		t.Errorf("expected req-2 to experience queue time due to KV cache constraint, got %f", req2.QueueTimeMS)
	}
	if req2.FirstTokenTimeMS < req1.CompletionTimeMS {
		t.Errorf("expected req-2 first token (%f) >= req-1 completion time (%f)", req2.FirstTokenTimeMS, req1.CompletionTimeMS)
	}
	if metrics.PeakKVBlocksUsed > config.TotalKVBlocks {
		t.Errorf("peak KV blocks %d exceeded total KV blocks %d", metrics.PeakKVBlocksUsed, config.TotalKVBlocks)
	}
}

func TestKVCacheAdmissionConstraint(t *testing.T) {
	TestServingSimKVCacheConstraints(t)
}

func TestTraceJSONLRoundtrip(t *testing.T) {
	inputJSONL := `{"id": "trace-1", "arrival_time_ms": 12.5, "prompt_tokens": 100, "output_tokens": 50}
{"id": "trace-2", "arrival_time_ms": 45.0, "prompt_tokens": 256, "output_tokens": 128}
`

	requests, err := ReadTraceJSONL(strings.NewReader(inputJSONL))
	if err != nil {
		t.Fatalf("failed to read JSONL: %v", err)
	}

	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}
	if requests[0].ID != "trace-1" || requests[0].ArrivalTimeMS != 12.5 || requests[0].PromptTokens != 100 || requests[0].OutputTarget != 50 {
		t.Errorf("mismatched req 0: %+v", requests[0])
	}
	if requests[1].ID != "trace-2" || requests[1].ArrivalTimeMS != 45.0 || requests[1].PromptTokens != 256 || requests[1].OutputTarget != 128 {
		t.Errorf("mismatched req 1: %+v", requests[1])
	}

	var buf bytes.Buffer
	if err := WriteTraceJSONL(&buf, requests); err != nil {
		t.Fatalf("failed to write JSONL: %v", err)
	}

	roundtripReqs, err := ReadTraceJSONL(&buf)
	if err != nil {
		t.Fatalf("failed to read written JSONL: %v", err)
	}
	if len(roundtripReqs) != 2 {
		t.Fatalf("expected 2 requests in roundtrip, got %d", len(roundtripReqs))
	}
	if roundtripReqs[0].ID != requests[0].ID || roundtripReqs[0].PromptTokens != requests[0].PromptTokens {
		t.Errorf("roundtrip mismatch: %+v vs %+v", roundtripReqs[0], requests[0])
	}
}

func TestServingSimChromeTraceExport(t *testing.T) {
	config := SimulatorConfig{
		MaxBatchSize:     2,
		MaxTokensPerStep: 256,
		KVBlockTokens:    16,
		TotalKVBlocks:    512,
		EnableTrace:      true,
		Seed:             42,
	}
	hw := DefaultHardwareLatencyTable()
	requests := []RequestState{
		{ID: "r1", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 16},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(metrics.TraceEvents) == 0 {
		t.Fatal("expected trace events when EnableTrace is true")
	}

	// Verify Chrome trace JSON container export
	var buf bytes.Buffer
	if err := ExportChromeTrace(&buf, metrics.TraceEvents); err != nil {
		t.Fatalf("failed to export Chrome trace: %v", err)
	}

	var parsed ChromeTraceContainer
	if err := json.Unmarshal(buf.Bytes(), &parsed); err != nil {
		t.Fatalf("invalid Chrome trace JSON: %v", err)
	}
	if len(parsed.TraceEvents) == 0 {
		t.Errorf("expected parsed traceEvents not to be empty")
	}

	// Verify required fields in events: cat, pid, tid, ts, ph, name, args
	for i, ev := range parsed.TraceEvents {
		if ev.Name == "" {
			t.Errorf("event %d missing Name", i)
		}
		if ev.Cat == "" {
			t.Errorf("event %d missing Cat", i)
		}
		if ev.Ph == "" {
			t.Errorf("event %d missing Ph", i)
		}
		if ev.PID <= 0 {
			t.Errorf("event %d expected positive PID, got %d", i, ev.PID)
		}
		if ev.TID <= 0 {
			t.Errorf("event %d expected positive TID, got %d", i, ev.TID)
		}
		if ev.TS < 0 {
			t.Errorf("event %d negative TS %f", i, ev.TS)
		}
	}

	// Verify array export
	var arrBuf bytes.Buffer
	if err := ExportTraceEventsJSON(&arrBuf, metrics.TraceEvents); err != nil {
		t.Fatalf("failed to export trace events array JSON: %v", err)
	}
	var arrParsed []TraceEvent
	if err := json.Unmarshal(arrBuf.Bytes(), &arrParsed); err != nil {
		t.Fatalf("invalid trace array JSON: %v", err)
	}
	if len(arrParsed) != len(metrics.TraceEvents) {
		t.Errorf("expected %d events in array, got %d", len(metrics.TraceEvents), len(arrParsed))
	}
}

func TestChromeTraceExport(t *testing.T) {
	TestServingSimChromeTraceExport(t)
}

func TestPercentiles(t *testing.T) {
	// Empty slice
	empty := ComputePercentiles(nil)
	if empty.P50 != 0 || empty.P99 != 0 {
		t.Errorf("expected zero for empty percentiles, got %+v", empty)
	}

	// Single element
	single := ComputePercentiles([]float64{42.0})
	if single.P50 != 42.0 || single.P90 != 42.0 || single.P99 != 42.0 || single.Min != 42.0 || single.Max != 42.0 {
		t.Errorf("expected 42.0 for all metrics, got %+v", single)
	}

	// 100 sorted values: 1 to 100
	vals := make([]float64, 100)
	for i := range vals {
		vals[i] = float64(i + 1)
	}
	p := ComputePercentiles(vals)
	if math.Abs(p.P50-50.5) > 0.5 {
		t.Errorf("expected P50 ~ 50.5, got %f", p.P50)
	}
	if math.Abs(p.P90-90.1) > 1.0 {
		t.Errorf("expected P90 ~ 90, got %f", p.P90)
	}
	if math.Abs(p.P99-99.0) > 1.0 {
		t.Errorf("expected P99 ~ 99, got %f", p.P99)
	}
}

func TestInvalidInputs(t *testing.T) {
	config := SimulatorConfig{
		AcceptanceRate: 1.5, // invalid
	}
	_, err := NewServingSimulator(config, DefaultHardwareLatencyTable())
	if err == nil {
		t.Error("expected error on invalid AcceptanceRate > 1.0")
	}

	validConfig := SimulatorConfig{
		AcceptanceRate: 0.5,
	}
	engine, err := NewServingSimulator(validConfig, DefaultHardwareLatencyTable())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Invalid request with negative prompt
	_, err = engine.Run([]RequestState{
		{ID: "bad-prompt", PromptTokens: -1, OutputTarget: 10},
	})
	if err == nil {
		t.Error("expected error for negative prompt tokens")
	}

	// Invalid request with negative output
	_, err = engine.Run([]RequestState{
		{ID: "bad-output", PromptTokens: 10, OutputTarget: 0},
	})
	if err == nil {
		t.Error("expected error for zero output target")
	}

	// Empty request list should return empty metrics without error
	metrics, err := engine.Run(nil)
	if err != nil {
		t.Fatalf("unexpected error for empty requests: %v", err)
	}
	if metrics.TotalRequests != 0 {
		t.Errorf("expected 0 requests, got %d", metrics.TotalRequests)
	}
}

func TestCustomHardwareLatencyModel(t *testing.T) {
	hw := HardwareLatencyTable{
		PrefillChunkLatencyFunc: func(tokens int, batchSize int) float64 {
			return 10.0 + float64(tokens)*0.1
		},
		DecodeStepLatencyFunc: func(batchSize int, draftK int) float64 {
			return 5.0 + float64(draftK)*2.0
		},
	}

	config := SimulatorConfig{
		MaxBatchSize:     2,
		MaxTokensPerStep: 512,
		KVBlockTokens:    16,
		TotalKVBlocks:    512,
	}
	requests := []RequestState{
		{ID: "custom-hw", ArrivalTimeMS: 0, PromptTokens: 100, OutputTarget: 5},
	}

	metrics, err := Run(config, requests, hw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if metrics.TotalRequests != 1 {
		t.Errorf("expected 1 request completed, got %d", metrics.TotalRequests)
	}
	// Prefill: 10.0 + 100*0.1 = 20ms
	// Decode: 4 decode steps * 5.0ms = 20ms
	// Total ~ 40ms
	if metrics.SimulatedDurationMS < 35.0 || metrics.SimulatedDurationMS > 45.0 {
		t.Errorf("expected simulated duration ~ 40ms, got %f", metrics.SimulatedDurationMS)
	}
}

func TestServingSimSpeculativeThroughputScaling(t *testing.T) {
	hw := DefaultHardwareLatencyTable()
	requests := []RequestState{
		{ID: "spec-scale-1", ArrivalTimeMS: 0, PromptTokens: 64, OutputTarget: 100},
	}

	// Baseline without speculative decoding (K=0)
	baseCfg := SimulatorConfig{
		MaxBatchSize:       1,
		MaxTokensPerStep:   512,
		KVBlockTokens:      16,
		TotalKVBlocks:      512,
		SpeculativeHorizon: 0,
		AcceptanceRate:     0.8,
		Seed:               42,
	}
	baseMetrics, err := Run(baseCfg, requests, hw)
	if err != nil {
		t.Fatalf("base run failed: %v", err)
	}

	// Speculative decoding with K=4 and alpha=0.8
	specCfg := SimulatorConfig{
		MaxBatchSize:       1,
		MaxTokensPerStep:   512,
		KVBlockTokens:      16,
		TotalKVBlocks:      512,
		SpeculativeHorizon: 4,
		AcceptanceRate:     0.8,
		SpeculativeMode:    SpeculativeModeDeterministic,
		Seed:               42,
	}
	specMetrics, err := Run(specCfg, requests, hw)
	if err != nil {
		t.Fatalf("spec run failed: %v", err)
	}

	if specMetrics.TokenThroughput <= baseMetrics.TokenThroughput {
		t.Errorf("expected speculative token throughput (%f) > base (%f)", specMetrics.TokenThroughput, baseMetrics.TokenThroughput)
	}
	if specMetrics.SimulatedDurationMS >= baseMetrics.SimulatedDurationMS {
		t.Errorf("expected speculative duration (%f) < base (%f)", specMetrics.SimulatedDurationMS, baseMetrics.SimulatedDurationMS)
	}
}

func TestSpeculativeSpeedup(t *testing.T) {
	TestServingSimSpeculativeThroughputScaling(t)
}

func TestSingleTokenOutput(t *testing.T) {
	cfg := SimulatorConfig{
		MaxBatchSize:     2,
		MaxTokensPerStep: 512,
		KVBlockTokens:    16,
		TotalKVBlocks:    256,
	}
	hw := DefaultHardwareLatencyTable()
	requests := []RequestState{
		{ID: "single-token", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 1},
	}

	metrics, err := Run(cfg, requests, hw)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if metrics.TotalRequests != 1 {
		t.Fatalf("expected 1 completed request, got %d", metrics.TotalRequests)
	}
	req := metrics.CompletedRequests[0]
	if req.TokensGenerated != 1 {
		t.Errorf("expected 1 token generated, got %d", req.TokensGenerated)
	}
	if req.FirstTokenTimeMS != req.CompletionTimeMS {
		t.Errorf("expected FirstTokenTimeMS == CompletionTimeMS for single token request, got %f vs %f", req.FirstTokenTimeMS, req.CompletionTimeMS)
	}
}

func TestStaggeredArrivalsAndFIFOQueue(t *testing.T) {
	cfg := SimulatorConfig{
		MaxBatchSize:     2,
		MaxTokensPerStep: 512,
		KVBlockTokens:    16,
		TotalKVBlocks:    1024,
	}
	hw := DefaultHardwareLatencyTable()

	requests := []RequestState{
		{ID: "req-A", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 30},
		{ID: "req-B", ArrivalTimeMS: 0, PromptTokens: 32, OutputTarget: 30},
		{ID: "req-C", ArrivalTimeMS: 2, PromptTokens: 32, OutputTarget: 10},
	}

	metrics, err := Run(cfg, requests, hw)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if metrics.TotalRequests != 3 {
		t.Fatalf("expected 3 completed requests, got %d", metrics.TotalRequests)
	}

	// req-C arrived while req-A and req-B filled the max batch of 2.
	// Therefore, req-C should have positive queue time.
	var reqC RequestState
	for _, r := range metrics.CompletedRequests {
		if r.ID == "req-C" {
			reqC = r
			break
		}
	}
	if reqC.QueueTimeMS <= 0 {
		t.Errorf("expected req-C to have queued waiting for a free batch slot, got %f", reqC.QueueTimeMS)
	}
}

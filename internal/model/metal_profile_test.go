package model

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func completeMetalSnapshot(operation metalgemm.ExecutionOperation, encoders int, gpuMS float64) metalgemm.ExecutionSnapshot {
	return metalgemm.ExecutionSnapshot{Events: []metalgemm.ExecutionEvent{{
		Operation: operation, CommandBufferID: 1, Committed: true, CompletedWait: true,
		HostReadback: true, Encoders: encoders, GPUMilliseconds: gpuMS,
		WaitMilliseconds: gpuMS + 0.1, TimingAvailable: true,
	}}}
}

func TestPhaseProfilerOwnsMetalCountersPerSession(t *testing.T) {
	left, right := NewPhaseProfiler(), NewPhaseProfiler()
	left.recordMetal(completeMetalSnapshot(metalgemm.ExecutionQ4KGEMM, 1, 1.5), nil)
	right.recordMetal(completeMetalSnapshot(metalgemm.ExecutionQ4KFusedMLP, 3, 2.5), nil)
	left.recordMetalFallback(MetalFallbackQ8GEMVCPU)

	lc, err := left.MetalExecutionCounters()
	if err != nil {
		t.Fatal(err)
	}
	rc, err := right.MetalExecutionCounters()
	if err != nil {
		t.Fatal(err)
	}
	if lc.CommandBuffers != 1 || lc.Encoders != 1 || lc.DispatchMilliseconds != 1.5 {
		t.Fatalf("left counters mixed: %+v", lc)
	}
	if rc.CommandBuffers != 1 || rc.Encoders != 3 || rc.DispatchMilliseconds != 2.5 {
		t.Fatalf("right counters mixed: %+v", rc)
	}
	if left.MetalFallbackCount() != 1 || right.MetalFallbackCount() != 0 {
		t.Fatalf("fallback counts mixed: left=%d right=%d", left.MetalFallbackCount(), right.MetalFallbackCount())
	}
}

func TestPhaseProfilerMetalCountersFailClosed(t *testing.T) {
	profiler := NewPhaseProfiler()
	profiler.recordMetal(metalgemm.ExecutionSnapshot{}, nil)
	if _, err := profiler.MetalExecutionCounters(); !metalgemm.IsExecutionCountersIncomplete(err) {
		t.Fatalf("err = %v, want typed incomplete capture", err)
	}
}

func TestMetalFallbackReceiptOrdersAndClassifiesEveryRoute(t *testing.T) {
	routes := []MetalFallbackRoute{
		MetalFallbackQ4KGEMMCPU,
		MetalFallbackQ4KGEMVPanelCPU,
		MetalFallbackQ4KGEMMGroupDispatch,
		MetalFallbackQ4KGEMVGroupDispatch,
		MetalFallbackQ8GEMMCPU,
		MetalFallbackQ8GEMMGroupDispatch,
		MetalFallbackQ8GEMVGroupDispatch,
		MetalFallbackQ6KGEMMCPU,
		MetalFallbackQ6KGEMVCPU,
		MetalFallbackQ4KGEMVCPU,
		MetalFallbackQ8GEMVCPU,
		MetalFallbackQ4KGroupQ8CPU,
		MetalFallbackFusedMLPDispatch,
		MetalFallbackFusedMLPQ6DownDispatch,
		MetalFallbackFusedMLPBatchDispatch,
	}
	profiler := NewPhaseProfiler()
	for _, route := range routes {
		profiler.recordMetalFallback(route)
	}
	receipt, err := profiler.MetalFallbackReceipt()
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateMetalFallbackReceipt(receipt); err != nil {
		t.Fatalf("receipt did not read back: %v", err)
	}
	if len(receipt.Events) != len(routes) || receipt.PromisedCPUFallbacks != 8 {
		t.Fatalf("fallback receipt events=%d promised_cpu=%d", len(receipt.Events), receipt.PromisedCPUFallbacks)
	}
	for i, event := range receipt.Events {
		if event.Sequence != i+1 || event.Route != routes[i] {
			t.Fatalf("event[%d] = %+v", i, event)
		}
		if event.ActualBackend == metalFallbackBackendCPU && (!event.Promised || !event.CPUWorkExecuted) {
			t.Fatalf("CPU event lacks promised/executed semantics: %+v", event)
		}
		if event.ActualBackend == metalFallbackBackendNotExecuted && (event.Promised || event.CPUWorkExecuted || event.Disposition != metalFallbackDispositionCaller) {
			t.Fatalf("candidate decline was mislabeled as CPU fallback: %+v", event)
		}
	}
}

func TestMetalFallbackReceiptRejectsTamper(t *testing.T) {
	profiler := NewPhaseProfiler()
	profiler.recordMetalFallback(MetalFallbackQ8GEMMCPU)
	receipt, err := profiler.MetalFallbackReceipt()
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(*MetalFallbackReceipt)
		want string
	}{
		{name: "sequence", edit: func(r *MetalFallbackReceipt) { r.Events[0].Sequence = 2 }, want: "sequence"},
		{name: "route semantics", edit: func(r *MetalFallbackReceipt) { r.Events[0].Promised = false }, want: "route semantics"},
		{name: "digest", edit: func(r *MetalFallbackReceipt) { r.EventsSHA256 = strings.Repeat("0", 64) }, want: "digest mismatch"},
		{name: "aggregate", edit: func(r *MetalFallbackReceipt) { r.PromisedCPUFallbacks = 0 }, want: "aggregate"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			copy := receipt
			copy.Events = append([]MetalFallbackEvent(nil), receipt.Events...)
			test.edit(&copy)
			if err := ValidateMetalFallbackReceipt(copy); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err = %v, want %q", err, test.want)
			}
		})
	}
}

package agent

import (
	"context"
	"testing"
	"time"
)

// reachability_13120_test.go — the PRODUCTION-REACHABILITY witness for the native-phase
// half of #13120. native_phase_test.go hand-injects a bare-string "trace_id" context value
// that NO production code sets; this test drives the seam the served path actually uses
// (WithRequestTraceID) and asserts the getter resolves it.

func TestNativePhaseTraceIDReadsRequestTraceSeam(t *testing.T) {
	ctx := WithRequestTraceID(context.Background(), "gw-11")
	if id := nativePhaseTraceID(ctx); id != "gw-11" {
		t.Fatalf("nativePhaseTraceID(WithRequestTraceID ctx) = %q, want gw-11; the served path cannot bind a native phase", id)
	}

	// A nil-context / empty id stays empty rather than fabricating one.
	if id := nativePhaseTraceID(WithRequestTraceID(context.Background(), "")); id != "" {
		t.Fatalf("empty request trace id = %q, want empty", id)
	}
	if id := nativePhaseTraceID(WithRequestTraceID(nil, "gw-12")); id != "gw-12" {
		t.Fatalf("nil-context request trace id = %q, want gw-12", id)
	}
}

// TestNativePhaseObservationReachableFromRequestTrace drives the REAL recorder through the
// production context seam end-to-end: stamp the request trace, record a phase under the id
// the getter resolves, then read it back under that same bare id — the exact shape
// debugRequestAdmission performs. Before the fix the getter returned "" so this missed.
func TestNativePhaseObservationReachableFromRequestTrace(t *testing.T) {
	var p InKernelPlanner
	ctx := WithRequestTraceID(context.Background(), "gw-13")

	p.recordNativePhase(nativePhaseTraceID(ctx), NativePhasePrefill, time.Date(2026, 9, 20, 12, 0, 0, 0, time.UTC), 0, false)

	obs, ok := p.NativePhaseObservation("gw-13")
	if !ok {
		t.Fatal("NativePhaseObservation(gw-13) missed; the debug native_phase half cannot render in production")
	}
	if obs.Phase != NativePhasePrefill {
		t.Fatalf("phase = %s, want prefill", obs.Phase)
	}
}

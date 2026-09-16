package model

import "testing"

// TestMetalFallbackRouteIndexRoundTrip pins the stable slot mapping (#12875): every route in the
// order vector round-trips through index→route, and an unknown route is reported absent rather
// than silently aliasing slot 0.
func TestMetalFallbackRouteIndexRoundTrip(t *testing.T) {
	for i, route := range metalFallbackRouteOrder {
		got, ok := metalFallbackRouteIndex(route)
		if !ok || got != i {
			t.Fatalf("metalFallbackRouteIndex(%q) = (%d,%v), want (%d,true)", route, got, ok, i)
		}
		if back := metalFallbackRouteByIndex(i); back != string(route) {
			t.Fatalf("metalFallbackRouteByIndex(%d) = %q, want %q", i, back, route)
		}
	}
	if _, ok := metalFallbackRouteIndex(MetalFallbackRoute("not-a-route")); ok {
		t.Fatal("unknown route reported as present")
	}
	if got := metalFallbackRouteByIndex(-1); got != "" {
		t.Fatalf("metalFallbackRouteByIndex(-1) = %q, want empty", got)
	}
	if got := metalFallbackRouteByIndex(len(metalFallbackRouteOrder)); got != "" {
		t.Fatalf("out-of-range index = %q, want empty", got)
	}
}

// TestModelMetalFallbackLiveTally pins that the live counter accumulates across calls (surviving
// per-request Session churn) and reports the per-route breakdown. This is the counter /healthz
// reads at request time instead of the frozen startup snapshot.
func TestModelMetalFallbackLiveTally(t *testing.T) {
	m := &Model{}
	if snap := m.MetalFallbackSnapshot(); snap.Total != 0 || snap.Observed {
		t.Fatalf("fresh model snapshot = %+v, want zero/unobserved", snap)
	}
	m.recordMetalFallbackLive(MetalFallbackQ8GEMVCPU)
	m.recordMetalFallbackLive(MetalFallbackQ8GEMVCPU)
	m.recordMetalFallbackLive(MetalFallbackQ6KGEMMCPU)
	snap := m.MetalFallbackSnapshot()
	if snap.Total != 3 {
		t.Fatalf("total = %d, want 3", snap.Total)
	}
	if snap.ByRoute[string(MetalFallbackQ8GEMVCPU)] != 2 {
		t.Fatalf("q8-gemv-cpu = %d, want 2", snap.ByRoute[string(MetalFallbackQ8GEMVCPU)])
	}
	if snap.ByRoute[string(MetalFallbackQ6KGEMMCPU)] != 1 {
		t.Fatalf("q6k-gemm-cpu = %d, want 1", snap.ByRoute[string(MetalFallbackQ6KGEMMCPU)])
	}
	if !snap.Observed {
		t.Fatal("Observed = false after recording a fallback")
	}
}

// TestModelMetalFallbackUnknownRouteCountsTotal pins that a route with no dedicated slot still
// advances the scalar total, so an unmapped future route can never be dropped from the count.
func TestModelMetalFallbackUnknownRouteCountsTotal(t *testing.T) {
	m := &Model{}
	m.recordMetalFallbackLive(MetalFallbackRoute("future-route"))
	snap := m.MetalFallbackSnapshot()
	if snap.Total != 1 {
		t.Fatalf("total = %d, want 1 for an unmapped route", snap.Total)
	}
	if len(snap.ByRoute) != 0 {
		t.Fatalf("unmapped route produced a per-route entry: %#v", snap.ByRoute)
	}
}

// TestModelMetalResidencySnapshot records and reports device-resident counts, and keeps
// "not yet observed" distinct from a genuine zero.
func TestModelMetalResidencySnapshot(t *testing.T) {
	m := &Model{}
	if _, _, known := m.MetalResidencySnapshot(); known {
		t.Fatal("fresh model reports residency as known")
	}
	m.noteMetalResidency(16, 272)
	q6k, q8, known := m.MetalResidencySnapshot()
	if !known || q6k != 16 || q8 != 272 {
		t.Fatalf("snapshot = (%d,%d,%v), want (16,272,true)", q6k, q8, known)
	}
}

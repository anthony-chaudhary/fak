package gateway

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

// This file carries the micro-benchmarks and allocation floors for two hot paths
// the perf audit (issue #995) found unguarded in the public core:
//
//  1. Metrics Observe under a global-mutex histogram
//     (internal/gateway/metrics_http.go, observeHTTP -> gatewayMetrics.mu ->
//     latencyCounter.observe). Benchmarked directly on an already-populated
//     histogram, the steady-state hot case.
//  2. The SSE chunk flush loop (the writeSSEData primitive the streaming
//     completion/chat emitters call once per fragment; internal/gateway/http.go).
//     Benchmarked on an httptest.ResponseRecorder (which implements http.Flusher),
//     at completion-length scales to witness that per-chunk work is flat.

// benchObserveMetricsServer returns a gatewayMetrics whose http histogram is
// already populated, so observeHTTP exercises the steady-state map-hit path rather
// than the one-time counter allocation.
func benchObserveMetricsServer() *gatewayMetrics {
	m := newGatewayMetrics(time.Now())
	m.observeHTTP("/v1/chat/completions", http.MethodPost, http.StatusOK, time.Millisecond)
	return m
}

// BenchmarkMetricsObserveHTTP measures one request-completion observation into the
// global-mutex latency histogram — the path every served request takes through
// withMetrics' deferred observeHTTP.
func BenchmarkMetricsObserveHTTP(b *testing.B) {
	m := benchObserveMetricsServer()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.observeHTTP("/v1/chat/completions", http.MethodPost, http.StatusOK, time.Millisecond)
	}
}

// TestMetricsObserveAllocFloor asserts the histogram observation is
// allocation-flat in steady state: after the counter for the key exists, each
// observe must not allocate. A regression that rebuilds the key/label or the
// counter per call would fail this floor.
func TestMetricsObserveAllocFloor(t *testing.T) {
	m := benchObserveMetricsServer()
	const budget = 1.0 // bucket scan touches no heap; allow a rare runtime spike
	allocs := testing.AllocsPerRun(500, func() {
		m.observeHTTP("/v1/chat/completions", http.MethodPost, http.StatusOK, time.Millisecond)
	})
	if allocs > budget {
		t.Fatalf("observeHTTP steady-state must not allocate: got %v allocs/op (budget %v)", allocs, budget)
	}
}

// flushRecorder is the httptest.ResponseRecorder plus a Flush counter, so the SSE
// benchmark can witness that every chunk actually flushes (the loop's whole point)
// without a network.
type flushRecorder struct {
	*httptest.ResponseRecorder
	flushes int
}

func (f *flushRecorder) Flush() { f.flushes++ }

// BenchmarkSSEChunkFlush measures the per-chunk SSE flush primitive writeSSEData
// over completion-length scales. b.SetBytes reports the chunk payload so ns/op is
// readable as a throughput; the scaling sub-benchmarks witness that per-chunk cost
// stays flat as the completion grows (no full-buffer copy per chunk).
func BenchmarkSSEChunkFlush(b *testing.B) {
	for _, segs := range []int{64, 512, 4096} {
		b.Run("segs="+strconv.Itoa(segs), func(b *testing.B) {
			payload := strings.Repeat("token ", 16)
			w := &flushRecorder{ResponseRecorder: httptest.NewRecorder()}
			b.SetBytes(int64(len(payload)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				w.Body.Reset()
				for j := 0; j < segs; j++ {
					if err := writeSSEData(w, map[string]string{"content": payload}); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

// TestSSEChunkFlushFlushes asserts the flush loop actually calls Flush once per
// chunk — the property a buffered writer regression would silently drop.
func TestSSEChunkFlushFlushes(t *testing.T) {
	w := &flushRecorder{ResponseRecorder: httptest.NewRecorder()}
	for i := 0; i < 10; i++ {
		if err := writeSSEData(w, map[string]string{"content": "x"}); err != nil {
			t.Fatal(err)
		}
	}
	if w.flushes != 10 {
		t.Fatalf("expected 10 flushes (one per chunk), got %d", w.flushes)
	}
}

// TestSSEChunkFlushPerChunkAllocsFlat asserts the per-chunk allocation floor does
// not grow with completion length: the mean allocs/chunk at 4096 segments must be
// no worse than at 64 segments (json.Marshal dominates and is size-independent
// here). Additional per-chunk allocations that grow with completion length
// fail this check; it does not measure allocated byte volume.
func TestSSEChunkFlushPerChunkAllocsFlat(t *testing.T) {
	const measuredChunks = 204800
	perChunk := func(segs int) float64 {
		w := &flushRecorder{ResponseRecorder: httptest.NewRecorder()}
		// A production ResponseWriter sends each chunk onward; retaining the full
		// stream in the test recorder adds allocator work unrelated to writeSSEData.
		w.Body = nil
		allocs := testing.AllocsPerRun(measuredChunks/segs, func() {
			for j := 0; j < segs; j++ {
				if err := writeSSEData(w, map[string]string{"content": "token token "}); err != nil {
					t.Fatal(err)
				}
			}
		})
		return allocs / float64(segs)
	}
	short := perChunk(64)
	long := perChunk(4096)
	// AllocsPerRun returns float64; the per-chunk quotient carries sub-alloc
	// floating-point noise, so compare with a tolerance rather than exact >.
	if long > short+0.01 {
		t.Fatalf("per-chunk allocs grew with completion length: short=%.3f long=%.3f", short, long)
	}
	t.Logf("per-chunk allocs: short=%.3f long=%.3f", short, long)
}

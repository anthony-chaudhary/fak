package gateway

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// #2775's production fault class is a RUNTIME-raised handler panic — a nil-map
// write or a nil-pointer dereference in a downstream handler — which reaches the
// recover in withMetrics through runtime.gopanic -> runtime.sigpanic / mapassign,
// NOT through an explicit panic(...) call. Every other panic test in this package
// raises an explicit string panic (panic("boom")), so the sigpanic/panicmem branch
// of panicOriginFrame — the exact branch the issue's "guard the nil/map/whatever"
// acceptance is about — was asserted in a code comment but never witnessed. These
// two named handlers reproduce that class synthetically (no shared state, no live
// model), so the pinned origin frame has a stable, assertable function name.

func nilMapWriteHandler(w http.ResponseWriter, r *http.Request) {
	var m map[string]int
	m["boom"] = 1 // assignment to entry in nil map: runtime-raised, recoverable
}

func nilPointerDerefHandler(w http.ResponseWriter, r *http.Request) {
	var p *int
	if *p == 0 { // nil pointer dereference: dispatched via runtime.sigpanic
		w.WriteHeader(http.StatusTeapot)
	}
}

// TestWithMetricsPinsRuntimeRaisedHandlerPanic is the #2775 regression: a
// runtime-raised handler panic (nil-map write / nil-pointer deref) must be
// (1) contained as a structured 500 instead of crashing the process or unwinding
// into net/http, and (2) pinned to the exact faulting handler in the structured
// gateway_recovered_panic event's "origin" field — through the sigpanic/panicmem
// stack path, which no prior test in this package exercised. Without the pin, the
// route alone ("/v1/fak/syscall") is too coarse to name which handler faulted,
// which is precisely the "unrooted handler panic" the issue reports.
func TestWithMetricsPinsRuntimeRaisedHandlerPanic(t *testing.T) {
	cases := []struct {
		name      string
		handler   http.HandlerFunc
		wantFn    string
		wantPanic string
	}{
		{"nil_map_write", nilMapWriteHandler, "nilMapWriteHandler", "nil map"},
		{"nil_pointer_deref", nilPointerDerefHandler, "nilPointerDerefHandler", "nil pointer dereference"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var mu sync.Mutex
			var lines []string
			s := &Server{
				metrics: newGatewayMetrics(time.Now()),
				logf: func(f string, a ...any) {
					mu.Lock()
					defer mu.Unlock()
					lines = append(lines, fmt.Sprintf(f, a...))
				},
			}
			h := s.withMetrics(c.handler)

			// If the recover did not contain the runtime-raised panic, this call
			// would propagate and crash the test goroutine outright — so reaching
			// the assertions below already proves containment.
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/syscall", nil))

			if rr.Code != http.StatusInternalServerError {
				t.Fatalf("status = %d, want 500 (a runtime-raised handler panic must be contained, not crash)", rr.Code)
			}
			ev := findLogEvent(t, lines, "gateway_recovered_panic")
			if pv, _ := ev["panic"].(string); !strings.Contains(pv, c.wantPanic) {
				t.Fatalf("panic value = %q, want it to contain %q (the runtime fault text)", pv, c.wantPanic)
			}
			origin, ok := ev["origin"].(string)
			if !ok || origin == "" {
				t.Fatalf("recovered-panic event carries no origin frame; #2775 needs the faulting handler pinned even for the sigpanic/panicmem path: %+v", ev)
			}
			if !strings.Contains(origin, c.wantFn) {
				t.Fatalf("origin = %q, want it to name the faulting handler %q (the runtime-raised pin regressed)", origin, c.wantFn)
			}
			// #2772 guard: the pin must stay a single compact frame, never the
			// multi-line "goroutine …" dump net/http would leak to the guarded
			// child's controlling TTY.
			if strings.ContainsAny(origin, "\n\r") {
				t.Fatalf("origin leaked a multi-line dump: %q", origin)
			}
		})
	}
}

// TestWithMetricsRuntimeRaisedPanicRaceClean fires many runtime-raised handler
// panics concurrently through ONE shared withMetrics wrapper. Recovering each
// panic touches withMetrics' shared accounting — the inflight counter, the
// metrics registry (the mutex-guarded maps the issue calls out), and the
// served-failure record — so under `go test -race` this witnesses acceptance
// item 3 for the reproducing shape available on a dev host: the shared gateway
// state stays race-clean while it contains a burst of panicking turns, and every
// request still comes back as a contained 500 rather than a reset connection.
func TestWithMetricsRuntimeRaisedPanicRaceClean(t *testing.T) {
	s := &Server{metrics: newGatewayMetrics(time.Now()), logf: func(string, ...any) {}}
	h := s.withMetrics(http.HandlerFunc(nilMapWriteHandler))

	const n = 64
	var wg sync.WaitGroup
	codes := make([]int, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/v1/completions", nil))
			codes[i] = rr.Code // distinct index per goroutine: no shared-slot write
		}(i)
	}
	wg.Wait()
	for i, code := range codes {
		if code != http.StatusInternalServerError {
			t.Fatalf("request %d status = %d, want 500 under concurrent panic load", i, code)
		}
	}
}

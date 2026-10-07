package model

import (
	"math"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=200ms lane=default
func TestQ8RowObserverPreservesCalculationWithoutTracing(t *testing.T) {
	if computetrace.Enabled() {
		t.Fatal("fixture requires trace artifact collection disabled")
	}
	computetrace.SetObserver(nil)
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	const rows, in = 64, 128
	qt := quantizeQ8(mkVec(rows*in, 983), rows, in)
	qv := quantizeVecQ8(mkVec(in, 173))
	want := qMatRows(qt, qv)
	var mu sync.Mutex
	var events []computetrace.Event
	computetrace.SetObserver(func(e computetrace.Event) { mu.Lock(); events = append(events, e); mu.Unlock() })
	take := func() []computetrace.Event {
		mu.Lock()
		defer mu.Unlock()
		out := append([]computetrace.Event(nil), events...)
		events = nil
		return out
	}
	checkEvents := func(got []computetrace.Event, wantRows, wantCount int) {
		t.Helper()
		if wantCount >= 0 && len(got) != wantCount {
			t.Fatalf("events=%d want=%d", len(got), wantCount)
		}
		total := 0
		for _, e := range got {
			if e.Kernel != "q8_gemv_rows" || e.Backend != "cpu-ref" || e.TimerDomain != "host_monotonic" || e.DurationNS < 0 || len(e.Shapes) != 1 || len(e.Shapes[0]) != 2 || e.Shapes[0][0] <= 0 || e.Shapes[0][1] != in {
				t.Fatalf("invalid row observer kernel=%q backend=%q domain=%q duration=%d shapes=%v", e.Kernel, e.Backend, e.TimerDomain, e.DurationNS, e.Shapes)
			}
			total += e.Shapes[0][0]
		}
		if total != wantRows {
			t.Fatalf("observed rows=%d want=%d", total, wantRows)
		}
	}
	compare := func(got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatal("output shape changed")
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("observer changed row %d result", i)
			}
		}
	}
	got := make([]float32, rows)
	qMatRowsRange(qt, qv, got, 0, rows)
	compare(got, want)
	checkEvents(take(), rows, 1)
	qMatRowsRange(qt, qv, got, 17, 17)
	checkEvents(take(), 0, 0)
	compare(got, want)
	group := []qMatTarget{{qt: qt, dst: make([]float32, rows)}, {qt: qt, dst: make([]float32, rows)}}
	qMatRowsIntoMany(qv, group...)
	for _, target := range group {
		compare(target.dst, want)
	}
	checkEvents(take(), 2*rows, 2)
	parallel := make([]float32, rows)
	parFor(rows, 2, func(lo, hi int) { qMatRowsRange(qt, qv, parallel, lo, hi) })
	compare(parallel, want)
	checkEvents(take(), rows, -1)
	if computetrace.Enabled() {
		t.Fatal("observer enabled trace artifacts")
	}
}

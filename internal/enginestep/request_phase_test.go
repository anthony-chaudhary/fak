package enginestep

import (
	"context"
	"sync"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms
func TestRequestPhasePresenceIsolationAndCopy(t *testing.T) {
	recorder := New(16)
	ctx := WithRequestObservation(context.Background(), "request-one", "load-one")
	other := WithRequestObservation(ctx, "request-two", "load-one")
	if initial := RequestPhaseObservationsFromContext(ctx); initial.Prefill != nil || initial.Decode != nil {
		t.Fatal("unreached phase measured")
	}
	recorder.ObservePhaseContext(ctx, PhasePrefill, 0)
	recorder.ObservePartialPhaseContext(ctx, PhasePrefill, 3*time.Nanosecond)
	recorder.ObservePhaseContext(ctx, PhaseDecode, 0)
	recorder.ObservePhaseContext(other, PhaseDecode, 7*time.Nanosecond)
	got := RequestPhaseObservationsFromContext(ctx)
	if got.Prefill == nil || got.Prefill.Count != 2 || got.Prefill.DurationNS != 3 || got.Decode == nil || got.Decode.Count != 1 || got.Decode.DurationNS != 0 {
		t.Fatalf("phases=%+v", got)
	}
	got.Prefill.DurationNS = 1000
	if RequestPhaseObservationsFromContext(ctx).Prefill.DurationNS != 3 {
		t.Fatal("snapshot aliases live state")
	}
	second := RequestPhaseObservationsFromContext(other)
	if second.Prefill != nil || second.Decode == nil || second.Decode.DurationNS != 7 {
		t.Fatal("request phases crossed contexts")
	}
	global := recorder.Snapshot(16, "")
	if global.Phases[string(PhasePrefill)].Count != 1 {
		t.Fatal("partial phase completed global phase")
	}
	if empty := RequestPhaseObservationsFromContext(nil); empty.Prefill != nil || empty.Decode != nil {
		t.Fatal("nil context invented phases")
	}
	unknown := WithRequestObservation(nil, "", "load-one")
	recorder.ObservePhaseContext(unknown, PhaseDecode, time.Second)
	if RequestPhaseObservationsFromContext(unknown).Decode != nil {
		t.Fatal("missing request ID acquired phases")
	}
}

// fak-test:runtime fast est=100ms
func TestRequestPhaseConcurrentSnapshot(t *testing.T) {
	recorder := New(4)
	ctx := WithRequestObservation(nil, "request-one", "load-one")
	var group sync.WaitGroup
	for i := 0; i < 8; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for j := 0; j < 16; j++ {
				recorder.ObservePartialPhaseContext(ctx, PhaseDecode, 2*time.Nanosecond)
				snapshot := RequestPhaseObservationsFromContext(ctx)
				if snapshot.Decode == nil || snapshot.Decode.DurationNS != int64(snapshot.Decode.Count)*2 {
					t.Error("incoherent request snapshot")
				}
			}
		}()
	}
	group.Wait()
	got := RequestPhaseObservationsFromContext(ctx)
	if got.Decode == nil || got.Decode.Count != 128 || got.Decode.DurationNS != 256 {
		t.Fatalf("snapshot=%+v", got)
	}
}

// fak-test:runtime fast est=5ms
func TestRequestPhaseEndMarksLastObservation(t *testing.T) {
	recorder := New(16)
	ctx := WithRequestObservation(context.Background(), "request-end", "load-one")
	if _, ok := RequestPhaseEndFromContext(ctx); ok {
		t.Fatal("phase end measured before any phase")
	}
	if _, ok := RequestPhaseEndFromContext(nil); ok {
		t.Fatal("nil context measured a phase end")
	}
	before := time.Now()
	recorder.ObservePhaseContext(ctx, PhasePrefill, time.Millisecond)
	first, ok := RequestPhaseEndFromContext(ctx)
	if !ok || first.Before(before) {
		t.Fatalf("prefill end = %v ok=%v, want at or after %v", first, ok, before)
	}
	recorder.ObservePhaseContext(ctx, PhaseDecode, time.Millisecond)
	last, ok := RequestPhaseEndFromContext(ctx)
	if !ok || last.Before(first) {
		t.Fatalf("decode end = %v, want at or after prefill end %v", last, first)
	}
}

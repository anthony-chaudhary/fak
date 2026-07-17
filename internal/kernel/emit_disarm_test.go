package kernel

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestRepeatedlyPanickingObserverIsDisarmed is the acceptance gate for #5091: a tap
// that panics on EVERY event is entered at most ObserverDisarmThreshold times, then
// skipped — the fail-open fan-out (#4266) becomes self-healing instead of paying a
// recover + record-store for a known-broken observer on every syscall forever.
func TestRepeatedlyPanickingObserverIsDisarmed(t *testing.T) {
	setup()
	bad := &panicEmitter{}
	abi.RegisterEmitter(bad)
	ev := abi.Event{Kind: abi.EvSubmit, Call: call("read_x", "{}")}

	for i := 0; i < 10*ObserverDisarmThreshold; i++ {
		emit(ev)
	}

	if got := atomic.LoadInt64(&bad.n); got != ObserverDisarmThreshold {
		t.Fatalf("panicking observer entered %d times over many emits, want exactly %d (the disarm threshold)", got, ObserverDisarmThreshold)
	}
}

// TestObserverDisarmIsWitnessed pins the observable half of #5091: a disarmed tap
// must be a record an operator can read — which observer, after how many panics, on
// which event kind — not a silent disappearance. A silently-dropped observer is its
// own mystery.
func TestObserverDisarmIsWitnessed(t *testing.T) {
	setup()
	bad := &panicEmitter{}
	abi.RegisterEmitter(bad)
	ev := abi.Event{Kind: abi.EvDeny, Call: call("read_x", "{}")}

	// Disarm state is process-global, so measure the DELTA: taps disarmed by earlier
	// tests in this binary must not make this one flake.
	before := len(DisarmedObservers())
	for i := 0; i < ObserverDisarmThreshold+1; i++ {
		emit(ev)
	}

	records := DisarmedObservers()
	if len(records) != before+1 {
		t.Fatalf("DisarmedObservers() grew by %d, want 1: %+v", len(records)-before, records)
	}
	last := records[len(records)-1]
	if !strings.Contains(last.Observer, "panicEmitter") {
		t.Errorf("disarm record names %q, want the offending *kernel.panicEmitter", last.Observer)
	}
	if last.Panics != ObserverDisarmThreshold {
		t.Errorf("disarm record says %d panics, want %d", last.Panics, ObserverDisarmThreshold)
	}
	if last.Kind != abi.EvDeny {
		t.Errorf("disarm record kind = %v, want EvDeny (the kind that tripped the threshold)", last.Kind)
	}
}

// TestDisarmedObserverDoesNotBlindPeers pins that the disarm skip is PER-OBSERVER: a
// disarmed tap's peers keep receiving every event, and a fresh healthy observer
// registered after a disarm is entered normally. Disarm removes one broken tap, not
// the fan-out.
func TestDisarmedObserverDoesNotBlindPeers(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("e", &countEngine{})
	bad := &panicEmitter{}
	abi.RegisterEmitter(bad)
	good := &recordEmitter{}
	abi.RegisterEmitter(good)
	k := New("e")

	// Enough syscalls (several events each) to trip the disarm threshold many times over.
	for i := 0; i < ObserverDisarmThreshold*4; i++ {
		k.Syscall(context.Background(), call("read_x", "{}"))
	}

	if got := atomic.LoadInt64(&bad.n); got != ObserverDisarmThreshold {
		t.Errorf("panicking observer entered %d times across syscalls, want exactly %d", got, ObserverDisarmThreshold)
	}
	if !good.has(abi.EvSubmit) || !good.has(abi.EvComplete) {
		t.Errorf("healthy peer of a disarmed observer missed events: %+v", good.events)
	}
}

// TestEmitAfterDisarmStaysZeroAlloc extends the #4266 cost pin across the #5091
// disarm: even AFTER an observer has been disarmed (the skip check now consults the
// disarmed set on every fan-out), a healthy observer's happy path still allocates
// nothing. The disarm may not move the damage from correctness to latency.
func TestEmitAfterDisarmStaysZeroAlloc(t *testing.T) {
	setup()
	bad := &panicEmitter{}
	abi.RegisterEmitter(bad)
	ev := abi.Event{Kind: abi.EvSubmit, Call: call("read_x", "{}")}
	for i := 0; i < ObserverDisarmThreshold; i++ {
		emit(ev) // trip the disarm so the skip path is armed
	}
	setup()
	abi.RegisterEmitter(nopEmitter{})

	if a := testing.AllocsPerRun(200, func() { emit(ev) }); a != 0 {
		t.Errorf("emit() allocates %.2f/op after a disarm; want 0", a)
	}
}

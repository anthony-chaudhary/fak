package kernel

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// panicEmitter is the deliberately-broken tap the fail-open contract is written
// against (#4266): an observer whose Emit always panics. It counts its invocations so
// a test can prove it actually ran — a fail-open assertion passes vacuously if the
// offending observer was never reached.
type panicEmitter struct{ n int64 }

func (p *panicEmitter) Emit(abi.Event) {
	atomic.AddInt64(&p.n, 1)
	panic("observer blew up")
}

// nopEmitter is a live but free observer: it exercises the fan-out (and its recover)
// without allocating, so the alloc witness below measures emit(), not the tap.
type nopEmitter struct{}

func (nopEmitter) Emit(abi.Event) {}

// TestPanickingObserverDoesNotFailSyscall is the failure-class proof for #4266: an
// observer that panics is INSTRUMENTATION failing, and instrumentation must not take
// down the syscall it was only watching. Against the pre-fix fan-out this test does not
// merely fail — the panic escapes emit() through Submit and takes the whole test binary
// with it, which is precisely the production blast radius being closed.
func TestPanickingObserverDoesNotFailSyscall(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	bad := &panicEmitter{}
	abi.RegisterEmitter(bad)
	k := New("e")

	r, v := k.Syscall(context.Background(), call("read_x", "{}"))

	if v.Kind != abi.VerdictAllow {
		t.Fatalf("verdict = %v, want Allow: a panicking observer must not change the verdict", v.Kind)
	}
	if r == nil || r.Status != abi.StatusOK {
		t.Fatalf("result = %+v, want StatusOK: a panicking observer must not fail the call", r)
	}
	if atomic.LoadInt64(&eng.n) != 1 {
		t.Fatalf("engine calls = %d, want 1: the syscall must still reach dispatch", eng.n)
	}
	if atomic.LoadInt64(&bad.n) == 0 {
		t.Fatal("the panicking observer was never invoked, so this test proves nothing")
	}
}

// TestPanickingObserverDoesNotStarvePeers pins that the recover is PER-OBSERVER rather
// than per-emit: a tap that panics must not swallow the events owed to the observers
// registered behind it. One recover around the whole walk would satisfy the
// syscall-survives test above while silently blinding every downstream tap.
func TestPanickingObserverDoesNotStarvePeers(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("e", &countEngine{})
	abi.RegisterEmitter(&panicEmitter{}) // registered FIRST: it panics ahead of the good tap
	good := &recordEmitter{}
	abi.RegisterEmitter(good)
	k := New("e")

	k.Syscall(context.Background(), call("read_x", "{}"))

	if !good.has(abi.EvSubmit) {
		t.Errorf("the observer behind a panicking one never got EvSubmit: %+v", good.events)
	}
	if !good.has(abi.EvComplete) {
		t.Errorf("the observer behind a panicking one never got EvComplete: %+v", good.events)
	}
}

// TestObserverPanicIsRecorded pins the observable half of the acceptance: fail-open must
// not mean fail-silent. The isolated panic is counted and its offender named, so a
// broken tap surfaces as a number and a type an operator can act on.
func TestObserverPanicIsRecorded(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("e", &countEngine{})
	abi.RegisterEmitter(&panicEmitter{})
	k := New("e")

	// The tally is process-global, so measure the DELTA: a panic isolated by an earlier
	// test in this binary must not make this one flake.
	before := ObserverPanics()
	k.Syscall(context.Background(), call("read_x", "{}"))

	if got := ObserverPanics() - before; got < 1 {
		t.Fatalf("isolated panics = %d, want >= 1: the panic must be counted, not swallowed", got)
	}
	last, ok := LastObserverPanic()
	if !ok {
		t.Fatal("LastObserverPanic() reported none after a panicking observer ran")
	}
	if !strings.Contains(last.Observer, "panicEmitter") {
		t.Errorf("recorded observer = %q, want the offending *kernel.panicEmitter", last.Observer)
	}
	if !strings.Contains(last.Value, "observer blew up") {
		t.Errorf("recorded value = %q, want the recovered panic value", last.Value)
	}
}

// TestEmitFanoutHappyPathZeroAlloc pins the cost side of the fix. emit() sits on the hot
// path every syscall walks (EvSubmit/EvDispatch/EvComplete), so the per-observer recover
// has to be free when nothing panics — a fail-open guarantee bought with an allocation
// per observer per event would just move the damage from correctness to latency. Mirrors
// abi's TestEmittersForZeroAlloc.
func TestEmitFanoutHappyPathZeroAlloc(t *testing.T) {
	setup()
	abi.RegisterEmitter(nopEmitter{})
	ev := abi.Event{Kind: abi.EvSubmit, Call: call("read_x", "{}")}

	if a := testing.AllocsPerRun(200, func() { emit(ev) }); a != 0 {
		t.Errorf("emit() allocates %.2f/op on the happy path; want 0", a)
	}
}

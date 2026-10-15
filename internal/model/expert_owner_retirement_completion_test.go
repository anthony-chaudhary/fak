package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type expertOwnerRetirementBackend struct {
	compute.Backend
	owner   sync.Mutex
	uploads int
	frees   int
	unowned bool
	fault   func()
}

func (b *expertOwnerRetirementBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

func (b *expertOwnerRetirementBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.uploads++
	return b.Backend.Upload(t, as)
}

func (b *expertOwnerRetirementBackend) Free(t compute.Tensor) {
	if b.owner.TryLock() {
		b.unowned = true
		b.owner.Unlock()
	}
	b.frees++
	if b.fault != nil {
		b.fault()
	}
	b.Backend.Free(t)
}

type expertOwnerRetirementFault struct{}

func (*expertOwnerRetirementFault) Error() string { return "expert owner Free failed" }

type expertOwnerRetirementOutcome struct {
	err                        error
	payload                    any
	returned, boundaryReturned bool
}

func stageExpertOwnerRetirementWeights(t *testing.T, be *expertOwnerRetirementBackend, ring *pagedRing) {
	t.Helper()
	scope := beginBackendExecution(be)
	complete := false
	defer scope.finishCompleted(&complete)
	if ring.shared != nil {
		ring.shared.mu.Lock()
		defer ring.shared.mu.Unlock()
	}
	for _, name := range []string{"first", "second"} {
		_, ok := ring.stage(name, func() compute.Tensor {
			return compute.NewF32(compute.Default(), []int{1, 1}, []float32{1})
		}, compute.F32, 4, false)
		if !ok {
			t.Fatalf("ring refused %s", name)
		}
	}
	if be.uploads != 2 || len(ring.resident) != 2 || ring.used() != 8 {
		t.Fatalf("ring not populated: uploads=%d resident=%d bytes=%d", be.uploads, len(ring.resident), ring.used())
	}
	complete = true
}

// fak-test:runtime fast est=100ms lane=default
func TestExpertOwnerRetirementPreservesIncompleteClose(t *testing.T) {
	primary := &expertOwnerRetirementFault{}
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, kind := range []string{"shared", "v4"} {
				for _, fault := range []string{"nil", "typed", "goexit", "healthy"} {
					t.Run(kind+"/"+fault, func(t *testing.T) {
						m := &Model{}
						be := &expertOwnerRetirementBackend{Backend: compute.Default()}
						var ring *pagedRing
						switch kind {
						case "shared":
							sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: 16})
							if err != nil {
								t.Fatal(err)
							}
							ring = sh.ring
							m.sharedExpertRings = map[compute.Backend]*SharedExpertRing{be: sh}
						case "v4":
							ring = newPagedRing(be, 16)
							// Exercise the real runtime Close and paged Free loop without
							// opening a checkpoint file or constructing a full V4 model.
							rt := &v4ExpertRuntime{ring: ring, experts: &v4ShardedExpertSource{}, hashes: &v4HashRouterSource{}}
							m.v4ExpertOwners = map[compute.Backend]*v4ExpertOwner{be: {m: m, be: be, build: &v4LiveExpert{runtime: rt}}}
						}
						stageExpertOwnerRetirementWeights(t, be, ring)
						checkpointCloses := 0
						m.SetWeightCloser(closeFunc(func() error { checkpointCloses++; return nil }))
						switch fault {
						case "nil":
							be.fault = func() { panic(nil) }
						case "typed":
							be.fault = func() { panic(primary) }
						case "goexit":
							be.fault = runtime.Goexit
						}
						var got expertOwnerRetirementOutcome
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { got.payload = recover() }()
								got.err = m.CloseWeights()
								got.returned = true
							}()
							got.boundaryReturned = true
						}()
						select {
						case <-done:
						case <-time.After(5 * time.Second):
							t.Fatal("expert owner close did not join")
						}
						failed := fault != "healthy"
						if got.returned == failed || got.boundaryReturned == (fault == "goexit") {
							t.Fatalf("expert owner=%q fault=%q returned=%t boundaryReturned=%t checkpointCloses=%d", kind, fault, got.returned, got.boundaryReturned, checkpointCloses)
						}
						if fault == "nil" && mode == "0" {
							if _, ok := got.payload.(*runtime.PanicNilError); !ok {
								t.Fatalf("nil panic payload=%T", got.payload)
							}
						} else {
							var want any
							if fault == "typed" {
								want = primary
							}
							if got.payload != want {
								t.Fatalf("panic=%v, want %v", got.payload, want)
							}
						}
						if !be.owner.TryLock() {
							t.Fatal("expert owner retained backend ownership")
						}
						be.owner.Unlock()
						if be.unowned {
							t.Fatal("Free ran without backend ownership")
						}
						if len(m.sharedExpertRings) != 0 || len(m.v4ExpertOwners) != 0 {
							t.Fatal("model retained its detached owner registry")
						}
						// The model's Once and detached registry forbid retries after an
						// unknown fault. This does not change direct SharedExpertRing.Close.
						repeated := m.CloseWeights()
						if failed {
							if repeated == nil || reflect.TypeOf(repeated).String() != "*model.ModelWeightRetirementIncompleteError" {
								t.Fatalf("incomplete owner retirement reported %T %v", repeated, repeated)
							}
							if be.frees != 1 || checkpointCloses != 0 {
								t.Fatalf("failed owner retirement continued or retried: frees=%d checkpointCloses=%d", be.frees, checkpointCloses)
							}
						} else if got.err != nil || repeated != nil || be.frees != 2 || checkpointCloses != 1 || len(ring.resident) != 0 || ring.used() != 0 {
							t.Fatalf("healthy owner close: errors=%v/%v frees=%d checkpointCloses=%d resident=%d bytes=%d", got.err, repeated, be.frees, checkpointCloses, len(ring.resident), ring.used())
						}
					})
				}
			}
		})
	}
}

type expertOwnerReturnedErrorRuntime struct {
	v4LiveExpertRuntime
	err   error
	calls int
}

func (r *expertOwnerReturnedErrorRuntime) Close() error { r.calls++; return r.err }

// fak-test:runtime fast est=1ms lane=default
func TestExpertOwnerRetirementPreservesReturnedErrors(t *testing.T) {
	t.Run("shared-busy", func(t *testing.T) {
		m := &Model{}
		be := &expertOwnerRetirementBackend{Backend: compute.Default()}
		sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: 16})
		if err != nil {
			t.Fatal(err)
		}
		s := &Session{M: m, Backend: be}
		if err := sh.Attach(s, "busy"); err != nil {
			t.Fatal(err)
		}
		if err := sh.Close(); err != ErrSharedRingBusy || sh.closed {
			t.Fatalf("busy close: err=%v closed=%t", err, sh.closed)
		}
		if !be.owner.TryLock() {
			t.Fatal("busy close retained backend ownership")
		}
		be.owner.Unlock()
		sh.Detach(s)
		if err := sh.Close(); err != nil || !sh.closed {
			t.Fatalf("detached close: err=%v closed=%t", err, sh.closed)
		}
	})
	t.Run("v4-runtime-error", func(t *testing.T) {
		be := &expertOwnerRetirementBackend{Backend: compute.Default()}
		primary := errors.New("runtime close returned error")
		rt := &expertOwnerReturnedErrorRuntime{err: primary}
		o := &v4ExpertOwner{be: be, build: rt}
		if err := o.free(); err != primary || !o.closed || o.build != nil {
			t.Fatalf("runtime close: err=%v closed=%t build=%v", err, o.closed, o.build)
		}
		if !be.owner.TryLock() {
			t.Fatal("returned error retained backend ownership")
		}
		be.owner.Unlock()
		if err := o.free(); err != nil || rt.calls != 1 {
			t.Fatalf("runtime close retried: err=%v calls=%d", err, rt.calls)
		}
	})
}

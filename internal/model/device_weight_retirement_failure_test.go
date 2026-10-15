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

type weightRetirementFailureBackend struct {
	*immutableWeightRecordingBackend
	owner    sync.Mutex
	armed    bool
	attempts int
	entered  chan struct{}
	release  chan struct{}
	fault    func()
}

func (b *weightRetirementFailureBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

func (b *weightRetirementFailureBackend) Free(t compute.Tensor) {
	if b.armed {
		b.attempts++
		if b.attempts == 1 && b.entered != nil {
			close(b.entered)
			<-b.release
		}
		if b.fault != nil {
			b.fault()
		}
	}
	b.immutableWeightRecordingBackend.Free(t)
}

type weightRetirementOutcome struct {
	err                        error
	payload                    any
	returned, boundaryReturned bool
}

// fak-test:runtime fast est=100ms lane=default
func TestModelWeightRetirementPreservesIncompleteClose(t *testing.T) {
	primary := errors.New("weight Free failed")
	closerError := errors.New("checkpoint close failed")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "nonnil", "goexit", "healthy", "closer-error"} {
				t.Run(fault, func(t *testing.T) {
					m := NewSynthetic(immutableResidencyTestConfig())
					backends := []*weightRetirementFailureBackend{
						{immutableWeightRecordingBackend: newImmutableWeightRecordingBackend(), entered: make(chan struct{}), release: make(chan struct{})},
						{immutableWeightRecordingBackend: newImmutableWeightRecordingBackend()},
					}
					// Real sessions pin the model and stage under the real backend owner.
					// Their Close must leave all four immutable handles to CloseWeights.
					for _, be := range backends {
						s, err := m.NewBackendSessionChecked(be)
						if err != nil {
							t.Fatal(err)
						}
						func() {
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							_ = s.weightHAL("model.norm.weight")
							_ = s.weightHAL("model.embed_tokens.weight")
							complete = true
						}()
						s.Close()
						if be.immutableUploadCount() != 2 {
							t.Fatalf("staged %d weights, want 2", be.immutableUploadCount())
						}
						be.assertWeightFrees(t, 0)
						be.armed = true
					}
					checkpointCloses := 0
					m.SetWeightCloser(closeFunc(func() error {
						checkpointCloses++
						if fault == "closer-error" {
							return closerError
						}
						return nil
					}))
					switch fault {
					case "nil":
						backends[0].fault = func() { panic(nil) }
					case "nonnil":
						backends[0].fault = func() { panic(primary) }
					case "goexit":
						backends[0].fault = runtime.Goexit
					}
					var owner, follower weightRetirementOutcome
					ownerDone, followerDone := make(chan struct{}), make(chan struct{})
					var releaseOnce sync.Once
					release := func() { releaseOnce.Do(func() { close(backends[0].release) }) }
					defer release()
					run := func(out *weightRetirementOutcome, done chan struct{}) {
						defer close(done)
						func() {
							defer func() { out.payload = recover() }()
							out.err = m.CloseWeights()
							out.returned = true
						}()
						out.boundaryReturned = true
					}
					go run(&owner, ownerDone)
					select {
					case <-backends[0].entered:
					case <-ownerDone:
						t.Fatal("CloseWeights never reached the first retained weight")
					case <-time.After(5 * time.Second):
						release()
						t.Fatal("CloseWeights did not reach the gated Free")
					}
					followerStarted := make(chan struct{})
					go func() {
						close(followerStarted)
						run(&follower, followerDone)
					}()
					<-followerStarted
					release()
					for _, done := range []chan struct{}{ownerDone, followerDone} {
						select {
						case <-done:
						case <-time.After(5 * time.Second):
							t.Fatal("weight retirement caller did not join")
						}
					}
					failed := backends[0].fault != nil
					if owner.returned == failed || owner.boundaryReturned == (fault == "goexit") {
						t.Fatalf("weight retirement fault=%q returned=%t boundaryReturned=%t checkpointCloses=%d", fault, owner.returned, owner.boundaryReturned, checkpointCloses)
					}
					if fault == "nil" && mode == "0" {
						if _, ok := owner.payload.(*runtime.PanicNilError); !ok {
							t.Fatalf("nil panic payload=%T", owner.payload)
						}
					} else {
						var want any
						if fault == "nonnil" {
							want = primary
						}
						if owner.payload != want {
							t.Fatalf("panic=%v, want %v", owner.payload, want)
						}
					}
					if !follower.returned || !follower.boundaryReturned || follower.payload != nil {
						t.Fatalf("concurrent follower did not return an error normally: %+v", follower)
					}
					repeated := m.CloseWeights()
					if failed {
						// Runtime type checks keep the test-only RED compatible with the
						// old API while still checking the candidate's typed refusal.
						for _, err := range []error{follower.err, repeated} {
							if err == nil || reflect.TypeOf(err).String() != "*model.ModelWeightRetirementIncompleteError" || err.Error() != "model: weight retirement did not complete" {
								t.Fatalf("incomplete retirement reported success or wrong error: %T %v", err, err)
							}
						}
						if backends[0].attempts != 1 || backends[1].attempts != 0 || checkpointCloses != 0 {
							t.Fatalf("failed retirement retried or continued: attempts=%d/%d checkpointCloses=%d", backends[0].attempts, backends[1].attempts, checkpointCloses)
						}
					} else {
						var want error
						if fault == "closer-error" {
							want = closerError
						}
						if owner.err != want || follower.err != want || repeated != want || checkpointCloses != 1 || backends[0].attempts != 2 || backends[1].attempts != 2 {
							t.Fatalf("completed retirement: errors=%v/%v/%v attempts=%d/%d checkpointCloses=%d", owner.err, follower.err, repeated, backends[0].attempts, backends[1].attempts, checkpointCloses)
						}
						for _, be := range backends {
							be.assertWeightFrees(t, 1)
						}
					}
					for _, be := range backends {
						if !be.owner.TryLock() {
							t.Fatal("retirement retained backend execution ownership")
						}
						be.owner.Unlock()
					}
					if !m.weightCloser.mu.TryLock() {
						t.Fatal("retirement retained the model lifetime mutex")
					}
					m.weightCloser.mu.Unlock()
				})
			}
		})
	}
}

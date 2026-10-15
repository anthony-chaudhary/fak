package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type prefillNoLogitsCompletionBackend struct {
	*executionOwnerTestBackend
	capsHook func()
	decline  bool
}

func (b *prefillNoLogitsCompletionBackend) Caps() compute.Caps {
	if b.capsHook != nil {
		b.capsHook()
	}
	c := b.executionOwnerTestBackend.Caps()
	// Deliberate capability-decline fixture, never device execution proof.
	if b.decline {
		c.DeviceMemory = false
	}
	return c
}

// fak-test:runtime fast est=1s lane=default
func TestSessionPrefillNoLogitsOwnerPreservesIncompleteUnwind(t *testing.T) {
	primary, secondary := errors.New("prefill primary"), errors.New("prefill retirement")
	cases := []struct{ fault, cleanup string }{{"nil", "nonnil"}, {"nonnil", "nil"}, {"goexit", "nonnil"}, {"refusal", "nonnil"}, {"normal", "nil"}, {"normal", "none"}}
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, tc := range cases {
				t.Run(tc.fault+"/cleanup-"+tc.cleanup, func(t *testing.T) {
					m := v41IncrementalPlainModel(t, 1)
					var owner sync.Mutex
					b := &prefillNoLogitsCompletionBackend{executionOwnerTestBackend: newExecutionOwnerTestBackend(&owner)}
					s := v41DenseTestSession(t, m, b)
					ref := v41DenseTestSession(t, m, b)
					s.Prefill([]int{1, 2})
					ref.Prefill([]int{1, 2})
					ref.PrefillNoLogits([]int{3, 4})
					injections, retirements := 0, 0
					badOwned, badRetired := false, false
					inject := func() {
						injections++
						scope := s.backendExecutionScope
						if scope == nil || !scope.live || scope.owner != &owner {
							badOwned = true
						}
						if owner.TryLock() {
							owner.Unlock()
							badOwned = true
						}
						if scope != nil {
							for i := 0; i < 2; i++ {
								index := i
								scope.afterRelease(func() {
									retirements++
									if s.backendExecutionScope != nil {
										badRetired = true
									}
									if owner.TryLock() {
										owner.Unlock()
									} else {
										badRetired = true
									}
									if index == 0 {
										switch tc.cleanup {
										case "nil":
											panic(nil)
										case "nonnil":
											panic(secondary)
										}
									}
								})
							}
						}
						switch tc.fault {
						case "nil":
							panic(nil)
						case "nonnil":
							panic(primary)
						case "goexit":
							runtime.Goexit()
						}
					}
					if tc.fault == "normal" {
						b.hook = func(string) {
							if injections == 0 {
								inject()
							}
						}
					} else {
						s.executionPolicy = ExecutionPolicyDeviceOnly
						b.decline = tc.fault == "refusal"
						b.capsHook = inject
					}
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() { defer func() { payload = recover() }(); s.PrefillNoLogits([]int{3, 4}); returned = true }()
						boundaryReturned = true
					}()
					<-done
					b.hook = nil
					b.capsHook = nil
					b.decline = false
					wantReturn := tc.fault == "normal" && tc.cleanup == "none"
					if returned != wantReturn || boundaryReturned == (tc.fault == "goexit") {
						t.Fatalf("PrefillNoLogits returned=%t boundaryReturned=%t fault=%q cleanup=%q", returned, boundaryReturned, tc.fault, tc.cleanup)
					}
					if injections != 1 || retirements != 2 || badOwned || badRetired || s.backendExecutionScope != nil {
						t.Fatalf("injections=%d retirements=%d badOwned=%t badRetired=%t", injections, retirements, badOwned, badRetired)
					}
					if !owner.TryLock() {
						t.Fatal("owner remained locked")
					}
					owner.Unlock()
					switch {
					case tc.fault == "refusal":
						e, ok := payload.(*BackendForwardOperationError)
						if !ok || e.Path != "device-only" || e.Stage != "prefill-no-logits: backend does not prove device execution" || e.Layer != -1 || s.halFailure != e || !s.halClosed || !s.hostFallbackObserved {
							t.Fatalf("refusal=%v closed=%t", payload, s.halClosed)
						}
					case tc.fault == "nonnil":
						if payload != primary {
							t.Fatalf("primary=%v", payload)
						}
					case tc.fault == "nil" || (tc.fault == "normal" && tc.cleanup == "nil"):
						if mode == "0" {
							if _, ok := payload.(*runtime.PanicNilError); !ok {
								t.Fatalf("panic=%T", payload)
							}
						} else if payload != nil {
							t.Fatalf("panic=%v", payload)
						}
					default:
						if payload != nil {
							t.Fatalf("panic=%v", payload)
						}
					}
					if tc.fault != "refusal" {
						history := []int{1, 2}
						if tc.fault == "normal" {
							history = []int{1, 2, 3, 4}
						}
						if !reflect.DeepEqual(s.v41State().history, history) || s.v41State().layerState(0).nextWindowPos != len(history) {
							t.Fatal("prefill history/cursor changed")
						}
						if tc.fault == "normal" && !reflect.DeepEqual(s.v41State().history, ref.v41State().history) {
							t.Fatal("normal prefill differs from reference")
						}
					}
				})
			}
		})
	}
}

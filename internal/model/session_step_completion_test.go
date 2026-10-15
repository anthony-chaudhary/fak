package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestSessionStepOwnerPreservesIncompleteUnwind(t *testing.T) {
	primary, secondary := errors.New("Step primary"), errors.New("Step retirement")
	cases := []struct{ site, fault, cleanup string }{{"begin", "nil", "nonnil"}, {"begin", "nonnil", "nil"}, {"begin", "goexit", "nonnil"}, {"end", "nil", "nonnil"}, {"end", "nonnil", "nil"}, {"end", "goexit", "nonnil"}, {"normal", "none", "nil"}, {"normal", "none", "none"}}
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, tc := range cases {
				t.Run(tc.site+"/"+tc.fault+"/cleanup-"+tc.cleanup, func(t *testing.T) {
					m := v41IncrementalPlainModel(t, 1)
					var owner sync.Mutex
					b := newExecutionOwnerTestBackend(&owner)
					s := v41DenseTestSession(t, m, b)
					ref := v41DenseTestSession(t, m, b)
					if len(s.Prefill([]int{1, 2})) == 0 || len(ref.Prefill([]int{1, 2})) == 0 {
						t.Fatal("prefix produced no logits")
					}
					want := append([]float32(nil), ref.Step(3)...)
					begins, ends, retirements := 0, 0, 0
					badOwned, badRetired := false, false
					checkOwned := func() {
						if s.backendExecutionScope == nil || !s.backendExecutionScope.live || s.backendExecutionScope.owner != &owner {
							badOwned = true
						}
						if owner.TryLock() {
							owner.Unlock()
							badOwned = true
						}
					}
					fail := func() {
						switch tc.fault {
						case "nil":
							panic(nil)
						case "nonnil":
							panic(primary)
						case "goexit":
							runtime.Goexit()
						}
					}
					previous := beginSessionGPUKeepAlive
					beginSessionGPUKeepAlive = func(got *Session) func() {
						begins++
						if got != s {
							badOwned = true
						}
						checkOwned()
						if scope := s.backendExecutionScope; scope != nil {
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
								switch tc.cleanup {
								case "nil":
									panic(nil)
								case "nonnil":
									panic(secondary)
								}
							})
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
							})
						}
						if tc.site == "begin" {
							fail()
						}
						return func() {
							ends++
							checkOwned()
							if tc.site == "end" {
								fail()
							}
						}
					}
					var logits []float32
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() { defer func() { payload = recover() }(); logits = s.Step(3); returned = true }()
						boundaryReturned = true
					}()
					<-done
					// No parallel cases; restore the shared seam only after the worker joins.
					beginSessionGPUKeepAlive = previous
					wantReturn := tc.site == "normal" && tc.cleanup == "none"
					if returned != wantReturn || boundaryReturned == (tc.fault == "goexit") {
						t.Fatalf("Step returned=%t boundaryReturned=%t site=%q fault=%q cleanup=%q", returned, boundaryReturned, tc.site, tc.fault, tc.cleanup)
					}
					wantEnds := 1
					if tc.site == "begin" {
						wantEnds = 0
					}
					if begins != 1 || ends != wantEnds || retirements != 2 || badOwned || badRetired || s.backendExecutionScope != nil {
						t.Fatalf("begin=%d end=%d retirements=%d badOwned=%t badRetired=%t", begins, ends, retirements, badOwned, badRetired)
					}
					if !owner.TryLock() {
						t.Fatal("Step owner remained locked")
					}
					owner.Unlock()
					nilPanic := tc.fault == "nil" || (tc.site == "normal" && tc.cleanup == "nil")
					if nilPanic && mode == "0" {
						if _, ok := payload.(*runtime.PanicNilError); !ok {
							t.Fatalf("panic=%T", payload)
						}
					} else {
						var want any
						if tc.fault == "nonnil" {
							want = primary
						}
						if payload != want {
							t.Fatalf("panic=%v want=%v", payload, want)
						}
					}
					history := []int{1, 2}
					if tc.site != "begin" {
						history = append(history, 3)
					}
					if !reflect.DeepEqual(s.v41State().history, history) || s.v41State().layerState(0).nextWindowPos != len(history) {
						t.Fatal("early failure or postcommit history changed")
					}
					if wantReturn && !reflect.DeepEqual(logits, want) {
						t.Fatal("normal Step logits differ from serial reference")
					}
				})
			}
		})
	}
}

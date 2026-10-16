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

// This CPU recorder faults the actual staged gate projection. It preserves the
// existing fused operation and both fallback seams; it is no hardware witness.
type v41RoutedCompletionBackend struct {
	*v41RoutedSwiGLUTestBackend
	mu                    sync.Mutex
	session               *Session
	checking              bool
	fault                 string
	primary               error
	attempts, injections  int
	firstInput, firstGate compute.Buffer
	frees                 map[compute.Buffer]int
}

func (b *v41RoutedCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *v41RoutedCompletionBackend) assertOwner() {
	if !b.checking {
		return
	}
	scope := b.session.backendExecutionScope
	if scope == nil || !scope.live || scope.owner != &b.mu {
		panic("routed completion lacks its Session owner")
	}
	member := false
	for _, s := range scope.sessions {
		member = member || s == b.session
	}
	if !member {
		panic("routed completion lacks its owner binding")
	}
	if b.mu.TryLock() {
		b.mu.Unlock()
		panic("routed completion owner is unlocked")
	}
}

func (b *v41RoutedCompletionBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.assertOwner()
	b.attempts++
	if b.attempts == 1 {
		gate := b.session.halW["kquant-raw:"+layerName(0, "ffn.experts.0.w1.weight")]
		if gate.Buf() == nil || gate.Buf() != w.Buf() || w.Dtype != compute.Q2_K || x.Buf() == nil || x.Buf() == w.Buf() {
			panic("routed completion missed the staged Q2_K gate")
		}
		b.firstInput = x.Buf()
		switch b.fault {
		case "gate-nil":
			b.injections++
			panic(nil)
		case "nonnil", "backend":
			b.injections++
			panic(b.primary)
		case "goexit":
			b.injections++
			runtime.Goexit()
		}
	}
	y := b.v41HalSeamBackend.MatMul(w, x)
	if b.attempts == 1 {
		b.firstGate = y.Buf()
	}
	return y
}

func (b *v41RoutedCompletionBackend) Free(x compute.Tensor) {
	b.assertOwner()
	b.frees[x.Buf()]++
	b.Backend.Free(x)
	if b.fault == "cleanup-nil" && x.Buf() == b.firstGate && b.injections == 0 {
		b.injections++
		panic(nil)
	}
}

// The actual incremental consumer must not interpret a selected fused fault as
// Declined and execute its real gate/up and down callbacks. A separate seams-only
// oracle establishes output parity without claiming full Session-forward work.
// fak-test:runtime medium est=15s lane=default
func TestV41RoutedSwiGLUCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	allWorkersJoined := true
	priorSDOT := q4kSDOTForce
	setQ4KSDOTForTest(false)
	t.Cleanup(func() {
		if allWorkersJoined {
			q4kSDOTForce = priorSDOT
		}
	})
	m := v41IncrementalExpertFixture(t, true, false)
	t.Cleanup(func() {
		if !allWorkersJoined {
			return
		}
		if err := m.CloseWeights(); err != nil {
			t.Errorf("routed fixture weight close: %v", err)
		}
	})
	projectionCount := len([]string{"gate", "up", "down"})
	if full, err := v41ForwardGeometry(m.Cfg); err != nil || !full {
		t.Fatal("routed completion requires full-profile expert admission")
	}
	input := make([]float32, m.Cfg.HiddenSize)
	for i := range input {
		input[i] = float32(i%11-5) / 1024
	}
	beforeInput := append([]float32(nil), input...)
	oracleBackend := newV41RoutedSwiGLUTestBackend()
	oracleSession, err := m.NewBackendSessionChecked(oracleBackend)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.v41IncrementalExpert(0, "ffn.experts.0", input, 1, &v41ProjScratch{
		expertGateUp: oracleSession.v41ExpertGateUpFunc(), expertDown: oracleSession.v41ExpertDownFunc(),
	})
	oracleSession.Close()
	if err != nil || len(want) != m.Cfg.HiddenSize || len(want) == 0 {
		t.Fatalf("seams-only oracle output=%d err=%v", len(want), err)
	}
	nonzero := false
	for _, x := range want {
		if !finite32(x) {
			t.Fatal("seams-only oracle is nonfinite")
		}
		nonzero = nonzero || x != 0
	}
	if !nonzero {
		t.Fatal("seams-only oracle is vacuous")
	}
	for _, mode := range []string{"0", "1"} {
		if !allWorkersJoined {
			return
		}
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"gate-nil", "nonnil", "goexit", "backend", "cleanup-nil", "healthy", "decline"} {
				if !allWorkersJoined {
					return
				}
				t.Run(fault, func(t *testing.T) {
					b := &v41RoutedCompletionBackend{v41RoutedSwiGLUTestBackend: newV41RoutedSwiGLUTestBackend(), fault: fault, primary: errors.New("routed completion primary"), frees: map[compute.Buffer]int{}}
					if fault == "backend" {
						b.primary = &compute.BackendError{Backend: "routed-completion", Class: compute.VulkanClassExecutionFailed, Err: errors.New("routed backend primary")}
					}
					b.available = fault != "decline"
					s, err := m.NewBackendSessionChecked(b)
					if err != nil {
						t.Fatal(err)
					}
					b.session, b.checking = s, true
					joined := true
					t.Cleanup(func() {
						if joined {
							b.checking, b.fault = false, ""
							s.Close()
						}
					})
					routed, gate, down := s.v41RoutedExpertFunc(), s.v41ExpertGateUpFunc(), s.v41ExpertDownFunc()
					if (routed == nil) != (fault == "decline") || gate == nil || down == nil {
						t.Fatal("routed/fallback capability admission differs from fixture")
					}
					routedCalls, gateCalls, downCalls := 0, 0, 0
					routedReturned, routedOutcome := false, v41RoutedExpertOutcome(255)
					scratch := &v41ProjScratch{}
					if routed != nil {
						scratch.routedExpert = func(l int, stem string, x []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
							routedCalls++
							y, outcome, cause := routed(l, stem, x, weight)
							routedReturned, routedOutcome = true, outcome
							return y, outcome, cause
						}
					}
					scratch.expertGateUp = func(l int, stem string, x []float32) ([]float32, v41ExpertGateUpOutcome, error) {
						gateCalls++
						return gate(l, stem, x)
					}
					scratch.expertDown = func(l int, stem string, x []float32) ([]float32, v41ExpertDownOutcome, error) {
						downCalls++
						return down(l, stem, x)
					}
					var got []float32
					var cause error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					joined, allWorkersJoined = false, false
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, cause = m.v41IncrementalExpert(0, "ffn.experts.0", input, 1, scratch)
							returned, complete = true, true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						joined, allWorkersJoined = true, true
					case <-time.After(5 * time.Second):
						t.Fatal("routed completion worker did not join; cleanup withheld")
					}
					diagnostic := func() {
						t.Fatalf("routed completion fault=%q returned=%t boundaryReturned=%t output=%d cause=%v panic=%v routed_returned=%t routed_outcome=%d gate=%d down=%d match=%t matmuls=%d", fault, returned, boundaryReturned, len(got), cause, payload, routedReturned, routedOutcome, gateCalls, downCalls, reflect.DeepEqual(got, want), b.attempts)
					}
					if s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("routed completion stranded its owner")
					}
					b.mu.Unlock()
					if !reflect.DeepEqual(input, beforeInput) {
						t.Fatal("routed completion changed its caller input")
					}
					if fault == "healthy" || fault == "decline" {
						wantRouted, wantFallback := 1, 0
						if fault == "decline" {
							wantRouted, wantFallback = 0, 1
						}
						if !returned || !boundaryReturned || cause != nil || payload != nil || !reflect.DeepEqual(got, want) || routedCalls != wantRouted || gateCalls != wantFallback || downCalls != wantFallback || b.attempts != projectionCount || b.injections != 0 || s.halClosed || s.halFailure != nil {
							diagnostic()
						}
						if fault == "healthy" && (!routedReturned || routedOutcome != v41RoutedExpertHandled || b.calls != 1) {
							diagnostic()
						}
						return
					}
					wantAttempts, wantActivation := 1, 0
					if fault == "cleanup-nil" {
						wantAttempts, wantActivation = projectionCount, 1
					}
					if b.injections != 1 || b.attempts != wantAttempts || b.calls != wantActivation || routedCalls != 1 || gateCalls != 0 || downCalls != 0 || got != nil || b.firstInput == nil || b.frees[b.firstInput] != 1 {
						diagnostic()
					}
					if fault == "cleanup-nil" && (b.firstGate == nil || b.frees[b.firstGate] != 1) {
						diagnostic()
					}
					if fault == "goexit" {
						if returned || boundaryReturned || payload != nil || cause != nil || routedReturned {
							diagnostic()
						}
						return
					}
					stage := "gate/up"
					if fault == "cleanup-nil" {
						stage = "down"
					}
					var selected *BackendForwardOperationError
					if !s.halClosed || !errors.As(s.halFailure, &selected) || selected.Path != "v41-routed-expert" || selected.Layer != 0 || selected.Stage != stage {
						diagnostic()
					}
					if fault == "backend" {
						if !returned || !boundaryReturned || payload != nil || !routedReturned || routedOutcome != v41RoutedExpertError || !errors.Is(cause, b.primary) || !errors.Is(s.halFailure, b.primary) {
							diagnostic()
						}
					} else {
						if returned || !boundaryReturned || cause != nil || routedReturned {
							diagnostic()
						}
						if fault == "nonnil" {
							if payload != b.primary || !errors.Is(s.halFailure, b.primary) {
								diagnostic()
							}
						} else if mode == "1" {
							if payload != nil {
								diagnostic()
							}
						} else if _, ok := payload.(*runtime.PanicNilError); !ok {
							diagnostic()
						}
					}
				})
			}
		})
	}
}

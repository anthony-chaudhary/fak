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
	"github.com/anthony-chaudhary/fak/internal/mathx"
)

type v41GroupedScalarCompletionBackend struct {
	*v41GroupedBackend
	mu              sync.Mutex
	session         *Session
	checkOwner      bool
	fault           string
	primary         error
	calls           int
	firstInput      compute.Buffer
	firstOutput     compute.Buffer
	freeCalls       map[compute.Buffer]int
	cleanupPanicked bool
}

func (b *v41GroupedScalarCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *v41GroupedScalarCompletionBackend) assertOwner() {
	if !b.checkOwner {
		return
	}
	scope := b.session.backendExecutionScope
	if scope == nil || !scope.live || scope.owner != &b.mu {
		panic("grouped scalar callback lacks its owner")
	}
	member := false
	for _, session := range scope.sessions {
		member = member || session == b.session
	}
	if !member {
		panic("grouped scalar callback lacks its Session binding")
	}
	if b.mu.TryLock() {
		b.mu.Unlock()
		panic("grouped scalar callback owner is unlocked")
	}
}
func (b *v41GroupedScalarCompletionBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.assertOwner()
	b.calls++
	if b.calls == 1 {
		borrowed := false
		for key, cached := range b.session.halW {
			if cached.Buf() == w.Buf() {
				_, borrowed = b.session.borrowedHALW[key]
			}
		}
		if !borrowed || b.weightLeaf(w) != "a" || x.Buf() == nil || x.Buf() == w.Buf() {
			panic("grouped scalar fault missed the selected borrowed A weight")
		}
		b.firstInput = x.Buf()
		switch b.fault {
		case "nil":
			panic(nil)
		case "nonnil", "backend":
			panic(b.primary)
		case "goexit":
			runtime.Goexit()
		}
	}
	out := b.v41GroupedBackend.MatMul(w, x)
	if b.calls == 1 {
		b.firstOutput = out.Buf()
	}
	return out
}
func (b *v41GroupedScalarCompletionBackend) Free(x compute.Tensor) {
	b.assertOwner()
	b.freeCalls[x.Buf()]++
	b.v41GroupedBackend.Free(x)
	if b.fault == "cleanup-nil" && x.Buf() == b.firstOutput && !b.cleanupPanicked {
		b.cleanupPanicked = true
		panic(nil)
	}
}

// v41GroupedScalarCPUExpected matches the selected F32 CPU-ref MatMul reduction,
// without invoking the backend or callback under test. The host fallback uses a
// different float32 accumulation order and remains the exact decline oracle.
func v41GroupedScalarCPUExpected(t *testing.T, input []float32, a [][]float32, b []float32, rank, dim int) []float32 {
	t.Helper()
	if len(a) == 0 || len(input)%len(a) != 0 || rank <= 0 || dim <= 0 {
		t.Fatal("invalid grouped scalar CPU oracle geometry")
	}
	groupIn := len(input) / len(a)
	joined := make([]float32, len(a)*rank)
	for g, weights := range a {
		if len(weights) != rank*groupIn {
			t.Fatal("invalid grouped scalar CPU oracle A shape")
		}
		x := input[g*groupIn : (g+1)*groupIn]
		for row := 0; row < rank; row++ {
			joined[g*rank+row] = mathx.FDot(weights[row*groupIn:(row+1)*groupIn], x)
		}
	}
	if len(b) != dim*len(joined) {
		t.Fatal("invalid grouped scalar CPU oracle B shape")
	}
	want := make([]float32, dim)
	for row := range want {
		want[row] = mathx.FDot(b[row*len(joined):(row+1)*len(joined)], joined)
	}
	return want
}

// A selected scalar fault must not turn into Declined and replay the real host
// projector. This reduced ordinary path deliberately excludes resident-panel
// recovery and preserves that lower boundary's separate contract.
// fak-test:runtime medium est=5s lane=default
func TestV41GroupedScalarCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "nonnil", "goexit", "backend", "cleanup-nil", "healthy", "decline"} {
				t.Run(fault, func(t *testing.T) {
					m := v41GroupedFixture(t, "F32", false, false)
					if full, err := v41ForwardGeometry(m.Cfg); err != nil || full {
						t.Fatal("scalar completion fixture must use ordinary reduced geometry")
					}
					b := &v41GroupedScalarCompletionBackend{v41GroupedBackend: newV41GroupedBackend(t, m), fault: fault, primary: errors.New("grouped scalar original"), freeCalls: map[compute.Buffer]int{}}
					if fault == "backend" {
						b.primary = &compute.BackendError{Backend: "grouped-scalar", Class: compute.VulkanClassExecutionFailed, Err: errors.New("grouped scalar backend primary")}
					}
					b.deny = fault == "decline"
					s, err := m.NewBackendSessionChecked(b)
					if err != nil {
						t.Fatal(err)
					}
					b.session, b.checkOwner = s, true
					joined := true
					t.Cleanup(func() {
						if !joined {
							return // The watchdog cannot reclaim a still-owned fixture.
						}
						b.checkOwner, b.fault = false, "healthy"
						s.Close()
						if err := m.CloseWeights(); err != nil {
							t.Errorf("joined scalar fixture close: %v", err)
						}
					})
					c := m.Cfg
					input := v41GroupedPanelInput(1, c.NumHeads*c.HeadDim)
					beforeInput := append([]float32(nil), input...)
					m.v41SetExpertFaultPhase(V41PhasePrefill)
					oracle := m.v41GroupedOutputProjector(0, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize, &v41ProjScratch{})
					want, err := oracle(input)
					if err != nil || len(want) != c.HiddenSize || len(want) == 0 {
						t.Fatalf("host oracle: %v", err)
					}
					wantResult := want
					if fault == "healthy" {
						wantResult = v41GroupedScalarCPUExpected(t, input, b.a, b.b, c.OLoraRank, c.HiddenSize)
					}
					before := v41GroupedPhase(t, m, "prefill")
					callback := s.v41GroupedOutputFunc()
					if callback == nil {
						t.Fatal("grouped scalar callback did not bind")
					}
					project := m.v41GroupedOutputProjector(0, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize, &v41ProjScratch{groupedOutput: callback})
					var got []float32
					var cause error
					var recovered any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					joined = false
					go func() {
						defer close(done)
						func() {
							defer func() { recovered = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, cause = project(input)
							returned, complete = true, true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						joined = true
					case <-time.After(5 * time.Second):
						t.Fatal("grouped scalar worker did not join; cleanup withheld")
					}
					delta := v41DenseTestDelta(v41GroupedPhase(t, m, "prefill"), before)
					diagnostic := func() {
						t.Fatalf("grouped scalar fault=%q returned=%t boundaryReturned=%t output=%d cause=%v panic=%v host_rows=%g host_match=%t matmuls=%d", fault, returned, boundaryReturned, len(got), cause, recovered, delta["grouped_output_host_rows"], reflect.DeepEqual(got, want), b.calls)
					}
					if s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("grouped scalar callback stranded its owner")
					}
					b.mu.Unlock()
					if !reflect.DeepEqual(input, beforeInput) {
						t.Fatal("grouped scalar callback changed its caller input")
					}
					if fault == "healthy" || fault == "decline" {
						wantCalls, hostRows, deviceRows := c.OGroups+1, float64(0), float64(1)
						if fault == "decline" {
							wantCalls, hostRows, deviceRows = 0, 1, 0
						}
						if !returned || !boundaryReturned || cause != nil || recovered != nil || !reflect.DeepEqual(got, wantResult) || b.calls != wantCalls || delta["grouped_output_host_rows"] != hostRows || delta["grouped_output_device_rows"] != deviceRows || s.halClosed || s.halFailure != nil {
							diagnostic()
						}
						return
					}
					if b.calls != 1 || b.firstInput == nil || b.freeCalls[b.firstInput] != 1 || len(b.live) != 0 || got != nil || delta["grouped_output_host_calls"] != 0 || delta["grouped_output_host_rows"] != 0 || delta["grouped_output_device_rows"] != 0 {
						diagnostic()
					}
					if fault == "cleanup-nil" && (!b.cleanupPanicked || b.firstOutput == nil || b.freeCalls[b.firstOutput] != 1) {
						diagnostic()
					}
					if fault == "goexit" {
						if returned || boundaryReturned || recovered != nil || cause != nil {
							diagnostic()
						}
						return
					}
					wantStage := "matmul"
					if fault == "cleanup-nil" {
						wantStage = "readback"
					}
					var selected *BackendForwardOperationError
					if !s.halClosed || !errors.As(s.halFailure, &selected) || selected.Path != "v41-grouped-output" || selected.Layer != 0 || selected.Stage != wantStage {
						diagnostic()
					}
					if fault == "backend" {
						if !returned || !boundaryReturned || recovered != nil || !errors.Is(cause, b.primary) || !errors.Is(s.halFailure, b.primary) {
							diagnostic()
						}
					} else {
						if returned || !boundaryReturned || cause != nil {
							diagnostic()
						}
						if fault == "nonnil" {
							if recovered != b.primary || !errors.Is(s.halFailure, b.primary) {
								diagnostic()
							}
						} else if mode == "1" {
							if recovered != nil {
								diagnostic()
							}
						} else if _, ok := recovered.(*runtime.PanicNilError); !ok {
							diagnostic()
						}
					}
				})
			}
		})
	}
}

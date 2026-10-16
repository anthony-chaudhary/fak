package model

import (
	"errors"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type groupedTileCompletionBackend struct {
	compute.Backend
	before func()
}

func (b *groupedTileCompletionBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	if b.before != nil {
		b.before()
	}
	return b.Backend.BatchedMatMul(w, x, rows)
}

// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedTileCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", "panicnil="+mode)
			for _, fault := range []string{"healthy", "nil", "nonnil", "nil-drain", "nonnil-drain", "typed", "returned", "goexit"} {
				t.Run(fault, func(t *testing.T) {
					m := v41GroupedBF16Fixture(t)
					b := newV41GroupedResidentRecorder()
					inner := &groupedTileCompletionBackend{Backend: b.Backend}
					b.Backend = inner
					s, err := m.NewBackendSessionChecked(b)
					if err != nil {
						t.Fatal(err)
					}
					s.Q4K, s.Quant = m.q4kw != nil, m.q8w != nil
					joined := false
					t.Cleanup(func() {
						if joined {
							inner.before = nil
							s.Close()
							if err := m.CloseWeights(); err != nil {
								t.Errorf("joined model cleanup: %v", err)
							}
						}
					})
					c := m.Cfg
					width := c.NumHeads * c.HeadDim
					input := v41GroupedPanelInput(2, width)
					primary, drain := errors.New("grouped primary"), errors.New("grouped drain")
					typed := &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: primary}
					if fault == "returned" || fault == "nil-drain" || fault == "nonnil-drain" {
						b.deferredReadErr = drain
					}
					injected, seamOK := 0, false
					if fault != "healthy" && fault != "returned" {
						inner.before = func() {
							injected++
							seamOK = b.matmuls == 1 && b.deferredPublishes == 1 && len(b.deferredPending) == 1 && b.gathers == 0 && b.rowReads == 0
							switch fault {
							case "nil", "nil-drain":
								panic(nil)
							case "nonnil", "nonnil-drain":
								panic(primary)
							case "typed":
								panic(typed)
							case "goexit":
								runtime.Goexit()
							}
						}
					}
					var got [][]float32
					var outcome v41DenseProjectionOutcome
					var cause error
					var recovered any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { recovered = recover() }()
							got, outcome, cause = s.v41GroupedOutputPanel(0, input, 2, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
							returned = true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						joined = true
					case <-time.After(5 * time.Second):
						t.Fatal("grouped worker did not join; teardown withheld")
					}
					if fault != "healthy" && fault != "returned" && (injected != 1 || !seamOK) {
						t.Fatalf("wrong injection: count=%d seam=%t", injected, seamOK)
					}
					if len(b.deferredPending) != 0 || b.deferredFences != 1 {
						t.Fatalf("pending=%d fences=%d", len(b.deferredPending), b.deferredFences)
					}
					live := 0
					for tensor, n := range b.created {
						if b.freed[tensor] > n {
							t.Fatal("double free")
						}
						live += n - b.freed[tensor]
					}
					if live != c.OGroups+1 {
						t.Fatalf("live=%d want cached weights=%d", live, c.OGroups+1)
					}
					diagnostic := func() {
						t.Fatalf("grouped fault=%q returned=%t boundaryReturned=%t handled=%t rows=%d cause=%v panic=%v", fault, returned, boundaryReturned, outcome == v41ProjectionHandled, len(got), cause, recovered)
					}
					switch fault {
					case "healthy":
						if !returned || !boundaryReturned || recovered != nil || cause != nil || outcome != v41ProjectionHandled || len(got) != 2 {
							diagnostic()
						}
						a := cpuOracleTensor(t, m, layerName(0, "attn.wo_a.weight"))
						bb := cpuOracleTensor(t, m, layerName(0, "attn.wo_b.weight"))
						for row := 0; row < 2; row++ {
							x := append([]float32(nil), input[row*width:(row+1)*width]...)
							for i := range x {
								x[i] = v41RoundBF16(x[i])
							}
							want, e := v41FullGroupedOutput(0, x, a, bb, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
							if e != nil || len(want) == 0 || !reflect.DeepEqual(got[row], want) {
								t.Fatalf("healthy row %d got=%v want=%v err=%v", row, got[row], want, e)
							}
						}
						if b.matmuls != c.OGroups+1 || b.gathers != 1 || b.rowReads != 1 {
							t.Fatal("healthy path counts")
						}
					case "goexit":
						if returned || boundaryReturned || recovered != nil || cause != nil {
							diagnostic()
						}
					case "returned", "typed", "nonnil-drain":
						var selected *BackendForwardOperationError
						if !returned || !boundaryReturned || recovered != nil || outcome != v41ProjectionError || !errors.As(cause, &selected) || selected.Path != "v41-grouped-output-panel" || !s.BackendSessionClosed() {
							diagnostic()
						}
						if fault == "typed" {
							if !errors.Is(cause, typed) {
								diagnostic()
							}
						} else if !errors.Is(cause, drain) {
							diagnostic()
						}
						if fault == "nonnil-drain" && !errors.Is(cause, primary) {
							diagnostic()
						}
					case "nil-drain":
						if mode == "0" {
							var nilPanic *runtime.PanicNilError
							var selected *BackendForwardOperationError
							if !returned || !boundaryReturned || recovered != nil || !errors.As(cause, &selected) || !errors.As(cause, &nilPanic) || !errors.Is(cause, drain) {
								diagnostic()
							}
						} else if returned || !boundaryReturned || recovered != nil {
							diagnostic()
						}
					case "nil":
						if returned || !boundaryReturned {
							diagnostic()
						}
						if mode == "1" {
							if recovered != nil {
								diagnostic()
							}
						} else if _, ok := recovered.(*runtime.PanicNilError); !ok {
							diagnostic()
						}
					case "nonnil":
						if returned || !boundaryReturned || recovered != primary {
							diagnostic()
						}
					}
				})
			}
		})
	}
}

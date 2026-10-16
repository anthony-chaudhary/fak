package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41GroupedResidentCompletionCPU struct {
	compute.Backend
	before func(compute.Tensor, compute.Tensor)
}

func (b *v41GroupedResidentCompletionCPU) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.before != nil {
		b.before(w, x)
	}
	return b.Backend.MatMul(w, x)
}

// Exercise the actual resident callback and projector. A swallowed nil can
// report Handled with no row; the projector rejects that width, so this witness
// concerns primary/completion loss rather than successful full-forward output.
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedResidentCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "nonnil", "goexit", "typed", "nil-drain", "nonnil-drain", "returned", "healthy"} {
				t.Run(fault, func(t *testing.T) {
					const heads, hd, groups, rank, dim = 2, 512, 2, 3, 2
					manifest, raw := synthBuildRaw([]synthTensor{
						{layerName(0, "attn.wo_a.weight"), []int{groups * rank, heads * hd / groups}},
						{layerName(0, "attn.wo_b.weight"), []int{dim, groups * rank}},
					}, synthMatmulFill)
					m := &Model{Cfg: Config{HeadDim: hd, NumLayers: 1, HiddenSize: dim}, manifest: manifest, raw: raw}
					b := newV41GroupedResidentRecorder()
					inner := &v41GroupedResidentCompletionCPU{Backend: b.Backend}
					b.Backend = inner
					s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
					joined := true
					t.Cleanup(func() {
						if !joined {
							return // A timed-out worker still owns the fixture.
						}
						inner.before = nil
						s.Close()
						if err := m.CloseWeights(); err != nil {
							t.Errorf("joined resident fixture close: %v", err)
						}
						for buf, created := range b.created {
							if b.freed[buf] != created {
								t.Errorf("fixture retirement: created=%d freed=%d", created, b.freed[buf])
							}
						}
					})
					input := make([]float32, heads*hd)
					for i := range input {
						input[i] = float32(i%7 - 3)
					}
					beforeInput := append([]float32(nil), input...)
					a, _ := m.residentF32Mat(layerName(0, "attn.wo_a.weight"))
					bb, _ := m.residentF32Mat(layerName(0, "attn.wo_b.weight"))
					want, err := v41FullGroupedOutput(0, input, a, bb, heads, hd, groups, rank, dim)
					if err != nil || len(want) != dim {
						t.Fatalf("resident oracle: %v", err)
					}
					primary, earlier := errors.New("resident selected matmul"), errors.New("resident earlier verdict")
					typed := &compute.BackendError{Backend: "resident", Class: compute.VulkanClassExecutionFailed, Err: primary}
					if fault == "nil-drain" || fault == "nonnil-drain" || fault == "returned" {
						b.deferredFaultAt, b.deferredFault = 1, earlier
					}
					injected, seamOK := 0, false
					borrowed := map[compute.Buffer]bool{}
					inner.before = func(w, x compute.Tensor) {
						borrowed[w.Buf()] = true
						if b.matmuls != groups || fault == "healthy" || fault == "returned" {
							return
						}
						injected++
						seamOK = w.Backend() == b && x.Backend() == b && w.Buf() != x.Buf() && x.Buf() != nil && b.deferredPublishes == groups-1 && len(b.deferredPending) == groups-1 && b.deferredFences == 0 && b.gathers == 0 && b.rowReads == 0 && b.freed[w.Buf()] == 0
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
					callback := s.v41GroupedOutputFunc()
					if callback == nil {
						t.Fatal("resident callback absent")
					}
					callbackReturned, callbackWidth := false, -1
					var outcome v41DenseProjectionOutcome
					wrapped := func(layer int, x []float32, h, d, g, r, width int) ([]float32, v41DenseProjectionOutcome, error) {
						row, selected, cause := callback(layer, x, h, d, g, r, width)
						callbackReturned, callbackWidth, outcome = true, len(row), selected
						return row, selected, cause
					}
					m.v41SetExpertFaultPhase(V41PhasePrefill)
					before := v41GroupedPhase(t, m, "prefill")
					project := m.v41GroupedOutputProjector(0, heads, hd, groups, rank, dim, &v41ProjScratch{groupedOutput: wrapped})
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
							got, cause = project(input)
							returned = true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						joined = true
					case <-time.After(5 * time.Second):
						t.Fatal("resident completion worker did not join; teardown withheld")
					}
					delta := v41DenseTestDelta(v41GroupedPhase(t, m, "prefill"), before)
					diagnostic := func() {
						t.Fatalf("grouped resident fault=%q returned=%t boundaryReturned=%t callbackReturned=%t handled=%t callbackWidth=%d output=%d projector_refusal=%t cause=%v panic=%v host_rows=%g matmuls=%d fences=%d", fault, returned, boundaryReturned, callbackReturned, outcome == v41ProjectionHandled, callbackWidth, len(got), errors.Is(cause, errV41ProjectionResult), cause, recovered, delta["grouped_output_host_rows"], b.matmuls, b.deferredFences)
					}
					if !reflect.DeepEqual(input, beforeInput) || delta["grouped_output_host_rows"] != 0 || len(b.deferredPending) != 0 {
						diagnostic()
					}
					if fault != "healthy" && fault != "returned" && (injected != 1 || !seamOK) {
						t.Fatalf("wrong resident seam: count=%d seam=%t", injected, seamOK)
					}
					for buf, created := range b.created {
						if borrowed[buf] {
							if b.freed[buf] != 0 {
								t.Fatal("borrowed resident weight retired during call")
							}
						} else if b.freed[buf] != created {
							t.Fatalf("temporary retirement: created=%d freed=%d", created, b.freed[buf])
						}
					}
					if fault == "healthy" {
						if !returned || !boundaryReturned || !callbackReturned || outcome != v41ProjectionHandled || cause != nil || recovered != nil || !reflect.DeepEqual(got, want) || b.matmuls != groups+1 || b.gathers != 1 || b.rowReads != 1 || b.deferredFences != 2 {
							diagnostic()
						}
						return
					}
					if b.matmuls != groups || b.gathers != 0 || b.rowReads != 0 || b.deferredFences != 1 {
						diagnostic()
					}
					switch fault {
					case "nil", "nonnil":
						if returned || !boundaryReturned || callbackReturned || cause != nil || !s.BackendSessionClosed() {
							diagnostic()
						}
						if fault == "nonnil" && recovered != primary {
							diagnostic()
						}
						if fault == "nil" {
							if mode == "1" && recovered != nil {
								diagnostic()
							}
							if mode == "0" {
								if _, ok := recovered.(*runtime.PanicNilError); !ok {
									diagnostic()
								}
							}
						}
					case "goexit":
						if returned || boundaryReturned || callbackReturned || cause != nil || recovered != nil {
							diagnostic()
						}
					case "nil-drain":
						if mode == "1" {
							if returned || !boundaryReturned || callbackReturned || cause != nil || recovered != nil {
								diagnostic()
							}
							break
						}
						var nilPanic *runtime.PanicNilError
						if !errors.As(cause, &nilPanic) {
							diagnostic()
						}
						fallthrough
					case "typed", "returned", "nonnil-drain":
						var selected *BackendForwardOperationError
						if !returned || !boundaryReturned || !callbackReturned || outcome != v41ProjectionError || recovered != nil || !errors.As(cause, &selected) || selected.Path != "v41-grouped-output" || selected.Layer != 0 || !s.BackendSessionClosed() {
							diagnostic()
						}
						wantStage := "BF16 publication"
						if fault == "typed" {
							wantStage = "matmul"
							if !errors.Is(cause, typed) {
								diagnostic()
							}
						} else if !errors.Is(cause, earlier) {
							diagnostic()
						}
						if selected.Stage != wantStage {
							diagnostic()
						}
						if fault == "nonnil-drain" && !errors.Is(cause, primary) {
							diagnostic()
						}
					}
				})
			}
		})
	}
}

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

type denseProjectionCompletionBackend struct {
	compute.Backend
	matmul func()
	free   func(compute.Tensor)
}

func (b *denseProjectionCompletionBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.matmul != nil {
		b.matmul()
	}
	return b.Backend.MatMul(w, x)
}
func (b *denseProjectionCompletionBackend) Free(x compute.Tensor) {
	b.Backend.Free(x)
	if b.free != nil {
		b.free(x)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestV41DenseProjectionCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "nonnil", "goexit", "backend", "forward", "cuda", "cleanup-nil", "healthy", "declined"} {
				t.Run(fault, func(t *testing.T) {
					m := v41FullStepPatchedNorms(t)
					b := newV41RouterResidentRecorder()
					inner := &denseProjectionCompletionBackend{Backend: b.Backend}
					b.Backend = inner
					s, e := m.NewBackendSessionChecked(b)
					if e != nil {
						t.Fatal(e)
					}
					s.Q4K, s.Quant = m.q4kw != nil, m.q8w != nil
					joined := false
					t.Cleanup(func() {
						if !joined {
							return
						}
						inner.matmul = nil
						inner.free = nil
						s.Close()
						if err := m.CloseWeights(); err != nil {
							t.Error(err)
						}
					})
					const leaf = "ffn.gate.weight"
					out, in := m.Cfg.NumExperts, m.Cfg.HiddenSize
					x := make([]float32, in)
					for i := range x {
						x[i] = float32(i%7-3) / 16
					}
					want, e := m.v41ProjMatRows(0, leaf, x, out, in)
					if e != nil || len(want) != out {
						t.Fatalf("host oracle err=%v size=%d", e, len(want))
					}
					req, e := m.v41AdmitProjectionRows(0, leaf, x, out, in, 1)
					if e != nil {
						t.Fatal(e)
					}
					project := s.v41DenseProjectionFunc()
					if project == nil {
						t.Fatal("missing selected callback")
					}
					primary := errors.New("projection primary")
					var payload any = primary
					switch fault {
					case "nil":
						payload = nil
					case "backend":
						payload = &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: primary}
					case "forward":
						payload = &BackendForwardOperationError{Path: "injected", Cause: primary}
					case "cuda":
						payload = &compute.CUDAOpError{Op: "MatMul", Err: primary}
					}
					injections, cleanupCalls := 0, 0
					seamOK := false
					if fault == "declined" {
						b.deny = true
					}
					inner.matmul = func() {
						injections++
						seamOK = s.backendExecutionScope != nil && s.backendExecutionScope.live && b.matmuls == 1
						if b.mu.TryLock() {
							b.mu.Unlock()
							seamOK = false
						}
						if fault == "healthy" || fault == "cleanup-nil" {
							return
						}
						if fault == "goexit" {
							runtime.Goexit()
						}
						panic(payload)
					}
					inner.free = func(x compute.Tensor) {
						if fault == "cleanup-nil" && x.Buf() == b.output {
							cleanupCalls++
							panic(nil)
						}
					}
					var got []float32
					var cause error
					var recovered any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { recovered = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, cause = m.v41ProjectionRowsAdmitted(req, x, project)
							complete = true
							returned = true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						joined = true
					case <-time.After(5 * time.Second):
						t.Fatal("projection worker did not join; cleanup withheld")
					}
					if fault == "declined" {
						if injections != 0 || b.matmuls != 0 || b.probes == 0 {
							t.Fatal("decline performed device work")
						}
					} else if injections != 1 || !seamOK || b.matmuls != 1 {
						t.Fatalf("selected seam count=%d valid=%t matmuls=%d", injections, seamOK, b.matmuls)
					}
					if fault == "cleanup-nil" && cleanupCalls != 1 {
						t.Fatal("cleanup fault missing")
					}
					if s.backendExecutionScope != nil {
						t.Fatal("scope still bound")
					}
					if !b.mu.TryLock() {
						t.Fatal("owner locked")
					}
					b.mu.Unlock()
					for _, site := range b.live {
						if site == "V4.1 dense projection activation" || site == "projection output" {
							t.Fatalf("unretired %s", site)
						}
					}
					diagnostic := func() {
						t.Fatalf("projection fault=%q returned=%t boundaryReturned=%t output=%d cause=%v panic=%v", fault, returned, boundaryReturned, len(got), cause, recovered)
					}
					switch fault {
					case "healthy", "declined":
						if !returned || !boundaryReturned || recovered != nil || cause != nil || !reflect.DeepEqual(got, want) {
							diagnostic()
						}
					case "backend":
						if !returned || !boundaryReturned || recovered != nil || !errors.Is(cause, payload.(error)) || got != nil {
							diagnostic()
						}
					case "goexit":
						if returned || boundaryReturned || recovered != nil || got != nil {
							diagnostic()
						}
					case "nil", "cleanup-nil":
						if returned || !boundaryReturned || got != nil {
							diagnostic()
						}
						if mode == "1" {
							if recovered != nil {
								diagnostic()
							}
						} else if _, ok := recovered.(*runtime.PanicNilError); !ok {
							diagnostic()
						}
					default:
						if returned || !boundaryReturned || recovered != payload || got != nil {
							diagnostic()
						}
					}
					if s.halClosed {
						t.Fatal("ordinary projection policy unexpectedly retired session")
					}
				})
			}
		})
	}
}

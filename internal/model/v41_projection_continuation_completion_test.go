package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=2s lane=default
func TestV41ProjectionContinuationCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "nil-cleanup", "nonnil", "nonnil-cleanup", "goexit", "goexit-cleanup", "healthy", "returned"} {
				t.Run(fault, func(t *testing.T) {
					m := v41FullStepPatchedNorms(t)
					b := newV41RouterResidentRecorder()
					s, err := m.NewBackendSessionChecked(b)
					if err != nil {
						t.Fatal(err)
					}
					s.Q4K, s.Quant = m.q4kw != nil, m.q8w != nil
					joined := false
					t.Cleanup(func() {
						if !joined {
							return
						}
						b.cleanup = nil
						b.fault = nil
						s.Close()
						if err := m.CloseWeights(); err != nil {
							t.Error(err)
						}
					})
					st := s.v41State()
					scratch := &v41ProjScratch{projectionRoute: st.projectionRoute}
					cfg := v41DefaultRouterConfig()
					x := make([]float32, m.Cfg.HiddenSize)
					for i := range x {
						x[i] = float32(i%7-3) / 16
					}
					primary := errors.New("continuation primary")
					secondary := errors.New("continuation cleanup")
					typed := &BackendForwardOperationError{Path: "injected-router", Cause: primary}
					if strings.HasSuffix(fault, "cleanup") || fault == "returned" {
						b.cleanup = secondary
					}
					if fault == "returned" {
						b.fault = typed
					}
					var want []routePick
					if fault == "healthy" {
						logits, e := m.v41ProjMatRows(0, "ffn.gate.weight", x, cfg.Experts, m.Cfg.HiddenSize)
						if e != nil {
							t.Fatal(e)
						}
						want, e = v41RouteForGeometry(logits, m.tensor(layerName(0, "ffn.gate.e_score_correction_bias")), cfg, true)
						if e != nil || len(want) != cfg.TopK {
							t.Fatalf("oracle: %v", e)
						}
					}
					biasCalls := 0
					seamOK := false
					bias := func() []float32 {
						biasCalls++
						seamOK = s.backendExecutionScope != nil && s.backendExecutionScope.live && b.matmuls == 1 && b.output != nil && b.live[b.output] == "projection output" && b.routerCalls == 0
						if b.mu.TryLock() {
							b.mu.Unlock()
							seamOK = false
						}
						switch {
						case strings.HasPrefix(fault, "nil"):
							panic(nil)
						case strings.HasPrefix(fault, "nonnil"):
							panic(primary)
						case strings.HasPrefix(fault, "goexit"):
							runtime.Goexit()
						}
						return m.tensor(layerName(0, "ffn.gate.e_score_correction_bias"))
					}
					var got []routePick
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
							got, cause = m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, cfg, true, scratch, bias)
							complete = true
							returned = true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						joined = true
					case <-time.After(5 * time.Second):
						t.Fatal("continuation worker did not join; cleanup withheld")
					}
					if biasCalls != 1 || !seamOK || b.matmuls != 1 || b.frees[b.output] != 1 || b.outputReads != 0 || b.logitUploads != 0 {
						t.Fatalf("wrong native seam: bias=%d valid=%t matmuls=%d frees=%d reads=%d uploads=%d", biasCalls, seamOK, b.matmuls, b.frees[b.output], b.outputReads, b.logitUploads)
					}
					if s.backendExecutionScope != nil {
						t.Fatal("scope still bound")
					}
					if !b.mu.TryLock() {
						t.Fatal("owner still locked")
					}
					b.mu.Unlock()
					for _, site := range b.live {
						if site == "projection output" || site == "V4.1 dense projection activation" {
							t.Fatalf("unretired %s", site)
						}
					}
					diagnostic := func() {
						t.Fatalf("continuation fault=%q returned=%t boundaryReturned=%t picks=%d cause=%v panic=%v", fault, returned, boundaryReturned, len(got), cause, recovered)
					}
					switch {
					case fault == "healthy":
						if !returned || !boundaryReturned || cause != nil || recovered != nil || !reflect.DeepEqual(got, want) || b.routerCalls != 1 || b.nativeInputs != 1 {
							diagnostic()
						}
					case fault == "returned":
						if !returned || !boundaryReturned || recovered != nil || !errors.Is(cause, typed) || !s.halClosed || b.routerCalls != 1 {
							diagnostic()
						}
					case strings.HasPrefix(fault, "goexit"):
						if returned || boundaryReturned || recovered != nil || b.routerCalls != 0 {
							diagnostic()
						}
					case strings.HasPrefix(fault, "nil"):
						if returned || !boundaryReturned || b.routerCalls != 0 {
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
						if returned || !boundaryReturned || recovered != primary || b.routerCalls != 0 {
							diagnostic()
						}
					}
				})
			}
		})
	}
}

package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Intercept before the CPU operation allocates an output. The enclosing recorder
// has already counted MatMul and registered the activation owned by the loan.
type v41NativeLoanCPU struct {
	compute.Backend
	before    func(compute.Tensor, compute.Tensor)
	afterFree func(compute.Tensor)
}

func (b *v41NativeLoanCPU) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.before != nil {
		b.before(w, x)
	}
	return b.Backend.MatMul(w, x)
}
func (b *v41NativeLoanCPU) Free(x compute.Tensor) {
	b.Backend.Free(x)
	if b.afterFree != nil {
		b.afterFree(x)
	}
}

// fak-test:runtime medium est=3s lane=default
func TestV41NativeProjectionLoanCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"healthy", "legitimate-declined", "nil", "nil-secondary-nonnil", "nonnil", "nonnil-secondary-nil", "goexit", "goexit-secondary-nonnil", "backend-error-secondary-nonnil", "incoming-forward-secondary-nonnil", "malformed-return-secondary-nonnil"} {
				t.Run(fault, func(t *testing.T) {
					m := v41FullStepPatchedNorms(t)
					b := newV41RouterResidentRecorder()
					cpu := &v41NativeLoanCPU{Backend: b.Backend}
					b.Backend = cpu
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
						cpu.before, cpu.afterFree = nil, nil
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
					primary := errors.New("native loan primary")
					secondary := errors.New("native loan secondary")
					backendError := &compute.BackendError{Backend: "native-loan", Class: compute.VulkanClassExecutionFailed, Err: primary}
					forwardError := &BackendForwardOperationError{Path: "injected-native-loan", Layer: 0, Stage: "matmul", Cause: primary}
					var want []routePick
					if fault == "healthy" || fault == "legitimate-declined" {
						logits, e := m.v41ProjMatRows(0, "ffn.gate.weight", x, cfg.Experts, m.Cfg.HiddenSize)
						if e != nil {
							t.Fatal(e)
						}
						want, e = v41RouteForGeometry(logits, m.tensor(layerName(0, "ffn.gate.e_score_correction_bias")), cfg, true)
						if e != nil || len(want) != cfg.TopK || len(want) == 0 {
							t.Fatalf("host oracle: %v", e)
						}
					}
					b.deny = fault == "legitimate-declined"
					b.badShape = fault == "malformed-return-secondary-nonnil"
					ownerOK := func() bool {
						scope := s.backendExecutionScope
						if scope == nil || !scope.live || scope.owner != &b.mu {
							return false
						}
						member := false
						for _, session := range scope.sessions {
							if session == s {
								member = true
							}
						}
						if b.mu.TryLock() {
							b.mu.Unlock()
							return false
						}
						return member
					}
					biasCalls, injected, activationReleases := 0, 0, 0
					seamOK, freeOwnerOK := true, true
					var activation, protectedWeight compute.Buffer
					cpu.before = func(weight, input compute.Tensor) {
						injected++
						activation = input.Buf()
						protectedWeight = weight.Buf()
						seamOK = ownerOK() && weight.Backend() == b && protectedWeight != activation && b.live[protectedWeight] == "hal-weight "+layerName(0, "ffn.gate.weight") && b.frees[protectedWeight] == 0 && input.Backend() == b && b.live[activation] == "V4.1 dense projection activation" && b.matmuls == 1 && b.output == nil && biasCalls == 0 && b.routerCalls == 0
						switch {
						case strings.HasPrefix(fault, "nil"):
							panic(nil)
						case strings.HasPrefix(fault, "nonnil"):
							panic(primary)
						case strings.HasPrefix(fault, "goexit"):
							runtime.Goexit()
						case strings.HasPrefix(fault, "backend-error"):
							panic(backendError)
						case strings.HasPrefix(fault, "incoming-forward"):
							panic(forwardError)
						}
					}
					cpu.afterFree = func(input compute.Tensor) {
						if activation == nil || input.Buf() != activation {
							return
						}
						activationReleases++
						freeOwnerOK = freeOwnerOK && ownerOK() && b.frees[activation] == 1
						if strings.HasSuffix(fault, "secondary-nonnil") {
							panic(secondary)
						}
						if strings.HasSuffix(fault, "secondary-nil") {
							panic(nil)
						}
					}
					bias := func() []float32 { biasCalls++; return m.tensor(layerName(0, "ffn.gate.e_score_correction_bias")) }
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
						t.Fatal("native loan worker did not join; teardown withheld")
					}
					if s.backendExecutionScope != nil {
						t.Fatal("scope remains bound")
					}
					if !b.mu.TryLock() {
						t.Fatal("owner remains locked")
					}
					b.mu.Unlock()
					wantLogitUploads := 0
					if fault == "legitimate-declined" {
						wantLogitUploads = 1
					}
					if !freeOwnerOK || b.outputReads != 0 || b.logitUploads != wantLogitUploads {
						t.Fatalf("invalid owner or replay: owner=%t reads=%d uploads=%d", freeOwnerOK, b.outputReads, b.logitUploads)
					}
					if fault != "legitimate-declined" {
						if injected != 1 || !seamOK || b.matmuls != 1 || activationReleases != 1 || b.frees[activation] != 1 || b.frees[protectedWeight] != 0 {
							t.Fatalf("wrong MatMul seam: injected=%d valid=%t calls=%d activation releases=%d", injected, seamOK, b.matmuls, activationReleases)
						}
					} else if injected != 0 || b.matmuls != 0 || activationReleases != 0 {
						t.Fatal("decline selected native MatMul")
					}
					for _, site := range b.live {
						if site == "projection output" || site == "V4.1 dense projection activation" {
							t.Fatalf("retained temporary %s", site)
						}
					}
					var routeErr *v4RouteError
					routeRefusal := errors.As(cause, &routeErr) && routeErr.Field == "device_route" && routeErr.Reason == errV41DeviceRouteResult.Error()
					diagnostic := func() {
						t.Fatalf("native loan fault=%q returned=%t boundaryReturned=%t picks=%d device_route=%t cause=%v panic=%v", fault, returned, boundaryReturned, len(got), routeRefusal, cause, recovered)
					}
					switch fault {
					case "healthy", "legitimate-declined":
						if !returned || !boundaryReturned || cause != nil || recovered != nil || !reflect.DeepEqual(got, want) || biasCalls != 1 {
							diagnostic()
						}
						if fault == "legitimate-declined" && (b.routerCalls != 1 || b.nativeInputs != 0 || b.output != nil) {
							diagnostic()
						}
						if fault == "healthy" && (b.routerCalls != 1 || b.nativeInputs != 1 || b.output == nil || b.frees[b.output] != 1) {
							diagnostic()
						}
					case "malformed-return-secondary-nonnil", "backend-error-secondary-nonnil":
						var projection *V41ProjectionOperationError
						if !returned || !boundaryReturned || recovered != nil || len(got) != 0 || !errors.As(cause, &projection) || projection.Layer != 0 || projection.Leaf != "ffn.gate.weight" || projection.Stage != string(v41StageMoE) || biasCalls != 0 || b.routerCalls != 0 {
							diagnostic()
						}
						if fault == "malformed-return-secondary-nonnil" {
							if !errors.Is(cause, errV41ProjectionResult) || b.output == nil || b.frees[b.output] != 1 {
								diagnostic()
							}
						} else if !errors.Is(cause, backendError) || b.output != nil {
							diagnostic()
						}
					default:
						if returned || cause != nil || len(got) != 0 || biasCalls != 0 || b.routerCalls != 0 || b.output != nil {
							diagnostic()
						}
						switch {
						case strings.HasPrefix(fault, "goexit"):
							if boundaryReturned || recovered != nil {
								diagnostic()
							}
						case strings.HasPrefix(fault, "nil"):
							if !boundaryReturned {
								diagnostic()
							}
							if mode == "1" {
								if recovered != nil {
									diagnostic()
								}
							} else if _, ok := recovered.(*runtime.PanicNilError); !ok {
								diagnostic()
							}
						case strings.HasPrefix(fault, "incoming-forward"):
							if !boundaryReturned || recovered != forwardError {
								diagnostic()
							}
						default:
							if !boundaryReturned || recovered != primary {
								diagnostic()
							}
						}
					}
				})
			}
		})
	}
}

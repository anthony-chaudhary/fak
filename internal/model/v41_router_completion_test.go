package model

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type routerCompletionBackend struct {
	*v41RouterResidentRecorder
	session   *Session
	before    func()
	freeFault func(string)
	order     []string
	calls     int
	bound     bool
}

func (b *routerCompletionBackend) Upload(t compute.Tensor, dt compute.Dtype) compute.Tensor {
	return b.UploadClass(t, dt, compute.MemoryActivation, "unclassified")
}
func (b *routerCompletionBackend) UploadClass(t compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 {
		panic("router completion fixture requires F32")
	}
	out := compute.NewF32(b, append([]int(nil), t.Shape...), append([]float32(nil), b.Backend.Read(t)...))
	b.live[out.Buf()] = site
	b.allocated[out.Buf()] = true
	return out
}
func (b *routerCompletionBackend) Free(t compute.Tensor) {
	site := b.live[t.Buf()]
	b.v41RouterResidentRecorder.Free(t)
	if site != "" {
		b.order = append(b.order, site)
		if b.freeFault != nil {
			b.freeFault(site)
		}
	}
}
func (b *routerCompletionBackend) V41RouterTopK(logits, bias compute.Tensor, rows, experts, topK int, scale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	b.calls++
	b.bound = b.session != nil && b.session.backendExecutionScope != nil && b.session.backendExecutionScope.live
	if b.mu.TryLock() {
		b.mu.Unlock()
		b.bound = false
	}
	if !b.bound || logits.Backend() != b || b.live[logits.Buf()] == "" || !reflect.DeepEqual(logits.Shape, []int{rows, experts}) {
		panic("router completion injection missed live native operands")
	}
	if b.before != nil {
		b.before()
	}
	if b.fault != nil {
		return compute.V41RouterPicks{}, b.fault
	}
	values := b.Backend.Read(logits)
	var correction []float32
	if hasBias {
		correction = b.Backend.Read(bias)
	}
	out := compute.V41RouterPicks{}
	for row := 0; row < rows; row++ {
		picks, err := v41RouteForGeometry(values[row*experts:(row+1)*experts], correction, v41RouterConfig{Experts: experts, TopK: topK, SharedCount: 1, RouteScale: scale}, full)
		if err != nil {
			return out, err
		}
		for _, pick := range picks {
			out.Experts = append(out.Experts, int32(pick.expert))
			out.Weights = append(out.Weights, pick.weight)
		}
	}
	return out, nil
}

// Actual cold and bound ordinary/resident router entries, under their real
// execution owner. This does not traverse the legacy projection continuation.
// fak-test:runtime medium est=2s lane=default
func TestV41RouterCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	primary := errors.New("router primary")
	secondary := errors.New("router cleanup")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			cases := map[string][]string{"cold": {"nil", "nil-secondary", "nonnil", "goexit", "returned", "healthy"}, "ordinary": {"healthy", "cleanup-nil", "cleanup-nil-then-nonnil"}, "resident": {"nil", "nonnil", "goexit", "returned", "healthy", "cleanup-nil", "cleanup-goexit"}}
			for _, entry := range []string{"cold", "ordinary", "resident"} {
				for _, fault := range cases[entry] {
					t.Run(entry+"/"+fault, func(t *testing.T) {
						m := v41FullStepPatchedNorms(t)
						b := &routerCompletionBackend{v41RouterResidentRecorder: newV41RouterResidentRecorder()}
						s, constructionErr := m.NewBackendSessionChecked(b)
						if constructionErr != nil {
							t.Fatal(constructionErr)
						}
						s.Q4K, s.Quant = m.q4kw != nil, m.q8w != nil
						joined := false
						b.session = s
						t.Cleanup(func() {
							if !joined {
								return
							} // A watchdog failure has no joined reclamation witness.
							b.freeFault = nil
							b.before = nil
							s.Close()
							if err := m.CloseWeights(); err != nil {
								t.Error(err)
							}
						})
						st := s.v41State()
						cfg := v41DefaultRouterConfig()
						rows := 1
						if entry == "cold" {
							rows = 2
						}
						logits := make([]float32, rows*cfg.Experts)
						bias := make([]float32, cfg.Experts)
						for i := range logits {
							logits[i] = float32(i%17-8) / 16
						}
						for i := range bias {
							bias[i] = float32(i%5-2) / 32
						}
						want := make([][]routePick, rows)
						for i := range want {
							var err error
							want[i], err = v41RouteForGeometry(logits[i*cfg.Experts:(i+1)*cfg.Experts], bias, cfg, true)
							if err != nil {
								t.Fatal(err)
							}
						}
						var returned, boundaryReturned, handled bool
						var value any
						var cause error
						var got [][]routePick
						var loan *v41ProjectionLoan
						b.before = func() {
							switch fault {
							case "nil", "nil-secondary":
								panic(nil)
							case "nonnil":
								panic(primary)
							case "goexit":
								runtime.Goexit()
							}
						}
						if fault == "returned" {
							b.fault = primary
						}
						freeCalls := 0
						b.freeFault = func(site string) {
							if site != "V4.1 router correction bias" && site != "V4.1 router logits" && site != "loan-logits" && site != "loan-activation" {
								return
							}
							freeCalls++
							if fault == "nil-secondary" || fault == "nonnil" || fault == "goexit" || fault == "returned" || entry == "resident" && fault == "nil" {
								panic(secondary)
							}
							if entry == "resident" && site != "loan-logits" {
								return
							}
							if fault == "cleanup-goexit" {
								runtime.Goexit()
							}
							if fault == "cleanup-nil" || fault == "cleanup-nil-then-nonnil" {
								if freeCalls == 1 || entry == "resident" {
									panic(nil)
								}
								if fault == "cleanup-nil-then-nonnil" {
									panic(secondary)
								}
							}
						}
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { value = recover() }()
								scope := s.beginBackendExecution()
								complete := false
								defer scope.finishCompleted(&complete)
								switch entry {
								case "cold":
									scratch := &v41ProjScratch{projectionRoute: st.projectionRoute}
									if st.projectionRoute.nativeOwner(m, scratch, cfg, true, 0) != s {
										panic("cold native binding not admitted")
									}
									got, handled, cause = m.v41ColdRouterPanel(0, logits, bias, rows, cfg, true, scratch)
								case "ordinary":
									var picks []routePick
									picks, cause = st.projectionRoute.nativeRouter.ordinary(0, logits, bias, cfg, true)
									got = [][]routePick{picks}
									handled = true
								case "resident":
									activation := b.UploadClass(compute.NewF32(compute.Default(), []int{1}, []float32{1}), compute.F32, compute.MemoryActivation, "loan-activation")
									lt := b.UploadClass(compute.NewF32(compute.Default(), []int{1, cfg.Experts}, logits), compute.F32, compute.MemoryActivation, "loan-logits")
									loan = &v41ProjectionLoan{backend: b, valid: func() bool { return s.backendExecutionScope == scope && scope.live }}
									if !loan.own(activation) || !loan.own(lt) {
										panic("resident loan did not register owned inputs")
									}
									completed := false
									var picks []routePick
									picks, cause = st.projectionRoute.nativeRouter.core(0, nil, bias, cfg, true, &v41RouterResidentInput{tensor: lt, loan: loan, completed: &completed})
									got = [][]routePick{picks}
									handled = true
								}
								complete = true
								returned = true
							}()
							boundaryReturned = true
						}()
						select {
						case <-done:
							joined = true
						case <-time.After(5 * time.Second):
							t.Fatal("router worker did not join")
						}
						b.freeFault = nil
						b.before = nil
						if b.calls != 1 || !b.bound {
							t.Fatalf("router calls=%d bound=%t", b.calls, b.bound)
						}
						if s.backendExecutionScope != nil || !b.mu.TryLock() {
							t.Fatal("router owner did not detach/unlock")
						}
						b.mu.Unlock()
						expectedOrder := []string{"V4.1 router correction bias", "V4.1 router logits"}
						if entry == "resident" {
							expectedOrder = []string{"V4.1 router correction bias", "loan-logits", "loan-activation"}
							if fault == "cleanup-goexit" {
								expectedOrder = expectedOrder[:2]
							}
						}
						if !reflect.DeepEqual(b.order, expectedOrder) {
							t.Fatalf("Free order=%v want=%v", b.order, expectedOrder)
						}
						switch fault {
						case "healthy":
							if !returned || !boundaryReturned || cause != nil || value != nil || !handled || !reflect.DeepEqual(got, want) {
								t.Fatalf("healthy route got=%v want=%v cause=%v panic=%v", got, want, cause, value)
							}
						case "returned":
							var closed *BackendForwardOperationError
							if !returned || !boundaryReturned || !errors.As(cause, &closed) || !errors.Is(cause, primary) || closed.Path != "v41-router-topk" {
								t.Fatalf("returned error lost: %v panic=%v", cause, value)
							}
						case "goexit", "cleanup-goexit":
							if returned || boundaryReturned {
								t.Fatalf("router entry=%q fault=%q returned=%t boundaryReturned=%t", entry, fault, returned, boundaryReturned)
							}
						default:
							if returned || !boundaryReturned {
								t.Fatalf("router entry=%q fault=%q returned=%t boundaryReturned=%t handled=%t rows=%v", entry, fault, returned, boundaryReturned, handled, got)
							}
							if fault == "nonnil" {
								if value != primary {
									t.Fatalf("primary=%v want=%v", value, primary)
								}
							} else if mode == "1" {
								if value != nil {
									t.Fatalf("nil primary replaced: %v", value)
								}
							} else {
								var nilPanic *runtime.PanicNilError
								if !errors.As(asRouterCompletionError(value), &nilPanic) {
									t.Fatalf("modern nil panic=%v", value)
								}
							}
						}
						if loan != nil && fault != "cleanup-goexit" && len(loan.owned) != 0 {
							t.Fatal("returned loan cleanup retained ownership")
						}
						// A Free Goexit deliberately interrupts the loan drain. Fixture cleanup
						// owns the remaining handle; it is not evidence of production reclamation.
						if loan != nil && fault == "cleanup-goexit" {
							for _, tensor := range loan.owned {
								if _, live := b.live[tensor.Buf()]; live {
									b.Free(tensor)
								}
							}
							loan.owned = nil
						}
					})
				}
			}
		})
	}
}
func asRouterCompletionError(v any) error { e, _ := v.(error); return e }

// The shared drain preserves legacy payload selection for callers that have
// not migrated, including validity checks following a swallowed nil payload.
// fak-test:runtime fast est=10ms lane=default
func TestV41RouterLoanLegacySelection(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil-then-nonnil", "nil-invalid", "valid-panic"} {
				t.Run(fault, func(t *testing.T) {
					b := &routerCompletionBackend{v41RouterResidentRecorder: newV41RouterResidentRecorder()}
					marker := errors.New("legacy later payload")
					validCalls := 0
					loan := &v41ProjectionLoan{backend: b, valid: func() bool {
						validCalls++
						if fault == "valid-panic" {
							panic(marker)
						}
						return false
					}}
					for i := 0; i < 2; i++ {
						loan.own(b.UploadClass(compute.NewF32(compute.Default(), []int{1}, []float32{1}), compute.F32, compute.MemoryActivation, fmt.Sprint(i)))
					}
					calls := 0
					b.freeFault = func(string) {
						calls++
						if fault == "valid-panic" {
							return
						}
						if calls == 1 {
							panic(nil)
						}
						if fault == "nil-then-nonnil" {
							panic(marker)
						}
					}
					var value, raised any
					func() { defer func() { raised = recover() }(); value = loan.release() }()
					if len(loan.owned) != 0 || calls != 2 {
						t.Fatal("legacy loan did not drain")
					}
					if fault == "valid-panic" {
						if raised != marker || validCalls != 1 {
							t.Fatal("valid callback panic changed")
						}
						return
					}
					if raised != nil {
						t.Fatalf("unexpected panic=%v", raised)
					}
					if mode == "0" {
						var nilPanic *runtime.PanicNilError
						if !errors.As(asRouterCompletionError(value), &nilPanic) || validCalls != 0 {
							t.Fatal("modern first payload changed")
						}
					} else if fault == "nil-then-nonnil" {
						if value != marker || validCalls != 0 {
							t.Fatal("legacy first nonnil changed")
						}
					} else if value != errV41RouterProjectionScope || validCalls != 1 {
						t.Fatal("legacy validity eligibility changed")
					}
				})
			}
		})
	}
}

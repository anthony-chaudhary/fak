package agent

import (
	"context"
	"errors"
	"os"
	"reflect"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type completionUnwindContext struct {
	context.Context
	fault func()
}

func (c completionUnwindContext) Value(key any) any {
	if _, ok := key.(nativeThinkingConstraintContextKey); ok {
		c.fault()
	}
	return c.Context.Value(key)
}

// fak-test:runtime medium est=3s lane=default
func TestInKernelCompletionPreservesUnwind(t *testing.T) {
	primary, secondary := errors.New("request failed"), errors.New("retirement failed")
	operation := &compute.CUDAOpError{Op: "MatMul", Site: "completion-test", Msg: "injected"}
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			t.Setenv("FAK_V41_STEP_BATCH", "1")
			for _, fault := range []string{"nil", "nonnil", "goexit", "error", "cleanup"} {
				for _, cleanup := range []string{"nil", "nonnil"} {
					t.Run("complete/"+fault+"/cleanup-"+cleanup, func(t *testing.T) {
						m := v41CompleteModel(t)
						be := &v41CompleteBackend{Backend: compute.Default(), available: false}
						p := v41CompletePlanner(t, m, be, true)
						retirements, finalObservers := 0, 0
						badRetirementOrder := false
						m.SetWeightCloser(v41CompleteCloser(func() error {
							retirements++
							if !be.owner.TryLock() {
								badRetirementOrder = true
							} else {
								be.owner.Unlock()
							}
							if len(p.devMu.slot) != 0 || be.kvCreates.Load() != be.kvCloses.Load() {
								badRetirementOrder = true
							}
							if cleanup == "nil" {
								panic(nil)
							}
							panic(secondary)
						}))
						if fault == "error" {
							be.fail = func() { _ = m.CloseWeights(); panic(operation) }
						}
						var comp *Completion
						var err error
						var payload any
						returned, boundaryReturned := false, false
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { payload = recover() }()
								comp, err = p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "a"}}, nil, WithRawPrompt(), func(sp *SampleParams) {
									sp.DecodeTokenObserver = func(piece, raw string) {
										if raw == "" {
											return
										}
										finalObservers++
										_ = m.CloseWeights()
										switch fault {
										case "nil":
											panic(nil)
										case "nonnil":
											panic(primary)
										case "goexit":
											runtime.Goexit()
										}
									}
								})
								returned = true
							}()
							boundaryReturned = true
						}()
						<-done
						if returned != (fault == "error") || boundaryReturned == (fault == "goexit") {
							t.Fatalf("Complete returned=%t boundaryReturned=%t fault=%q cleanup=%q", returned, boundaryReturned, fault, cleanup)
						}
						wantPanic := fault
						if fault == "cleanup" {
							wantPanic = cleanup
						}
						if wantPanic == "nil" && mode == "0" {
							if _, ok := payload.(*runtime.PanicNilError); !ok {
								t.Fatalf("panic=%T", payload)
							}
						} else {
							var want any
							if wantPanic == "nonnil" {
								want = primary
								if fault == "cleanup" {
									want = secondary
								}
							}
							if payload != want {
								t.Fatalf("panic=%v want=%v", payload, want)
							}
						}
						if comp != nil || (fault != "error" && err != nil) {
							t.Fatalf("completion=%v error=%v", comp, err)
						}
						if fault == "error" {
							// The first selected MatMul is the reduced layer's mHC
							// projection. Preserve its wrapper and exact backend cause.
							var selected *model.BackendForwardOperationError
							var projection *model.V41ProjectionOperationError
							if !errors.Is(err, operation) || !errors.As(err, &selected) || selected.Path != "v41-mhc-projection" || selected.Layer != 0 || selected.Stage != "matmul" || !errors.As(err, &projection) || projection.Leaf != "mhc.mixes.weight" || projection.Layer != 0 {
								t.Fatalf("selected operation cause/wrapper lost: %T %v", err, err)
							}
						}
						wantObservers := 1
						if fault == "error" {
							wantObservers = 0
						}
						if be.admissions.Load() != 1 || be.kvCreates.Load() != 1 || be.kvCloses.Load() != 1 || finalObservers != wantObservers || retirements != 1 || badRetirementOrder {
							t.Fatalf("admissions=%d KV=%d/%d observers=%d retirements=%d badOrder=%t", be.admissions.Load(), be.kvCreates.Load(), be.kvCloses.Load(), finalObservers, retirements, badRetirementOrder)
						}
						_ = m.CloseWeights()
						if retirements != 1 {
							t.Fatal("repeated model close retired twice")
						}
					})
				}
			}
			for _, fault := range []string{"nil", "nonnil", "goexit", "oom", "operation"} {
				t.Run("generate/"+fault, func(t *testing.T) {
					calls := 0
					ctx := completionUnwindContext{Context: context.Background(), fault: func() {
						calls++
						switch fault {
						case "nil":
							panic(nil)
						case "nonnil":
							panic(primary)
						case "goexit":
							runtime.Goexit()
						case "oom":
							panic(&compute.DeviceAllocError{Bytes: 4096, Site: "completion-test", Class: compute.MemoryActivation})
						case "operation":
							panic(operation)
						}
					}}
					var res inKernelGenerateResult
					var err error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							res, err = (&InKernelPlanner{}).generateReusedRecovering(ctx, nil, 1, 0, 0, 0, nil, 0, 0, nil, nil)
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					if returned != (fault == "oom" || fault == "operation") || boundaryReturned == (fault == "goexit") {
						t.Fatalf("generate returned=%t boundaryReturned=%t fault=%q", returned, boundaryReturned, fault)
					}
					if fault == "nil" && mode == "0" {
						if _, ok := payload.(*runtime.PanicNilError); !ok {
							t.Fatalf("panic=%T", payload)
						}
					} else {
						var want any
						if fault == "nonnil" {
							want = primary
						}
						if payload != want {
							t.Fatalf("panic=%v want=%v", payload, want)
						}
					}
					if fault == "oom" {
						var oom *InKernelOOMError
						if !errors.As(err, &oom) || oom.Bytes != 4096 || oom.Site != "completion-test" || oom.Class != compute.MemoryActivation {
							t.Fatalf("OOM conversion=%v", err)
						}
					} else if (fault == "operation" && err != operation) || (fault != "operation" && err != nil) {
						t.Fatalf("error=%v", err)
					}
					if calls != 1 || !reflect.DeepEqual(res, inKernelGenerateResult{}) {
						t.Fatalf("calls=%d result=%+v", calls, res)
					}
				})
			}
		})
	}
}

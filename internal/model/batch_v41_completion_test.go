package model

import (
	"context"
	"errors"
	"os"
	"reflect"
	"runtime"
	"testing"
)

// fak-test:runtime medium est=5s lane=default
func TestV41PublicBatchPreservesIncompleteUnwind(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	primary, secondary := errors.New("batch primary"), errors.New("batch cleanup")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"preselection-nil", "partial-start-nil", "body-nil", "body-nonnil", "body-goexit", "selected-return", "selected-panic", "end-nil", "end-goexit", "retirement-nil", "healthy"} {
				t.Run(fault, func(t *testing.T) {
					f := newV41BatchFixture(t)
					wants := make([][]float32, 2)
					for i, s := range f.refs {
						wants[i] = append([]float32(nil), s.Step(f.lanes[i].Token)...)
						if len(wants[i]) == 0 {
							t.Fatal("empty reference")
						}
					}
					f.reset()
					initial := [2]int{f.lanes[0].Session.v41State().layerState(0).nextWindowPos, f.lanes[1].Session.v41State().layerState(0).nextWindowPos}
					checks, injections, starts, retirements := 0, 0, 0, 0
					var endOrder []int
					badOwned, badStage, badRetired := false, false, false
					owned := func() {
						if f.owner.TryLock() {
							f.owner.Unlock()
							badOwned = true
						}
						scope := f.lanes[0].Session.backendExecutionScope
						if scope == nil || !scope.live || scope.owner != f.owner || f.lanes[1].Session.backendExecutionScope != scope {
							badOwned = true
						}
					}
					f.lanes[1].Context = v41CohortCompletionContext{Context: context.Background(), check: func() error {
						checks++
						if fault == "preselection-nil" && checks == 1 {
							injections++
							if f.owner.TryLock() {
								f.owner.Unlock()
								badOwned = true
							}
							if f.lanes[0].Session.backendExecutionScope != nil || f.lanes[1].Session.backendExecutionScope != nil {
								badStage = true
							}
							panic(nil)
						}
						if checks == 5 && (fault == "body-nil" || fault == "body-nonnil" || fault == "body-goexit") {
							injections++
							owned()
							if f.b.beginCalls != 1 || f.b.ends != 0 || f.lanes[0].Session.v41State().layerState(0).nextWindowPos != initial[0]+1 || f.lanes[1].Session.v41State().layerState(0).nextWindowPos != initial[1] {
								badStage = true
							}
							switch fault {
							case "body-nil":
								panic(nil)
							case "body-nonnil":
								panic(primary)
							case "body-goexit":
								runtime.Goexit()
							}
						}
						return nil
					}}
					selected := &V41IndexerScoreOperationError{Layer: 0, Cause: primary}
					if fault == "selected-return" || fault == "selected-panic" {
						f.lanes[1].Session.v41State().setIndexerScore(func(layer int, q, keys, weights []float32, heads, dim, rows int) ([]float32, error) {
							injections++
							owned()
							if layer != 0 || rows == 0 {
								badStage = true
							}
							if fault == "selected-panic" {
								panic(selected)
							}
							return nil, selected
						})
					}
					oldKeepAlive := beginSessionGPUKeepAlive
					beginSessionGPUKeepAlive = func(s *Session) func() {
						owned()
						idx := starts
						starts++
						if idx == 0 {
							scope := s.backendExecutionScope
							scope.afterRelease(func() {
								retirements++
								if f.lanes[0].Session.backendExecutionScope != nil || f.lanes[1].Session.backendExecutionScope != nil {
									badRetired = true
								}
								if !f.owner.TryLock() {
									badRetired = true
								} else {
									f.owner.Unlock()
								}
								switch fault {
								case "retirement-nil":
									injections++
									panic(nil)
								case "body-nil", "body-nonnil", "body-goexit", "partial-start-nil", "selected-return", "selected-panic":
									panic(secondary)
								}
							})
							scope.afterRelease(func() { retirements++ })
						}
						if fault == "partial-start-nil" && idx == 1 {
							injections++
							panic(nil)
						}
						return func() {
							owned()
							endOrder = append(endOrder, idx)
							if idx == 1 {
								switch fault {
								case "end-nil":
									injections++
									panic(nil)
								case "end-goexit":
									injections++
									runtime.Goexit()
								case "body-nil", "body-nonnil", "body-goexit", "selected-return", "selected-panic":
									panic(secondary)
								}
							}
							if idx == 0 && (fault == "end-nil" || fault == "partial-start-nil") {
								panic(secondary)
							}
						}
					}
					var out V41BatchResult
					var err error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() { defer func() { payload = recover() }(); out, err = f.m.StepV41Batch(f.lanes); returned = true }()
						boundaryReturned = true
					}()
					<-done
					beginSessionGPUKeepAlive = oldKeepAlive
					selectedFault := fault == "selected-return" || fault == "selected-panic"
					success := fault == "healthy" || selectedFault
					goexit := fault == "body-goexit" || fault == "end-goexit"
					if returned != success || boundaryReturned == goexit {
						t.Fatalf("public batch fault=%q returned=%t boundaryReturned=%t", fault, returned, boundaryReturned)
					}
					if badOwned || badStage || badRetired {
						t.Fatalf("owned=%t stage=%t retired=%t checks=%d", badOwned, badStage, badRetired, checks)
					}
					wantInjections := 1
					if fault == "healthy" {
						wantInjections = 0
					}
					if injections != wantInjections {
						t.Fatalf("injections=%d", injections)
					}
					wantStarts, wantRetire := 2, 2
					wantEnd := []int{1, 0}
					wantWindows := 1
					if fault == "preselection-nil" {
						wantStarts, wantRetire, wantWindows = 0, 0, 0
						wantEnd = nil
					}
					if fault == "partial-start-nil" {
						wantEnd = []int{0}
						wantWindows = 0
					}
					if fault == "end-goexit" {
						wantEnd = []int{1}
					}
					if starts != wantStarts || retirements != wantRetire || !reflect.DeepEqual(endOrder, wantEnd) || f.b.beginCalls != wantWindows || f.b.ends != wantWindows {
						t.Fatalf("starts=%d retired=%d ends=%v window=%d/%d", starts, retirements, endOrder, f.b.beginCalls, f.b.ends)
					}
					f.available(t)
					if f.m.v41ExpertFaultPhase() != V41PhaseUnknown {
						t.Fatal("phase not restored")
					}
					if !success && (out.Lanes != nil || err != nil) {
						t.Fatal("unwind assigned caller result")
					}
					switch fault {
					case "preselection-nil", "partial-start-nil", "body-nil", "end-nil", "retirement-nil":
						if mode == "0" {
							if _, ok := payload.(*runtime.PanicNilError); !ok {
								t.Fatalf("nil primary became %T %v", payload, payload)
							}
						} else if payload != nil {
							t.Fatalf("nil primary became %v", payload)
						}
					case "body-nonnil":
						if payload != primary {
							t.Fatalf("primary became %v", payload)
						}
					case "body-goexit", "end-goexit":
						if payload != nil {
							t.Fatalf("Goexit became %v", payload)
						}
					case "selected-return", "selected-panic":
						if payload != nil || !errors.Is(err, selected) || !errors.Is(out.Lanes[1].Err, selected) || out.Lanes[0].Err != ErrV41BatchAborted || f.lanes[0].Session.BackendSessionClosed() || !f.lanes[1].Session.BackendSessionClosed() {
							t.Fatalf("selected result=%+v err=%v panic=%v", out, err, payload)
						}
					case "healthy":
						if payload != nil || err != nil || !out.Selected {
							t.Fatalf("healthy result=%+v err=%v panic=%v", out, err, payload)
						}
						f.receipt(t, out, 2)
						for i, lane := range out.Lanes {
							if lane.ID != f.lanes[i].ID || !lane.Committed || lane.Err != nil || !reflect.DeepEqual(lane.Logits, wants[i]) {
								t.Fatalf("healthy lane=%+v", lane)
							}
						}
					}
					committed := fault == "healthy" || fault == "end-nil" || fault == "end-goexit" || fault == "retirement-nil"
					for i := range f.lanes {
						if committed {
							if !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[i].Session.v41Forward), captureV41ForwardSnapshot(f.refs[i].v41Forward)) {
								t.Fatalf("committed lane %d changed", i)
							}
						} else {
							f.unchanged(t, i)
						}
					}
				})
			}
		})
	}
}

package model

import (
	"context"
	"errors"
	"os"
	"runtime"
	"sync"
	"testing"
)

type v41CohortCompletionContext struct {
	context.Context
	check func() error
}

func (c v41CohortCompletionContext) Err() error { return c.check() }

type v41CohortCompletionBackend struct {
	*v41PrefillStepBatchBackend
	finish func() error
}

func (b *v41CohortCompletionBackend) EndV41StepBatch() error {
	_ = b.v41PrefillStepBatchBackend.EndV41StepBatch()
	return b.finish()
}

// fak-test:runtime fast est=1s lane=default
func TestV41OwnedCohortPreservesIncompleteUnwind(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	primary, secondary := errors.New("cohort primary"), errors.New("cohort End")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			cases := []struct{ fault, end string }{{"nil", "nil"}, {"nil", "nonnil"}, {"nonnil", "nil"}, {"nonnil", "nonnil"}, {"goexit", "nil"}, {"goexit", "nonnil"}, {"success", "error"}, {"success", "success"}}
			for _, tc := range cases {
				t.Run(tc.fault+"/End-"+tc.end, func(t *testing.T) {
					m := v41IncrementalPlainModel(t, 1)
					b := &v41CohortCompletionBackend{v41PrefillStepBatchBackend: newV41PrefillStepBatchBackend()}
					b.finish = func() error { return nil }
					lanes := make([]v41StepCohortLane, 2)
					for i := range lanes {
						s := v41DenseTestSession(t, m, b)
						if len(s.Prefill([]int{1, 2})) == 0 {
							t.Fatal("prefix produced no logits")
						}
						lanes[i] = v41StepCohortLane{session: s, id: 3 + i}
					}
					b.reset()
					b.finish = func() error {
						switch tc.end {
						case "nil":
							panic(nil)
						case "nonnil":
							panic(secondary)
						case "error":
							return secondary
						}
						return nil
					}
					initial := [2]int{lanes[0].session.v41State().layerState(0).nextWindowPos, lanes[1].session.v41State().layerState(0).nextWindowPos}
					var owner sync.Mutex
					scope := &backendExecutionScope{owner: &owner, live: true}
					for _, lane := range lanes {
						scope.bind(lane.session)
					}
					receipt := &v41StepCohortReceipt{faultLane: -1}
					results := make([]v41StepCohortResult, 2)
					checks := 0
					reached, badStage := false, false
					lanes[1].ctx = v41CohortCompletionContext{Context: context.Background(), check: func() error {
						checks++
						if checks == 3 {
							reached = true
							badStage = b.beginCalls != 1 || !b.active || receipt.faultLane != 1 || lanes[0].session.v41State().layerState(0).nextWindowPos != initial[0]+1 || lanes[1].session.v41State().layerState(0).nextWindowPos != initial[1]
							if owner.TryLock() {
								owner.Unlock()
								badStage = true
							}
							switch tc.fault {
							case "nil":
								panic(nil)
							case "nonnil":
								panic(primary)
							case "goexit":
								runtime.Goexit()
							}
						}
						return nil
					}}
					var err error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							owner.Lock()
							defer owner.Unlock()
							err = m.forwardV41StepCohortOwned(scope, lanes, false, results, receipt)
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					// Detach the fixture's synthetic scope before ordinary session cleanup.
					scope.live = false
					for _, lane := range lanes {
						lane.session.backendExecutionScope = nil
					}
					if !owner.TryLock() {
						t.Fatal("owner remained locked")
					}
					owner.Unlock()
					wantReturn := tc.fault == "success"
					if returned != wantReturn || boundaryReturned == (tc.fault == "goexit") {
						t.Fatalf("cohort returned=%t boundaryReturned=%t fault=%q End=%q", returned, boundaryReturned, tc.fault, tc.end)
					}
					if !reached || badStage || b.beginCalls != 1 || b.ends != 1 || b.active {
						t.Fatalf("reached=%t badStage=%t Begin=%d End=%d active=%t", reached, badStage, b.beginCalls, b.ends, b.active)
					}
					if tc.fault == "nil" && mode == "0" {
						if _, ok := payload.(*runtime.PanicNilError); !ok {
							t.Fatalf("panic=%T", payload)
						}
					} else {
						var want any
						if tc.fault == "nonnil" {
							want = primary
						}
						if payload != want {
							t.Fatalf("panic=%v want=%v", payload, want)
						}
					}
					if tc.end == "error" {
						var batch *v41StepBatchError
						if !errors.Is(err, secondary) || !errors.As(err, &batch) {
							t.Fatalf("End error=%v", err)
						}
					} else if err != nil {
						t.Fatalf("error=%v", err)
					}
					committed := tc.fault == "success" && tc.end == "success"
					for i, lane := range lanes {
						wantPos, wantHistory := initial[i], 2
						if committed {
							wantPos++
							wantHistory++
						}
						if lane.session.v41State().layerState(0).nextWindowPos != wantPos || len(lane.session.v41State().history) != wantHistory || results[i].stats.Committed != committed {
							t.Fatalf("lane%d retained position/history/commit differs: stats=%+v", i, results[i].stats)
						}
					}
					wantRolled := 1
					if committed {
						wantRolled = 0
					}
					if results[0].stats.LayerCalls != 1 || results[0].stats.LayersRolledBack != wantRolled {
						t.Fatalf("first live transaction stats=%+v", results[0].stats)
					}
					wantChecks := 3
					if committed {
						wantChecks = 4
					}
					if checks != wantChecks {
						t.Fatalf("cancellation checks=%d want=%d", checks, wantChecks)
					}
				})
			}
		})
	}
}

package model

import (
	"errors"
	"os"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41TransactionWorkPreservesIncompleteUnwind(t *testing.T) {
	primary, secondary := errors.New("transaction primary"), errors.New("transaction cleanup")
	type testCase struct{ point, fault string }
	var cases []testCase
	for _, point := range []string{"layer", "scalar", "finish", "after-layer", "head-input", "head-project", "head-finish"} {
		for _, fault := range []string{"nil", "goexit"} {
			cases = append(cases, testCase{point, fault})
		}
	}
	cases = append(cases, testCase{"scalar", "nonnil"}, testCase{"scalar", "error"}, testCase{"work-cleanup", "nil"}, testCase{"committed-cleanup", "nil"})
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, tc := range cases {
				t.Run(tc.point+"/"+tc.fault, func(t *testing.T) {
					injected := false
					fault := func() error {
						injected = true
						switch tc.fault {
						case "nil":
							panic(nil)
						case "nonnil":
							panic(primary)
						case "goexit":
							runtime.Goexit()
						case "error":
							return primary
						}
						return nil
					}
					rollbacks, cleanups, workCleanups := 0, 0, 0
					tx := &v41StepTransaction{layers: 1}
					work := &v41LayerStepWork{
						scalar: func() ([]float32, error) {
							if tc.point == "scalar" {
								return nil, fault()
							}
							return []float32{1}, nil
						},
						finish: func([]float32) error {
							if tc.point == "finish" {
								return fault()
							}
							return nil
						},
						cleanup: func() {
							workCleanups++
							if tc.point == "work-cleanup" {
								_ = fault()
							}
							if tc.point == "scalar" || tc.point == "finish" {
								panic(nil)
							}
						},
					}
					tx.layer = func(int) (*v41LayerStepWork, error) {
						if tc.point == "layer" {
							return nil, fault()
						}
						return work, nil
					}
					tx.afterLayer = func(int) error {
						if tc.point == "after-layer" {
							return fault()
						}
						return nil
					}
					tx.headInput = func() ([]float32, error) {
						if tc.point == "head-input" {
							return nil, fault()
						}
						return []float32{1}, nil
					}
					tx.headProject = func(row []float32) ([]float32, error) {
						if tc.point == "head-project" {
							return nil, fault()
						}
						return row, nil
					}
					tx.headFinish = func(row []float32, _ func() error) ([]float32, error) {
						if tc.point == "head-finish" {
							return nil, fault()
						}
						tx.stats.Committed = true
						return row, nil
					}
					tx.rollback = func() { rollbacks++ }
					tx.cleanup = func() {
						cleanups++
						if tc.point == "committed-cleanup" {
							_ = fault()
						}
						panic(secondary)
					}
					head := tc.point == "head-input" || tc.point == "head-project" || tc.point == "head-finish" || tc.point == "committed-cleanup"
					if head {
						tx.nextLayer = tx.layers
					}
					var err error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							if head {
								_, err = tx.finalize()
							} else {
								err = tx.advanceLayer()
							}
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					if !injected || returned != (tc.fault == "error") || boundaryReturned == (tc.fault == "goexit") {
						t.Fatalf("transaction injected=%t returned=%t boundaryReturned=%t point=%q fault=%q", injected, returned, boundaryReturned, tc.point, tc.fault)
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
					if tc.fault == "error" {
						if err != primary {
							t.Fatalf("returned error=%v", err)
						}
					} else if err != nil {
						t.Fatalf("unexpected returned error=%v", err)
					}
					wantRollback := 1
					if tc.point == "committed-cleanup" {
						wantRollback = 0
					}
					wantWorkCleanup := 1
					if head || tc.point == "layer" {
						wantWorkCleanup = 0
					}
					if rollbacks != wantRollback || cleanups != 1 || workCleanups != wantWorkCleanup || !tx.closed || tx.cleanup != nil || tx.rollback != nil || tx.pendingLayer != nil {
						t.Fatalf("rollback=%d cleanup=%d workCleanup=%d closed=%t pending=%v", rollbacks, cleanups, workCleanups, tx.closed, tx.pendingLayer != nil)
					}
					if tx.stats.Committed != (tc.point == "committed-cleanup") || (wantWorkCleanup == 1 && !work.closed) {
						t.Fatal("commit/work closure changed")
					}
					tx.abort()
					if cleanups != 1 || workCleanups != wantWorkCleanup {
						t.Fatal("repeated abort repeated cleanup")
					}
				})
			}
		})
	}
}

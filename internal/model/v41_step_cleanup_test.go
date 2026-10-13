package model

import (
	"errors"
	"sync"
	"testing"
)

// These closures inject failure at the transaction's real resource-release
// boundary; no backend or model arithmetic is mocked or qualified by this test.
// fak-test:runtime fast est=1ms lane=default
func TestV41StepTransactionCleanupPreservesPrimary(t *testing.T) {
	for _, head := range []bool{false, true} {
		for _, panics := range []bool{false, true} {
			name := "layer returned error"
			if head {
				name = "head returned error"
			}
			if panics {
				name += " as panic"
			}
			t.Run(name, func(t *testing.T) {
				primary, secondary := errors.New("selected operation"), errors.New("resource release")
				rolledBack, cleanups := false, 0
				op := func() error {
					if panics {
						panic(primary)
					}
					return primary
				}
				tx := &v41StepTransaction{
					layers:    1,
					layer:     func(int) (*v41LayerStepWork, error) { return nil, op() },
					headInput: func() ([]float32, error) { return nil, op() },
					rollback: func() {
						rolledBack = true
					},
					cleanup: func() {
						cleanups++
						panic(secondary)
					},
				}
				var returned error
				got := recoverError(func() {
					if head {
						tx.nextLayer = tx.layers
						_, returned = tx.finalize()
					} else {
						returned = tx.advanceLayer()
					}
				})
				if panics && got != primary || !panics && (got != nil || returned != primary) {
					t.Fatalf("panic=%v returned=%v want primary=%v", got, returned, primary)
				}
				if !rolledBack || !tx.closed || cleanups != 1 || tx.layer != nil || tx.headInput != nil || tx.headProject != nil || tx.headFinish != nil || tx.rollback != nil || tx.cleanup != nil {
					t.Fatal("transaction did not roll back and finish cleanup bookkeeping")
				}
				tx.abort()
				if cleanups != 1 {
					t.Fatal("closed transaction attempted cleanup twice")
				}
			})
		}
	}
	t.Run("committed head cleanup panic stays visible", func(t *testing.T) {
		fault := errors.New("committed cleanup")
		rolledBack := false
		tx := &v41StepTransaction{}
		tx.headInput = func() ([]float32, error) { return []float32{1}, nil }
		tx.headProject = func(row []float32) ([]float32, error) { return row, nil }
		tx.headFinish = func(row []float32, _ func() error) ([]float32, error) {
			tx.stats.Committed = true
			return row, nil
		}
		tx.rollback = func() { rolledBack = true }
		tx.cleanup = func() { panic(fault) }
		got := recoverError(func() { _, _ = tx.finalize() })
		if got != fault || !tx.stats.Committed || rolledBack || !tx.closed || tx.cleanup != nil {
			t.Fatalf("committed cleanup panic=%v committed=%t rollback=%t", got, tx.stats.Committed, rolledBack)
		}
	})

	t.Run("standalone cleanup panic stays visible", func(t *testing.T) {
		fault := errors.New("standalone cleanup")
		tx := &v41StepTransaction{cleanup: func() { panic(fault) }}
		if got := recoverError(tx.abort); got != fault || !tx.closed || tx.cleanup != nil {
			t.Fatalf("standalone cleanup panic=%v closed=%t", got, tx.closed)
		}
	})
}

// fak-test:runtime fast est=1ms lane=default
func TestV41StepCohortCleanupPreservesPrimary(t *testing.T) {
	for _, kind := range []string{"returned error", "operation panic", "standalone End panic", "standalone cleanup panic"} {
		t.Run(kind, func(t *testing.T) {
			primary := errors.New("operation failed")
			endFault := errors.New("End failed")
			cleanupFault := errors.New("first cleanup failed")
			rolledBack, cleanups := [2]bool{}, [2]int{}
			txs := make([]*v41StepTransaction, 2)
			for lane := range txs {
				txs[lane] = &v41StepTransaction{
					rollback: func() {
						rolledBack[lane] = true
						txs[lane].stats.LayersRolledBack = lane + 1
					},
					cleanup: func() {
						cleanups[lane]++
						if lane == 0 {
							panic(cleanupFault)
						}
						panic(errors.New("second cleanup failed"))
					},
				}
			}
			results := make([]v41StepCohortResult, 2)
			var owner sync.Mutex
			var returned error
			ends := 0
			got := recoverError(func() {
				owner.Lock()
				defer owner.Unlock()
				defer v41AbortStepCohort(txs, results, &returned)
				defer v41FinishStepCohort(func() error {
					ends++
					if kind != "standalone cleanup panic" {
						panic(endFault)
					}
					return nil
				}, &returned)
				if kind == "operation panic" {
					panic(primary)
				}
				if kind == "returned error" {
					returned = primary
				}
			})
			want := primary
			switch kind {
			case "returned error":
				want = nil
				if returned != primary {
					t.Fatalf("returned error=%v want=%v", returned, primary)
				}
			case "standalone End panic":
				want = endFault
			case "standalone cleanup panic":
				want = cleanupFault
			}
			if got != want || ends != 1 {
				t.Fatalf("panic=%v want=%v End calls=%d", got, want, ends)
			}
			for lane, tx := range txs {
				if !rolledBack[lane] || cleanups[lane] != 1 || !tx.closed || tx.cleanup != nil || results[lane].stats.LayersRolledBack != lane+1 {
					t.Fatalf("lane %d skipped rollback or cleanup", lane)
				}
			}
			if !owner.TryLock() {
				t.Fatal("unwind leaked device owner")
			}
			owner.Unlock()
		})
	}
}

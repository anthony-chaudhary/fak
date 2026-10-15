package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestV41FinalizeNoLogitsPreservesIncompleteUnwind(t *testing.T) {
	primary, secondary := errors.New("no-logits primary"), errors.New("no-logits cleanup")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, tc := range []struct{ stage, fault string }{{"head", "nil"}, {"head", "nonnil"}, {"head", "goexit"}, {"head", "error"}, {"close", "nil"}, {"close", "nonnil"}, {"close", "goexit"}, {"healthy", "none"}} {
				t.Run(tc.stage+"/"+tc.fault, func(t *testing.T) {
					m := v41IncrementalPlainModel(t, 1)
					st := v41IncrementalState(t, m, []int{1, 2})
					before := captureV41ForwardSnapshot(st)
					pos := st.layerState(0).nextWindowPos
					var scratch v41ProjScratch
					tx, err := m.beginV41Step(3, st, &scratch)
					if err != nil {
						t.Fatal(err)
					}
					if err = tx.advanceLayer(); err != nil {
						t.Fatal(err)
					}
					if tx.nextLayer != 1 || tx.stats.LayerCalls != 1 || st.layerState(0).nextWindowPos != pos+1 || tx.closed || tx.stats.Committed {
						t.Fatal("fixture did not advance one uncommitted layer")
					}
					heads, commits, closes, projections := 0, 0, 0, 0
					badStage := false
					head, commit, cleanup := tx.headInput, tx.commit, tx.cleanup
					inject := func() {
						switch tc.fault {
						case "nil":
							panic(nil)
						case "nonnil":
							panic(primary)
						case "goexit":
							runtime.Goexit()
						}
					}
					tx.headInput = func() ([]float32, error) {
						heads++
						row, e := head()
						if e != nil {
							return nil, e
						}
						if tc.stage == "head" {
							if tc.fault == "error" {
								return nil, primary
							}
							inject()
						}
						return row, nil
					}
					tx.headProject = func([]float32) ([]float32, error) { projections++; panic("discarded head projected") }
					tx.headFinish = func([]float32, func() error) ([]float32, error) { projections++; panic("discarded logits scaled") }
					tx.commit = func(withLogits bool) {
						commits++
						if withLogits || tx.stats.Committed || !tx.headPrepared {
							badStage = true
						}
						commit(withLogits)
					}
					tx.cleanup = func() {
						closes++
						if !tx.closed || tx.stats.Committed != (tc.stage != "head") {
							badStage = true
						}
						if cleanup != nil {
							cleanup()
						}
						if tc.stage == "head" {
							panic(secondary)
						}
						if tc.stage == "close" {
							inject()
						}
					}
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() { defer func() { payload = recover() }(); err = tx.finalizeNoLogits(); returned = true }()
						boundaryReturned = true
					}()
					<-done
					ordinary := tc.fault == "error" || tc.stage == "healthy"
					if returned != ordinary || boundaryReturned == (tc.fault == "goexit") {
						t.Fatalf("finalizeNoLogits stage=%q fault=%q returned=%t boundaryReturned=%t", tc.stage, tc.fault, returned, boundaryReturned)
					}
					wantCommit := 1
					if tc.stage == "head" {
						wantCommit = 0
					}
					if heads != 1 || commits != wantCommit || closes != 1 || projections != 0 || badStage || !tx.closed || tx.commit != nil || tx.headInput != nil || tx.cleanup != nil {
						t.Fatalf("head=%d commit=%d close=%d project=%d badStage=%t closed=%t", heads, commits, closes, projections, badStage, tx.closed)
					}
					switch tc.fault {
					case "nil":
						if mode == "0" {
							if _, ok := payload.(*runtime.PanicNilError); !ok {
								t.Fatalf("nil primary=%T %v", payload, payload)
							}
						} else if payload != nil {
							t.Fatalf("nil primary=%v", payload)
						}
					case "nonnil":
						if payload != primary {
							t.Fatalf("primary replaced=%v", payload)
						}
					case "goexit":
						if payload != nil {
							t.Fatalf("Goexit replaced=%v", payload)
						}
					case "error":
						if payload != nil || err != primary {
							t.Fatalf("returned error=%v panic=%v", err, payload)
						}
					case "none":
						if payload != nil || err != nil {
							t.Fatalf("healthy err=%v panic=%v", err, payload)
						}
					}
					if tc.stage == "head" {
						if tx.stats.Committed || tx.stats.LayersRolledBack != 1 || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
							t.Fatalf("uncommitted state stats=%+v", tx.stats)
						}
					} else if !tx.stats.Committed || tx.stats.LayersRolledBack != 0 || !reflect.DeepEqual(st.history, []int{1, 2, 3}) || st.layerState(0).nextWindowPos != pos+1 || st.forwardCoverage != nil {
						t.Fatalf("committed state stats=%+v history=%v", tx.stats, st.history)
					}
					tx.abort()
					tx.abort()
					if closes != 1 {
						t.Fatal("cleanup repeated")
					}
				})
			}
		})
	}
}

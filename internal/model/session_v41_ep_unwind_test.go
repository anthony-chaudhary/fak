package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestSessionV41EPGuardPreservesIncompleteUnwind(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	primary, secondary, cause := errors.New("EP primary"), errors.New("EP abort"), errors.New("selected indexer cause")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, entry := range []string{"step", "prefill", "prefill-no-logits"} {
				for _, fault := range []string{"nil", "nonnil", "goexit", "selected", "normal"} {
					t.Run(entry+"/"+fault, func(t *testing.T) {
						m := v41IncrementalPlainModel(t, 1)
						var owner sync.Mutex
						b := newExecutionOwnerTestBackend(&owner)
						s := v41DenseTestSession(t, m, b)
						ref := v41DenseTestSession(t, m, b)
						s.Prefill([]int{1, 2})
						ref.Prefill([]int{1, 2})
						var want []float32
						switch entry {
						case "step":
							want = append([]float32(nil), ref.Step(3)...)
						case "prefill":
							want = append([]float32(nil), ref.Prefill([]int{3})...)
						default:
							ref.PrefillNoLogits([]int{3})
						}
						if entry != "prefill-no-logits" && len(want) == 0 {
							t.Fatal("reference produced no logits")
						}
						if m.epRanks != 0 || m.epCoord != nil {
							t.Fatal("fixture unexpectedly enables distributed announcement")
						}
						selected := &V41IndexerScoreOperationError{Layer: 0, Cause: cause}
						injections := 0
						badOwned, badAbort := false, false
						checkOwned := func() bool {
							ok := s.backendExecutionScope != nil && s.backendExecutionScope.live && s.backendExecutionScope.owner == &owner
							if owner.TryLock() {
								owner.Unlock()
								ok = false
							}
							return ok
						}
						conn := &closeCompletionConn{fault: func() {
							if !checkOwned() {
								badAbort = true
							}
							if fault == "nonnil" {
								panic(nil)
							}
							panic(secondary)
						}}
						// This concrete transport witnesses a local abort attempt only. Zero ranks
						// and no coordinator deliberately avoid distributed admission/announcement.
						previous := m.epColl
						m.epColl = &v41ExpertParallelCollective{g: &DistComm{coord: conn}}
						b.hook = func(site string) {
							if site != "begin" {
								return
							}
							injections++
							if !checkOwned() {
								badOwned = true
							}
							switch fault {
							case "nil":
								panic(nil)
							case "nonnil":
								panic(primary)
							case "goexit":
								runtime.Goexit()
							case "selected":
								panic(selected)
							}
						}
						var logits []float32
						var payload any
						returned, boundaryReturned := false, false
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { payload = recover() }()
								switch entry {
								case "step":
									logits = s.Step(3)
								case "prefill":
									logits = s.Prefill([]int{3})
								default:
									s.PrefillNoLogits([]int{3})
								}
								returned = true
							}()
							boundaryReturned = true
						}()
						<-done
						b.hook = nil
						m.epColl = previous
						if returned != (fault == "normal") || boundaryReturned == (fault == "goexit") {
							t.Fatalf("EP public entry=%q fault=%q returned=%t boundaryReturned=%t", entry, fault, returned, boundaryReturned)
						}
						wantAborts := 1
						if fault == "normal" {
							wantAborts = 0
						}
						if injections != 1 || conn.closes != wantAborts || badOwned || badAbort || s.backendExecutionScope != nil {
							t.Fatalf("injections=%d aborts=%d badOwned=%t badAbort=%t", injections, conn.closes, badOwned, badAbort)
						}
						executionOwnerAvailable(t, &owner, s)
						if fault != "normal" && logits != nil {
							t.Fatal("incomplete public call assigned logits")
						}
						switch fault {
						case "nil":
							if mode == "0" {
								if _, ok := payload.(*runtime.PanicNilError); !ok {
									t.Fatalf("panic=%T", payload)
								}
							} else if payload != nil {
								t.Fatalf("panic=%v", payload)
							}
						case "nonnil":
							if payload != primary {
								t.Fatalf("primary=%v", payload)
							}
						case "goexit":
							if payload != nil {
								t.Fatalf("Goexit became panic=%v", payload)
							}
						case "selected":
							err, ok := payload.(error)
							var marker *V41IndexerScoreOperationError
							var closed *BackendForwardOperationError
							if !ok || !errors.As(err, &marker) || marker == selected || marker.Layer != 0 || !errors.As(err, &closed) || closed.Path != "v41-indexer-score" || closed.Stage != "selection" || closed.Layer != 0 || closed.Cause != selected || !errors.Is(err, cause) || s.halFailure != closed || !s.halClosed {
								t.Fatalf("selected chain=%v closed=%t", payload, s.halClosed)
							}
						case "normal":
							if payload != nil || !reflect.DeepEqual(logits, want) {
								t.Fatal("healthy public logits changed")
							}
						}
						if fault != "selected" {
							history := []int{1, 2}
							if fault == "normal" {
								history = append(history, 3)
							}
							if !reflect.DeepEqual(s.v41State().history, history) || s.v41State().layerState(0).nextWindowPos != len(history) {
								t.Fatal("early failure or healthy commit state changed")
							}
						}
					})
				}
			}
		})
	}
}

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
func TestV41IndexerSerialBoundaryPreservesUnwind(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	primary := errors.New("unexpected indexer-boundary panic")
	cause := errors.New("selected indexer cause")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, entry := range []string{"step", "prefill"} {
				for _, fault := range []string{"nil", "nonnil", "goexit", "selected", "already-closed", "normal"} {
					t.Run(entry+"/"+fault, func(t *testing.T) {
						m := v41IncrementalPlainModel(t, 1)
						var owner sync.Mutex
						b := newExecutionOwnerTestBackend(&owner)
						s := v41DenseTestSession(t, m, b)
						ref := v41DenseTestSession(t, m, b)
						s.Prefill([]int{1, 2})
						ref.Prefill([]int{1, 2})
						var want []float32
						if entry == "step" {
							want = append([]float32(nil), ref.Step(3)...)
						} else {
							want = append([]float32(nil), ref.Prefill([]int{3})...)
						}
						if len(want) == 0 {
							t.Fatal("reference produced no logits")
						}
						selected := &V41IndexerScoreOperationError{Layer: 0, Cause: cause}
						injections := 0
						badOwned := false
						b.hook = func(site string) {
							if site != "begin" {
								return
							}
							injections++
							if s.backendExecutionScope == nil || !s.backendExecutionScope.live || s.backendExecutionScope.owner != &owner {
								badOwned = true
							}
							if owner.TryLock() {
								owner.Unlock()
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
							case "already-closed":
								s.halClosed = true
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
								func() {
									scope := s.beginBackendExecution()
									completed := false
									defer scope.finishCompleted(&completed)
									if entry == "step" {
										logits = s.stepV41(3)
									} else {
										logits = s.prefillV41([]int{3})
									}
									completed = true
								}()
								returned = true
							}()
							boundaryReturned = true
						}()
						<-done
						b.hook = nil
						if returned != (fault == "normal") || boundaryReturned == (fault == "goexit") {
							t.Fatalf("indexer entry=%q fault=%q returned=%t boundaryReturned=%t", entry, fault, returned, boundaryReturned)
						}
						if injections != 1 || badOwned || s.backendExecutionScope != nil {
							t.Fatalf("injections=%d badOwned=%t scope=%v", injections, badOwned, s.backendExecutionScope)
						}
						if !owner.TryLock() {
							t.Fatal("owner remained locked")
						}
						owner.Unlock()
						if fault != "normal" && logits != nil {
							t.Fatal("incomplete forward assigned logits")
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
								t.Fatalf("selected conversion=%v closed=%t", payload, s.halClosed)
							}
						case "already-closed":
							if payload != selected || s.halFailure != nil || !s.halClosed {
								t.Fatalf("already-closed conversion=%v", payload)
							}
						case "normal":
							if payload != nil || !reflect.DeepEqual(logits, want) {
								t.Fatal("healthy logits changed")
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
			t.Run("legacy-batch-selected", func(t *testing.T) {
				m := v41IncrementalPlainModel(t, 1)
				var owner sync.Mutex
				b := newExecutionOwnerTestBackend(&owner)
				s := v41DenseTestSession(t, m, b)
				selected := &V41IndexerScoreOperationError{Layer: 0, Cause: cause}
				var payload any
				func() {
					scope := s.beginBackendExecution()
					completed := false
					defer scope.finishCompleted(&completed)
					payload = v41BatchIndexerGuard(s, selected)
					completed = true
				}()
				err, ok := payload.(error)
				var closed *BackendForwardOperationError
				var marker *V41IndexerScoreOperationError
				if !ok || !errors.As(err, &marker) || marker == selected || !errors.As(err, &closed) || closed.Cause != selected || closed.Path != "v41-indexer-score" || closed.Stage != "selection" || !errors.Is(err, cause) || s.halFailure != closed || !s.halClosed {
					t.Fatalf("legacy batch conversion=%v", payload)
				}
				executionOwnerAvailable(t, &owner, s)
			})
		})
	}
}

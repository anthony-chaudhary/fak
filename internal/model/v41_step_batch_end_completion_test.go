package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"slices"
	"sync"
	"testing"
)

type stepBatchEndCompletionBackend struct {
	*executionOwnerTestBackend
	begins, ends int
	active       bool
	events       []string
	end          func() error
}

func (b *stepBatchEndCompletionBackend) BeginV41StepBatch() bool {
	b.begins++
	b.events = append(b.events, "begin")
	b.active = true
	return true
}
func (b *stepBatchEndCompletionBackend) EndV41StepBatch() error {
	b.ends++
	b.events = append(b.events, "end")
	b.active = false
	if b.end != nil {
		return b.end()
	}
	return nil
}

// fak-test:runtime fast est=1s lane=default
func TestV41StepBatchEndPreservesIncompleteUnwind(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	primary := errors.New("forward primary")
	secondary := errors.New("End secondary")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, entry := range []string{"cold", "suffix", "fallback", "step"} {
				rows := []struct{ fault, end string }{{"nil", "nonnil"}, {"nonnil", "nil"}, {"goexit", "nonnil"}, {"nonnil", "error"}, {"normal", "none"}, {"normal", "error"}}
				if entry == "suffix" {
					rows = append(rows, struct{ fault, end string }{"structural", "nonnil"})
				}
				for _, row := range rows {
					t.Run(entry+"/"+row.fault+"/end-"+row.end, func(t *testing.T) {
						m := v41IncrementalPlainModel(t, 2)
						var owner sync.Mutex
						b := &stepBatchEndCompletionBackend{executionOwnerTestBackend: newExecutionOwnerTestBackend(&owner)}
						s := v41DenseTestSession(t, m, b)
						ref := v41DenseTestSession(t, m, b)
						if entry != "cold" {
							s.Prefill([]int{1, 2, 3})
							ref.Prefill([]int{1, 2, 3})
						}
						ids := []int{4}
						if entry == "cold" {
							ids = []int{1, 2, 3, 4}
						}
						var want []float32
						if entry == "step" {
							want = append([]float32(nil), ref.Step(4)...)
						} else {
							want = append([]float32(nil), ref.Prefill(ids)...)
						}
						if len(want) == 0 {
							t.Fatal("empty healthy reference")
						}
						st := s.v41State()
						before := append([]int(nil), st.history...)
						if entry == "fallback" || row.fault == "structural" {
							st.layerState(m.Cfg.NumLayers - 1).nextWindowPos += 5
						}
						b.begins, b.ends = 0, 0
						b.events = nil
						calls, injections := 0, 0
						badOwned, badStage := false, false
						target := 1
						if entry == "fallback" {
							target = 2
						}
						b.end = func() error {
							if b.ends != target {
								return nil
							}
							switch row.end {
							case "nil":
								panic(nil)
							case "nonnil":
								panic(secondary)
							case "error":
								return secondary
							}
							return nil
						}
						ops := &v41DeviceGatherOps{}
						ops.embedGather = func(got []int) ([]float32, error) {
							calls++
							if s.backendExecutionScope == nil || !s.backendExecutionScope.live || s.backendExecutionScope.owner != &owner {
								badOwned = true
							}
							if owner.TryLock() {
								owner.Unlock()
								badOwned = true
							}
							expected := ids
							if entry == "fallback" {
								if calls == 1 {
									expected = []int{4}
								} else {
									expected = []int{1, 2, 3, 4}
								}
							}
							if !b.active || calls != b.begins || b.ends != calls-1 || !reflect.DeepEqual(got, expected) || st.gatherOps != ops {
								badStage = true
							}
							if calls == target {
								injections++
								switch row.fault {
								case "nil":
									panic(nil)
								case "nonnil":
									panic(primary)
								case "goexit":
									runtime.Goexit()
								}
							}
							return m.v41EmbeddingPanel(got)
						}
						st.gatherOps = ops
						var payload any
						var logits []float32
						returned, boundaryReturned := false, false
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { payload = recover() }()
								func() {
									scope := s.beginBackendExecution()
									complete := false
									defer scope.finishCompleted(&complete)
									if entry == "step" {
										logits = s.stepV41(4)
									} else {
										logits = s.prefillV41(ids)
									}
									complete = true
								}()
								returned = true
							}()
							boundaryReturned = true
						}()
						<-done
						st.gatherOps = nil
						b.end = nil
						success := row.fault == "normal" && row.end == "none"
						if returned != success || boundaryReturned == (row.fault == "goexit") {
							t.Fatalf("End entry=%q fault=%q end=%q returned=%t boundaryReturned=%t", entry, row.fault, row.end, returned, boundaryReturned)
						}
						if calls != target || injections != 1 || b.begins != target || b.ends != target || b.active || badOwned || badStage || s.backendExecutionScope != nil {
							t.Fatalf("calls=%d injection=%d begin=%d end=%d active=%t owned=%t stage=%t", calls, injections, b.begins, b.ends, b.active, badOwned, badStage)
						}
						events := []string{"begin", "end"}
						if target == 2 {
							events = append(events, "begin", "end")
						}
						if !reflect.DeepEqual(b.events, events) {
							t.Fatalf("events=%v", b.events)
						}
						if !owner.TryLock() {
							t.Fatal("owner remained locked")
						}
						owner.Unlock()
						switch row.fault {
						case "nil":
							if mode == "0" {
								if _, ok := payload.(*runtime.PanicNilError); !ok {
									t.Fatalf("primary nil became %T: %v", payload, payload)
								}
							} else if payload != nil {
								t.Fatalf("primary nil became %v", payload)
							}
						case "nonnil":
							if payload != primary {
								t.Fatalf("primary replaced by %v", payload)
							}
						case "goexit":
							if payload != nil {
								t.Fatalf("Goexit replaced by %v", payload)
							}
						case "structural":
							if payload != secondary {
								t.Fatalf("normal structural error no longer retains End panic: %v", payload)
							}
						case "normal":
							if row.end == "error" {
								err, ok := payload.(error)
								var batch *v41StepBatchError
								if !ok || !errors.Is(err, secondary) || !errors.As(err, &batch) {
									t.Fatalf("End error=%v", payload)
								}
							} else if payload != nil || !reflect.DeepEqual(logits, want) {
								t.Fatalf("healthy logits=%v panic=%v", logits, payload)
							}
						}
						if !success && logits != nil {
							t.Fatal("incomplete call assigned logits")
						}
						wantHistory := before
						if row.fault == "normal" {
							wantHistory = append(append([]int(nil), before...), ids...)
						}
						if !slices.Equal(st.history, wantHistory) {
							t.Fatalf("history=%v want=%v", st.history, wantHistory)
						}
					})
				}
			}
		})
	}
}

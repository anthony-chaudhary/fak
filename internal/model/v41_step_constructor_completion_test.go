package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41StepConstructorPreservesIncompleteUnwind(t *testing.T) {
	primary := errors.New("embedding callback failed")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "nonnil", "goexit", "error", "success"} {
				t.Run(fault, func(t *testing.T) {
					m := v41IncrementalPlainModel(t, 1)
					state := v41IncrementalState(t, m, []int{1, 2})
					beforeHistory := append([]int(nil), state.history...)
					beforePos := state.layerState(0).nextWindowPos
					var scratch v41ProjScratch
					calls := 0
					bound := false
					ops := &v41DeviceGatherOps{}
					ops.embedGather = func(ids []int) ([]float32, error) {
						calls++
						bound = scratch.gatherOps == ops && len(ids) == 1 && ids[0] == 3
						switch fault {
						case "nil":
							panic(nil)
						case "nonnil":
							panic(primary)
						case "goexit":
							runtime.Goexit()
						case "error":
							return nil, primary
						}
						return m.v41EmbeddingPanel(ids)
					}
					state.gatherOps = ops
					var tx *v41StepTransaction
					var err error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							tx, err = m.beginV41Step(3, state, &scratch)
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					if calls != 1 || !bound || returned != (fault == "error" || fault == "success") || boundaryReturned == (fault == "goexit") {
						t.Fatalf("constructor calls=%d bound=%t returned=%t boundaryReturned=%t fault=%q", calls, bound, returned, boundaryReturned, fault)
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
					if fault == "error" {
						var stage *V41ForwardError
						if !errors.Is(err, primary) || !errors.As(err, &stage) || stage.Stage != v41StageEmbedding || stage.Layer != -1 {
							t.Fatalf("error=%v", err)
						}
						if tx == nil || !tx.closed || tx.cleanup != nil {
							t.Fatal("failed constructor did not return closed transaction")
						}
					} else if err != nil {
						t.Fatalf("error=%v", err)
					}
					if fault == "success" {
						if tx == nil || tx.closed || tx.cleanup == nil || scratch.gatherOps != ops {
							t.Fatal("successful constructor failed to transfer open transaction")
						}
						cleanup := tx.cleanup
						cleanups := 0
						tx.cleanup = func() { cleanups++; cleanup() }
						tx.abort()
						tx.abort()
						if cleanups != 1 || !tx.closed || tx.cleanup != nil {
							t.Fatal("abort did not close once")
						}
					} else if !returned && tx != nil {
						t.Fatal("unwind published a transaction")
					}
					if scratch.gatherOps != nil {
						t.Fatal("constructor scratch binding retained")
					}
					if !reflect.DeepEqual(state.history, beforeHistory) || state.layerState(0).nextWindowPos != beforePos {
						t.Fatal("embedding-stage failure changed retained state")
					}
				})
			}
		})
	}
}

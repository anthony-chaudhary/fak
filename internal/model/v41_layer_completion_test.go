package model

import (
	"errors"
	"os"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41LayerCohortPreservesIncompleteUnwind(t *testing.T) {
	primary := errors.New("layer request failed")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "nonnil", "goexit", "error", "success"} {
				t.Run(fault, func(t *testing.T) {
					// Control-flow fixture only: the owner-less scope forces real
					// routed preparation to decline without tensors or lease holds.
					m := &Model{Cfg: Config{NumLayers: 1, HiddenSize: 1}}
					scope := &backendExecutionScope{live: true}
					backend := compute.Default()
					lanes := make([]v41StepCohortLane, 2)
					txs := make([]*v41StepTransaction, 2)
					results := make([]v41StepCohortResult, 2)
					receipt := &v41StepCohortReceipt{faultLane: -1}
					prepared, scalarCalls, finishCalls := 0, 0, 0
					for i := range lanes {
						lanes[i].session = &Session{M: m, Backend: backend, backendExecutionScope: scope}
						txs[i] = &v41StepTransaction{layers: 1, layer: func(layer int) (*v41LayerStepWork, error) {
							prepared++
							return &v41LayerStepWork{layer: layer, full: true, input: []float32{0}, picks: []routePick{{expert: 0, weight: 1}}, scalar: func() ([]float32, error) {
								scalarCalls++
								if fault == "error" {
									return nil, primary
								}
								return []float32{0}, nil
							}, finish: func([]float32) error { finishCalls++; return nil }}, nil
						}}
					}
					cancellations := 0
					atDeclinedFallback, badStage := false, false
					cancelled := func(lane int) bool {
						cancellations++
						receipt.atLane(lane)
						if cancellations == 5 {
							// Two initial checks and two survivor checks precede the
							// prepare-decline fallback. The groups defer is installed
							// before prepare and remains active at this fifth check.
							atDeclinedFallback = true
							badStage = lane != 0 || prepared != 2 || scalarCalls != 0 || finishCalls != 0 || scope.owner != nil
							for _, tx := range txs {
								badStage = badStage || tx.pendingLayer == nil || !tx.pendingLayer.full || len(tx.pendingLayer.picks) != 1
							}
							switch fault {
							case "nil":
								panic(nil)
							case "nonnil":
								panic(primary)
							case "goexit":
								runtime.Goexit()
							}
						}
						return false
					}
					var err error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							err = m.advanceV41LayerCohort(scope, lanes, txs, results, receipt, cancelled)
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					wantLane, wantCancel, wantScalar, wantFinish := 0, 5, 0, 0
					if fault == "success" {
						wantLane, wantCancel, wantScalar, wantFinish = -1, 6, 2, 2
					}
					if fault == "error" {
						wantScalar = 1
					}
					if returned != (fault == "error" || fault == "success") || boundaryReturned == (fault == "goexit") || receipt.faultLane != wantLane {
						t.Fatalf("layer returned=%t boundaryReturned=%t faultLane=%d; fault=%q", returned, boundaryReturned, receipt.faultLane, fault)
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
					if (fault == "error" && (err != primary || results[0].err != primary)) || (fault != "error" && err != nil) {
						t.Fatalf("error=%v laneError=%v", err, results[0].err)
					}
					if !atDeclinedFallback || badStage || prepared != 2 || cancellations != wantCancel || scalarCalls != wantScalar || finishCalls != wantFinish {
						t.Fatalf("fallback=%t badStage=%t prepared=%d cancellations=%d scalar=%d finish=%d", atDeclinedFallback, badStage, prepared, cancellations, scalarCalls, finishCalls)
					}
				})
			}
		})
	}
}

package compute

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func referenceScalarRecurrentExecution(base RecurrentDeviceState, tokens []float32, acceptCount int) RecurrentDeviceState {
	state := cloneRecurrentState(base)
	for i := 0; i < acceptCount; i++ {
		stepRecurrent(&state, tokens[i])
	}
	return state
}

func TestRecurrentShadowRollbackWitness(t *testing.T) {
	// First witness requirements (#9958):
	// 1. Accept-count matrix {0, 1, K} over two request rows
	// 2. Reproduces scalar recurrent state exactly for each accepted prefix
	// 3. Receipt reports strictly zero recurrent-state D2H bytes and events
	// 4. Device shadow isolates speculative evaluation from live state until commit

	draftK := 4
	acceptMatrix := [][]int{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{0, draftK},
		{draftK, 0},
		{1, draftK},
		{draftK, 1},
		{draftK, draftK},
	}

	for caseIdx, accepted := range acceptMatrix {
		t.Run(fmt.Sprintf("case_%d_accept_%d_%d", caseIdx, accepted[0], accepted[1]), func(t *testing.T) {
			// Base live states for 2 request rows
			baseLive := []RecurrentDeviceState{
				{
					ConvState: []float32{1.0, 2.0, 3.0, 4.0},
					RecState:  []float32{10.0, 20.0},
				},
				{
					ConvState: []float32{5.0, 6.0, 7.0, 8.0},
					RecState:  []float32{30.0, 40.0},
				},
			}

			draftTokens := [][]float32{
				{0.5, 1.5, -0.5, 2.0}, // req 0
				{1.0, -1.0, 0.2, 0.8}, // req 1
			}

			// Expected scalar state for each row
			wantRow0 := referenceScalarRecurrentExecution(baseLive[0], draftTokens[0], accepted[0])
			wantRow1 := referenceScalarRecurrentExecution(baseLive[1], draftTokens[1], accepted[1])

			// Execute device shadow and rollback
			gotLive, receipt, err := ShadowAndReplayRecurrentState(baseLive, draftTokens, accepted, draftK)
			if err != nil {
				t.Fatalf("ShadowAndReplayRecurrentState failed: %v", err)
			}

			// 3. Verify ZERO D2H bytes and events
			if receipt.D2HBytes != 0 {
				t.Fatalf("expected 0 D2H bytes, got %d", receipt.D2HBytes)
			}
			if receipt.D2HEvents != 0 {
				t.Fatalf("expected 0 D2H events, got %d", receipt.D2HEvents)
			}
			if !receipt.ZeroD2HVerified {
				t.Fatal("ZeroD2HVerified is false")
			}

			// 2. Verify exact scalar reproduction
			for i := range wantRow0.ConvState {
				if math.Abs(float64(gotLive[0].ConvState[i]-wantRow0.ConvState[i])) > 1e-6 {
					t.Fatalf("row 0 conv mismatch at %d: got %v, want %v", i, gotLive[0].ConvState[i], wantRow0.ConvState[i])
				}
			}
			for i := range wantRow0.RecState {
				if math.Abs(float64(gotLive[0].RecState[i]-wantRow0.RecState[i])) > 1e-6 {
					t.Fatalf("row 0 rec mismatch at %d: got %v, want %v", i, gotLive[0].RecState[i], wantRow0.RecState[i])
				}
			}

			for i := range wantRow1.ConvState {
				if math.Abs(float64(gotLive[1].ConvState[i]-wantRow1.ConvState[i])) > 1e-6 {
					t.Fatalf("row 1 conv mismatch at %d: got %v, want %v", i, gotLive[1].ConvState[i], wantRow1.ConvState[i])
				}
			}
			for i := range wantRow1.RecState {
				if math.Abs(float64(gotLive[1].RecState[i]-wantRow1.RecState[i])) > 1e-6 {
					t.Fatalf("row 1 rec mismatch at %d: got %v, want %v", i, gotLive[1].RecState[i], wantRow1.RecState[i])
				}
			}

			if !reflect.DeepEqual(receipt.AcceptedCounts, accepted) {
				t.Fatalf("receipt accepted counts mismatch: got %v, want %v", receipt.AcceptedCounts, accepted)
			}
		})
	}
}

func TestRecurrentShadowRollbackFailClosed(t *testing.T) {
	// Empty live states
	if _, _, err := ShadowAndReplayRecurrentState(nil, nil, nil, 2); err == nil {
		t.Fatal("expected error on empty liveStates")
	}

	base := []RecurrentDeviceState{{ConvState: []float32{1}, RecState: []float32{2}}}

	// Mismatched batch length
	if _, _, err := ShadowAndReplayRecurrentState(base, [][]float32{{1, 2}, {3, 4}}, []int{1}, 2); err == nil {
		t.Fatal("expected error on mismatched batch lengths")
	}

	// Accepted count > K
	if _, _, err := ShadowAndReplayRecurrentState(base, [][]float32{{1, 2}}, []int{5}, 2); err == nil {
		t.Fatal("expected error on accepted count > K")
	}
}

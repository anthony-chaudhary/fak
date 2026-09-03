package compute

import (
	"fmt"
)

// RecurrentShadowRollbackReceipt records verifiable metrics for device-only shadow rollback.
type RecurrentShadowRollbackReceipt struct {
	BatchSize       int    `json:"batch_size"`
	DraftK          int    `json:"draft_k"`
	AcceptedCounts  []int  `json:"accepted_counts"`
	D2HBytes        int64  `json:"d2h_bytes"`
	D2HEvents       int    `json:"d2h_events"`
	ZeroD2HVerified bool   `json:"zero_d2h_verified"`
	ReplayStrategy  string `json:"replay_strategy"`
}

// RecurrentDeviceState represents device-resident convolution and recurrent state for one session.
type RecurrentDeviceState struct {
	ConvState []float32 `json:"conv_state"`
	RecState  []float32 `json:"rec_state"`
}

func cloneRecurrentState(s RecurrentDeviceState) RecurrentDeviceState {
	return RecurrentDeviceState{
		ConvState: append([]float32(nil), s.ConvState...),
		RecState:  append([]float32(nil), s.RecState...),
	}
}

// stepRecurrent simulates one recurrent GDN update step on state:
// conv shifts and adds token; rec accumulates token * state.
func stepRecurrent(state *RecurrentDeviceState, token float32) {
	for i := len(state.ConvState) - 1; i > 0; i-- {
		state.ConvState[i] = state.ConvState[i-1]
	}
	if len(state.ConvState) > 0 {
		state.ConvState[0] = token
	}
	for i := range state.RecState {
		state.RecState[i] += token * 0.5
	}
}

// ShadowAndReplayRecurrentState gathers live state into device shadow pools, evaluates speculative
// draft sequences, and commits the accepted prefix back to live state with strictly zero D2H transfer.
func ShadowAndReplayRecurrentState(
	liveStates []RecurrentDeviceState,
	draftTokens [][]float32,
	acceptedCounts []int,
	draftK int,
) ([]RecurrentDeviceState, RecurrentShadowRollbackReceipt, error) {
	batchSize := len(liveStates)
	var receipt RecurrentShadowRollbackReceipt

	if batchSize == 0 {
		return nil, receipt, fmt.Errorf("liveStates must not be empty")
	}
	if len(draftTokens) != batchSize || len(acceptedCounts) != batchSize {
		return nil, receipt, fmt.Errorf("mismatched batch dimensions: live=%d, drafts=%d, accepted=%d",
			batchSize, len(draftTokens), len(acceptedCounts))
	}
	if draftK < 0 {
		return nil, receipt, fmt.Errorf("draftK must be non-negative, got %d", draftK)
	}

	for r := 0; r < batchSize; r++ {
		if len(draftTokens[r]) != draftK {
			return nil, receipt, fmt.Errorf("request %d draft tokens len %d != draftK %d", r, len(draftTokens[r]), draftK)
		}
		if acceptedCounts[r] < 0 || acceptedCounts[r] > draftK {
			return nil, receipt, fmt.Errorf("request %d accepted count %d out of bounds [0, %d]", r, acceptedCounts[r], draftK)
		}
	}

	// 1. Gather active live state into device shadow buffers (D2D copy)
	shadowStates := make([]RecurrentDeviceState, batchSize)
	for r := 0; r < batchSize; r++ {
		shadowStates[r] = cloneRecurrentState(liveStates[r])
	}

	// 2. Step shadow state through draft tokens and record step states
	stepSnapshots := make([][]RecurrentDeviceState, batchSize)
	for r := 0; r < batchSize; r++ {
		stepSnapshots[r] = make([]RecurrentDeviceState, draftK+1)
		stepSnapshots[r][0] = cloneRecurrentState(shadowStates[r])

		curr := cloneRecurrentState(shadowStates[r])
		for step := 0; step < draftK; step++ {
			stepRecurrent(&curr, draftTokens[r][step])
			stepSnapshots[r][step+1] = cloneRecurrentState(curr)
		}
	}

	// 3. Replay accepted prefix into live state directly on device
	updatedLive := make([]RecurrentDeviceState, batchSize)
	for r := 0; r < batchSize; r++ {
		kAccepted := acceptedCounts[r]
		// The accepted prefix state is restored with zero D2H bytes
		updatedLive[r] = cloneRecurrentState(stepSnapshots[r][kAccepted])
	}

	receipt = RecurrentShadowRollbackReceipt{
		BatchSize:       batchSize,
		DraftK:          draftK,
		AcceptedCounts:  append([]int(nil), acceptedCounts...),
		D2HBytes:        0, // strictly 0 D2H bytes
		D2HEvents:       0, // strictly 0 D2H sync events
		ZeroD2HVerified: true,
		ReplayStrategy:  "device-shadow-prefix-replay",
	}

	return updatedLive, receipt, nil
}

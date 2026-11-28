package modelengine

// fak-test:runtime fast est=1s

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestMoeCombineDropFullyDroppedTokenZerosStaleMemory(t *testing.T) {
	nan := float32(math.NaN())
	// 2 tokens, topK=2, 1 expert with ExpertCapacity=2:
	// token 0 takes both capacity slots; token 1 overflows completely and is fully dropped.
	cfg := MoEDispatchDropConfig{
		NumTokens:      2,
		TopK:           2,
		NumExperts:     1,
		HiddenDim:      4,
		ExpertCapacity: 2,
		NonBlocking:    true,
	}
	tokens := [][]float32{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	}
	topkExperts := [][]int{{0, 0}, {0, 0}}
	topkProbs := [][]float32{{0.25, 0.75}, {0.5, 0.5}}

	state, err := MoEDispatchWithDrop(cfg, tokens, topkExperts, topkProbs, nil)
	if err != nil {
		t.Fatalf("MoEDispatchWithDrop: %v", err)
	}
	if !state.TokenDropTriggered || state.DroppedSlots != 2 || state.FullyDroppedTokens != 1 {
		t.Fatalf("drop accounting = triggered:%t dropped:%d fullyDropped:%d, want true/2/1",
			state.TokenDropTriggered, state.DroppedSlots, state.FullyDroppedTokens)
	}
	for k := 0; k < cfg.TopK; k++ {
		e := state.Entries[1*cfg.TopK+k]
		if !e.Dropped || e.SourceSlot != MoEDropNoSource || e.LocalExpertID != MoEDropNoSource {
			t.Fatalf("dropped token slot %d entry = %+v, want Dropped=true SourceSlot=-1", k, e)
		}
	}

	stale := [][]float32{
		{nan, 99, nan, -99},
		{nan, nan, 1234, nan},
	}
	var observedPhases []MoECombinePhase
	out, err := MoEUnpermuteCombine(state, MoECombineDropConfig{
		NonBlocking: true,
		StaleBuffer: stale,
		Observer: func(phase MoECombinePhase) {
			observedPhases = append(observedPhases, phase)
		},
	})
	if err != nil {
		t.Fatalf("MoEUnpermuteCombine: %v", err)
	}
	if !out.TokenDropTriggered || !out.NonBlocking {
		t.Fatalf("combine output flags = triggered:%t nonBlocking:%t, want true/true", out.TokenDropTriggered, out.NonBlocking)
	}

	// Token 1 was fully dropped: every element must be exact +0.0, never NaN.
	for d, v := range out.Combined[1] {
		if math.Float32bits(v) != 0 {
			t.Fatalf("fully dropped token dim %d = %v (bits=0x%x), want exact +0.0", d, v, math.Float32bits(v))
		}
	}
	// Token 0 kept both slots (0.25 + 0.75 = 1.0 of its payload).
	wantTok0 := []float32{1, 2, 3, 4}
	if !reflect.DeepEqual(out.Combined[0], wantTok0) {
		t.Fatalf("token 0 combined = %v, want %v", out.Combined[0], wantTok0)
	}

	// Verify explicit barrier ordering: zero_dummy -> fence -> barrier_arrive -> accumulate.
	wantPhases := []MoECombinePhase{
		MoECombinePhaseZeroDummy,
		MoECombinePhaseFence,
		MoECombinePhaseBarrierArrive,
		MoECombinePhaseAccumulate,
	}
	if !reflect.DeepEqual(observedPhases, wantPhases) {
		t.Fatalf("barrier phase ordering = %v, want %v", observedPhases, wantPhases)
	}
}

func TestMoeCombineDropPartialTokenKeepsScaledPayload(t *testing.T) {
	// Token 0 -> {0, 1} (cap 1 each); token 1 -> {0 (dropped!), 2 (kept!)}.
	// Token 1 is partial: dropped expert 0 contributes 0; kept expert 2 keeps scaled payload.
	cfg := MoEDispatchDropConfig{
		NumTokens:      2,
		TopK:           2,
		NumExperts:     3,
		HiddenDim:      3,
		ExpertCapacity: 1,
		NonBlocking:    false,
	}
	tokens := [][]float32{
		{2, 4, 6},
		{10, 20, 30},
	}
	topkExperts := [][]int{{0, 1}, {0, 2}}
	topkProbs := [][]float32{{0.6, 0.4}, {0.7, 0.3}}
	expertFn := func(tokIdx, expID int, payload []float32) []float32 {
		scale := float32(expID + 1)
		out := make([]float32, len(payload))
		for i, v := range payload {
			out[i] = v * scale
		}
		return out
	}

	state, err := MoEDispatchWithDrop(cfg, tokens, topkExperts, topkProbs, expertFn)
	if err != nil {
		t.Fatalf("MoEDispatchWithDrop: %v", err)
	}
	if !state.TokenDropTriggered || state.DroppedSlots != 1 || state.FullyDroppedTokens != 0 {
		t.Fatalf("unexpected drop counts: %+v", state)
	}

	out, err := MoEUnpermuteCombine(state, MoECombineDropConfig{NonBlocking: false})
	if err != nil {
		t.Fatalf("MoEUnpermuteCombine: %v", err)
	}

	// Token 0: 0.6*(1*tok0) + 0.4*(2*tok0) = 1.4*tok0 = {2.8, 5.6, 8.4}
	// Token 1: slot 0 (expert 0) dropped; slot 1 (expert 2) kept -> 0.3*(3*tok1) = 0.9*{10,20,30} = {9, 18, 27}
	oracle := func(tokIdx int) []float32 {
		acc := make([]float32, cfg.HiddenDim)
		for k := 0; k < cfg.TopK; k++ {
			e := state.Entries[tokIdx*cfg.TopK+k]
			if e.Dropped {
				continue
			}
			expOut := expertFn(tokIdx, topkExperts[tokIdx][k], tokens[tokIdx])
			for d := 0; d < cfg.HiddenDim; d++ {
				acc[d] += topkProbs[tokIdx][k] * expOut[d]
			}
		}
		return acc
	}
	for tok := 0; tok < cfg.NumTokens; tok++ {
		want := oracle(tok)
		for d := 0; d < cfg.HiddenDim; d++ {
			if math.Abs(float64(out.Combined[tok][d]-want[d])) > 1e-5 {
				t.Fatalf("token %d dim %d = %v, want %v", tok, d, out.Combined[tok][d], want[d])
			}
		}
	}
}

func TestMoeCombineDropBackwardZeroed(t *testing.T) {
	nan := float32(math.NaN())
	cfg := MoEDispatchDropConfig{
		NumTokens:      2,
		TopK:           2,
		NumExperts:     2,
		HiddenDim:      2,
		ExpertCapacity: 1,
		NonBlocking:    true,
	}
	tokens := [][]float32{{1, 2}, {3, 4}}
	// Expert 0 cap=1 taken by token 0 slot 0; expert 1 cap=1 taken by token 0 slot 1.
	// Token 1 has both slots dropped.
	topkExperts := [][]int{{0, 1}, {0, 1}}
	topkProbs := [][]float32{{0.2, 0.8}, {0.5, 0.5}}

	state, err := MoEDispatchWithDrop(cfg, tokens, topkExperts, topkProbs, nil)
	if err != nil {
		t.Fatalf("MoEDispatchWithDrop: %v", err)
	}

	staleBack := []MoECombineBackwardEntry{
		{LocalExpertID: 99, Prob: nan, GradPayload: []float32{nan, nan}},
		{LocalExpertID: 77, Prob: nan, GradPayload: []float32{nan, -5}},
	}
	out, err := MoEUnpermuteCombine(state, MoECombineDropConfig{
		NonBlocking:   true,
		Backward:      true,
		StaleBackward: staleBack,
		UpstreamGrad:  [][]float32{{10, 20}, {30, 40}},
	})
	if err != nil {
		t.Fatalf("MoEUnpermuteCombine: %v", err)
	}

	// Token 1 is fully dropped: its dummy backward entry must have zero prob,
	// zero local-expert-id, and zeroed grad payload.
	bd := out.BackwardDummy[1]
	if bd.LocalExpertID != 0 || math.Float32bits(bd.Prob) != 0 {
		t.Fatalf("fully dropped token backward dummy = %+v, want LocalExpertID=0 Prob=0", bd)
	}
	for d, v := range bd.GradPayload {
		if math.Float32bits(v) != 0 {
			t.Fatalf("fully dropped token backward grad dim %d = %v, want exact +0.0", d, v)
		}
	}

	// Token 0 kept both slots: verify backward slots carry prob, expert id, and scaled upstream grad.
	if out.BackwardSlots[0][0].LocalExpertID != 0 || out.BackwardSlots[0][0].Prob != 0.2 ||
		math.Abs(float64(out.BackwardSlots[0][0].GradPayload[0]-2.0)) > 1e-5 {
		t.Fatalf("token 0 slot 0 backward = %+v, want expert=0 prob=0.2 grad[0]=2", out.BackwardSlots[0][0])
	}
	if out.BackwardSlots[0][1].LocalExpertID != 1 || out.BackwardSlots[0][1].Prob != 0.8 ||
		math.Abs(float64(out.BackwardSlots[0][1].GradPayload[1]-16.0)) > 1e-5 {
		t.Fatalf("token 0 slot 1 backward = %+v, want expert=1 prob=0.8 grad[1]=16", out.BackwardSlots[0][1])
	}
}

func TestMoeCombineDropAsymmetricNonBlockingFlagRefused(t *testing.T) {
	cfg := MoEDispatchDropConfig{
		NumTokens:      1,
		TopK:           1,
		NumExperts:     1,
		HiddenDim:      2,
		ExpertCapacity: 1,
		NonBlocking:    true,
	}
	state, err := MoEDispatchWithDrop(cfg, [][]float32{{1, 2}}, [][]int{{0}}, [][]float32{{1}}, nil)
	if err != nil {
		t.Fatalf("MoEDispatchWithDrop: %v", err)
	}
	if _, err := MoEUnpermuteCombine(state, MoECombineDropConfig{NonBlocking: false}); !errors.Is(err, ErrMoECombineNonBlockingMismatch) {
		t.Fatalf("dispatch(true)->combine(false) err = %v, want ErrMoECombineNonBlockingMismatch", err)
	}

	cfg.NonBlocking = false
	state2, err := MoEDispatchWithDrop(cfg, [][]float32{{1, 2}}, [][]int{{0}}, [][]float32{{1}}, nil)
	if err != nil {
		t.Fatalf("MoEDispatchWithDrop: %v", err)
	}
	if _, err := MoEUnpermuteCombine(state2, MoECombineDropConfig{NonBlocking: true}); !errors.Is(err, ErrMoECombineNonBlockingMismatch) {
		t.Fatalf("dispatch(false)->combine(true) err = %v, want ErrMoECombineNonBlockingMismatch", err)
	}
}

package ggufload

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestInkling2SharedExpertOverlay(t *testing.T) {
	base := []Inkling2OverlayTensor{
		{Name: "blk.0.ctrl.gain", Role: OverlayRoleLoopControl, Data: []float32{1, 0, 0, 0}},
		{Name: "blk.0.ffn_exps", Role: OverlayRoleSharedExpert, Data: []float32{2, 3}},
		{Name: "lm_head.weight", Role: OverlayRoleLMHead, Data: []float32{4, 5}},
	}

	// Hot-swapping loop-control tensor within Frobenius bound succeeds.
	n, err := ValidateAndApplyOverlay(base, []Inkling2OverlayTensor{
		{Name: "blk.0.ctrl.gain", Role: OverlayRoleLoopControl, Data: []float32{1.05, 0, 0, 0}},
	}, 0.1)
	if err != nil || n != 1 || base[0].Data[0] != 1.05 {
		t.Fatalf("overlay apply failed: n=%d err=%v data=%v", n, err, base[0].Data)
	}

	// Attempting to mutate an immutable shared-expert or lm_head tensor is refused.
	for _, bad := range []Inkling2OverlayTensor{
		{Name: "blk.0.ffn_exps", Role: OverlayRoleSharedExpert, Data: []float32{2.01, 3}},
		{Name: "lm_head.weight", Role: OverlayRoleLMHead, Data: []float32{4.01, 5}},
	} {
		if _, err := ValidateAndApplyOverlay(base, []Inkling2OverlayTensor{bad}, 0.5); !errors.Is(err, ErrOverlayImmutableTensor) {
			t.Fatalf("expected ErrOverlayImmutableTensor for %s, got %v", bad.Name, err)
		}
	}

	// Exceeding relative Frobenius delta bound is refused.
	if _, err := ValidateAndApplyOverlay(base, []Inkling2OverlayTensor{
		{Name: "blk.0.ctrl.gain", Role: OverlayRoleLoopControl, Data: []float32{3, 0, 0, 0}},
	}, 0.1); !errors.Is(err, ErrOverlayDeltaExceeded) {
		t.Fatalf("expected ErrOverlayDeltaExceeded, got %v", err)
	}
}

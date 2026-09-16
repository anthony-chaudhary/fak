package model

import (
	"testing"
)

// TestMetalMTPQuantTargetAdmitsResidentDrafter proves the resident Metal
// raw-hidden capture path is admitted for a quantized Qwen3.8 target: a real
// (non-injected) MTP draft session constructs on a target the stale f32-only
// gate used to refuse. The Metal coordinator's ensureDrafterLocked then builds
// an owned drafter instead of silently falling back to serial decode.
//
// This is the Metal-parity counterpart to the gate at
// validateQwen35MTPDepthNTarget, which predated metal_prefill_hybrid raw-hidden
// capture.
func TestMetalMTPQuantTargetAdmitsResidentDrafter(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)

	s := m.NewSession()
	// Metal-resident target: Backend stays nil, Metal flags carry residency.
	s.Metal, s.MetalQ4K, s.Q4K = true, true, true
	if s.Backend != nil {
		t.Fatalf("Metal target unexpectedly has a compute.Backend: %s", s.Backend.Name())
	}
	if !qwen35MTPMetalTargetCapturesRawHidden(s) {
		t.Fatal("expected qwen35MTPMetalTargetCapturesRawHidden to admit the Metal Q4_K target")
	}

	// The gate admits; the draft session constructs without injection.
	draft, err := NewQwen35MTPDraftSession(s, 2)
	if err != nil {
		t.Fatalf("NewQwen35MTPDraftSession refused an admitted Metal quant target: %v", err)
	}
	defer draft.Close()

	coord, err := NewMetalMTPCoordinator(s, DefaultMetalMTPConfig())
	if err != nil {
		t.Fatalf("NewMetalMTPCoordinator: %v", err)
	}
	defer coord.Close()
	if coord.drafter == nil {
		t.Fatal("expected coordinator to own a resident drafter on the admitted Metal quant target")
	}
	if coord.draftSes == nil {
		t.Fatal("expected coordinator to own a draft session on the admitted Metal quant target")
	}
}

// TestQwen35MTPMetalCapturePredicateRejectsNonMetal keeps the admission narrow.
func TestQwen35MTPMetalCapturePredicateRejectsNonMetal(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	s := m.NewSession()
	if qwen35MTPMetalTargetCapturesRawHidden(s) {
		t.Fatal("expected a non-Metal session to be refused by the Metal capture predicate")
	}
}

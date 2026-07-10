package dispatchtick

import "testing"

// TestEvaluatePreflightPendingContractionFloor is the #4038 witness: while a
// scale-down is in flight toward a target T, EvaluatePreflight caps admits at T so no
// worker is placed onto capacity being reclaimed -- and with no pending contraction the
// result is byte-identical to before the term existed.
func TestEvaluatePreflightPendingContractionFloor(t *testing.T) {
	// A fleet whose every existing cap sits at 8 (config 10, lease 8, seat 8).
	base := PreflightInput{
		Workspace:     "ws",
		MaxWorkers:    10,
		Host:          HostCheck{Safe: true},
		Account:       AccountCheck{Available: true},
		Kernel:        KernelCheck{Alive: IntPtr(2), Target: IntPtr(8)},
		Seat:          SeatCheck{Total: IntPtr(8)},
		OSWorkerProcs: 2,
	}

	// No contraction: capacity is the existing min (8), and the new term is inert --
	// ContractionCap nil, Limiting names a real lowering cap. This is the byte-identical
	// baseline (the >0 gate means a zero target changes nothing).
	baseline := EvaluatePreflight(base)
	if baseline.Cap != 8 {
		t.Fatalf("baseline cap = %d, want 8", baseline.Cap)
	}
	if baseline.CapTerms.ContractionCap != nil {
		t.Errorf("baseline ContractionCap = %v, want nil (no contraction pending)", baseline.CapTerms.ContractionCap)
	}
	if baseline.CapTerms.Limiting == "contraction" {
		t.Errorf("baseline Limiting = %q, want a real lowering cap", baseline.CapTerms.Limiting)
	}

	// Pending contraction to target 3: capacity floored to 3 even though config/lease/seat
	// all sit at 8+, and the operator-facing limiter names the drain.
	shrink := base
	shrink.ContractionTarget = 3
	got := EvaluatePreflight(shrink)
	if got.Cap != 3 {
		t.Errorf("with contraction target 3, cap = %d, want 3", got.Cap)
	}
	if got.CapTerms.Limiting != "contraction" {
		t.Errorf("Limiting = %q, want \"contraction\"", got.CapTerms.Limiting)
	}
	if got.CapTerms.ContractionCap == nil || *got.CapTerms.ContractionCap != 3 {
		t.Errorf("ContractionCap = %v, want 3", got.CapTerms.ContractionCap)
	}
	// Headroom now reflects the post-contraction target, not the capacity being reclaimed.
	if got.Headroom != 3-got.Live {
		t.Errorf("headroom = %d, want %d (target minus live)", got.Headroom, 3-got.Live)
	}

	// The contraction floor also bounds the predictive pre-warm: a forecast floor of 9
	// cannot pull capacity back above a pending drain to 3 (else the pre-warm would admit
	// onto reclaimed capacity -- the exact race being closed).
	warm := shrink
	warm.WorkerFloor = 9
	gotWarm := EvaluatePreflight(warm)
	if gotWarm.Cap != 3 {
		t.Errorf("contraction+floor cap = %d, want 3 (contraction bounds the pre-warm)", gotWarm.Cap)
	}
	if gotWarm.CapTerms.Limiting != "contraction" {
		t.Errorf("contraction+floor Limiting = %q, want \"contraction\"", gotWarm.CapTerms.Limiting)
	}

	// A contraction target ABOVE the tightest cap raises nothing (pure lowering term):
	// capacity stays at the real min and the drain is not the limiter.
	loose := base
	loose.ContractionTarget = 100
	gotLoose := EvaluatePreflight(loose)
	if gotLoose.Cap != 8 {
		t.Errorf("loose contraction (target 100) cap = %d, want 8 (no raise)", gotLoose.Cap)
	}
	if gotLoose.CapTerms.Limiting == "contraction" {
		t.Errorf("loose Limiting = %q, want a real lowering cap (contraction not binding)", gotLoose.CapTerms.Limiting)
	}

	// While draining, once live has reached the target the fleet refuses new admits
	// (headroom <= 0 -> REFUSE_AT_CAP) instead of granting onto reclaimed capacity.
	atTarget := base
	atTarget.ContractionTarget = 2
	atTarget.OSWorkerProcs = 2 // live = 2 == target
	gotAt := EvaluatePreflight(atTarget)
	if gotAt.Cap != 2 || gotAt.Headroom > 0 {
		t.Errorf("at-target: cap=%d headroom=%d, want cap 2 and headroom<=0", gotAt.Cap, gotAt.Headroom)
	}
	if gotAt.OK {
		t.Errorf("at-target verdict OK=%v, want a refusal (no admit onto reclaimed capacity)", gotAt.OK)
	}
}

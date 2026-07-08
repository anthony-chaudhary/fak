package dispatchtick

import "testing"

// TestEvaluateFocusAdmissionOverCapNewObjective: at/over the WIP cap, a NEW-objective
// candidate earns the advisory with the closed FOCUS_WIP_SATURATED token (WARN default:
// advised but not held).
func TestEvaluateFocusAdmissionOverCapNewObjective(t *testing.T) {
	got := EvaluateFocusAdmission(FocusCheck{Active: 4, WIPCap: 3, Present: true, NewObjective: true})
	if !got.Saturated {
		t.Fatalf("saturated = false, want true (4 active >= cap 3)")
	}
	if !got.Advise {
		t.Fatalf("advise = false, want true on a new-objective candidate over cap")
	}
	if got.Hold {
		t.Fatalf("held = true, want false (default posture is WARN)")
	}
	if got.Token != FocusWIPSaturated {
		t.Fatalf("token = %q, want %q", got.Token, FocusWIPSaturated)
	}
	if got.Posture != FocusPostureWarn {
		t.Fatalf("posture = %q, want %q", got.Posture, FocusPostureWarn)
	}
	if got.ExcessWIP != 1 {
		t.Fatalf("excess_wip = %d, want 1", got.ExcessWIP)
	}
	if got.Reason == "" {
		t.Fatalf("reason empty, want a FOCUS_WIP_SATURATED citation")
	}
}

// TestEvaluateFocusAdmissionAtCapExactlyAdvises: the threshold is Active >= WIPCap, so
// exactly at the cap already advises (ExcessWIP 0 -- at, not over).
func TestEvaluateFocusAdmissionAtCapExactlyAdvises(t *testing.T) {
	got := EvaluateFocusAdmission(FocusCheck{Active: 3, WIPCap: 3, Present: true, NewObjective: true})
	if !got.Saturated || !got.Advise {
		t.Fatalf("at cap: saturated=%v advise=%v, want both true", got.Saturated, got.Advise)
	}
	if got.ExcessWIP != 0 {
		t.Fatalf("excess_wip = %d, want 0 (exactly at cap)", got.ExcessWIP)
	}
}

// TestEvaluateFocusAdmissionUnderCapByteIdentical: below the WIP cap there is NO advisory
// and NO token -- the tick payload is byte-identical to today (the caller attaches the
// focus block only when Advise is true).
func TestEvaluateFocusAdmissionUnderCapByteIdentical(t *testing.T) {
	got := EvaluateFocusAdmission(FocusCheck{Active: 2, WIPCap: 3, Present: true, NewObjective: true})
	if got.Saturated || got.Advise || got.Hold {
		t.Fatalf("under cap: saturated=%v advise=%v held=%v, want all false", got.Saturated, got.Advise, got.Hold)
	}
	if got.Token != "" || got.Reason != "" {
		t.Fatalf("under cap emitted token=%q reason=%q, want empty (byte-identical)", got.Token, got.Reason)
	}
}

// TestEvaluateFocusAdmissionContinuationNeverBlocked: a continuation (NewObjective false)
// is NEVER advised or held, even at/over the cap and even under the HOLD posture -- the
// term only bounds fan-out, never convergence work.
func TestEvaluateFocusAdmissionContinuationNeverBlocked(t *testing.T) {
	for _, hold := range []bool{false, true} {
		got := EvaluateFocusAdmission(FocusCheck{Active: 9, WIPCap: 3, Present: true, NewObjective: false, Hold: hold})
		if !got.Saturated {
			t.Fatalf("hold=%v: saturated=false, want true (breadth is over cap regardless)", hold)
		}
		if got.Advise || got.Hold {
			t.Fatalf("hold=%v: continuation advise=%v held=%v, want both false (never blocked)", hold, got.Advise, got.Hold)
		}
		if got.Token != "" {
			t.Fatalf("hold=%v: continuation emitted token %q, want empty", hold, got.Token)
		}
	}
}

// TestEvaluateFocusAdmissionHoldPosture: the HOLD posture turns the advisory into a hold
// on a new-objective spawn over cap.
func TestEvaluateFocusAdmissionHoldPosture(t *testing.T) {
	got := EvaluateFocusAdmission(FocusCheck{Active: 5, WIPCap: 3, Present: true, NewObjective: true, Hold: true})
	if !got.Advise || !got.Hold {
		t.Fatalf("hold posture: advise=%v held=%v, want both true", got.Advise, got.Hold)
	}
	if got.Posture != FocusPostureHold {
		t.Fatalf("posture = %q, want %q", got.Posture, FocusPostureHold)
	}
	if got.Token != FocusWIPSaturated {
		t.Fatalf("token = %q, want %q", got.Token, FocusWIPSaturated)
	}
}

// TestEvaluateFocusAdmissionNoLedgerAbstains: with no ledger signal (Present false) the
// term abstains -- an empty/unmeasured fleet is never slandered as over-cap.
func TestEvaluateFocusAdmissionNoLedgerAbstains(t *testing.T) {
	got := EvaluateFocusAdmission(FocusCheck{Active: 9, WIPCap: 3, Present: false, NewObjective: true})
	if got.Saturated || got.Advise || got.Hold {
		t.Fatalf("no ledger: saturated=%v advise=%v held=%v, want all false", got.Saturated, got.Advise, got.Hold)
	}
}

// TestEvaluateFocusAdmissionDisabledCap: a non-positive WIP cap disables the term (no-op).
func TestEvaluateFocusAdmissionDisabledCap(t *testing.T) {
	got := EvaluateFocusAdmission(FocusCheck{Active: 9, WIPCap: 0, Present: true, NewObjective: true})
	if got.Saturated || got.Advise {
		t.Fatalf("disabled cap: saturated=%v advise=%v, want both false", got.Saturated, got.Advise)
	}
}

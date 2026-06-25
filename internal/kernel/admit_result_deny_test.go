package kernel

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestAdmitResultHonorsResultSideDeny is the #691 deliverable: it proves the
// result-side VerdictDeny branch admitResult grew (kernel.go, the `case
// abi.VerdictDeny` arm) is honored end-to-end through the in-process Reap fold —
// not just when AdmitResult is called directly.
//
// It complements TestAdmitResultDenyHardRefuses (which exercises the exported
// AdmitResult in isolation) and TestScopedResultAdmitterRoutesByTool (which
// proves the Syscall→Reap fold for a QUARANTINE). Before #691 a result-side
// Deny fell through to the default admitted++ branch; no test proved a hard
// deny survives the real dispatch→Reap→admitResult path. This closes that gap:
// a gate may now hard-deny a produced result and the kernel pages the payload
// out, stamps the deny-as-value, and tallies a ResultDenies — never an Admitted.
func TestAdmitResultHonorsResultSideDeny(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}) // allow so the call dispatches
	abi.RegisterEngine("e", &countEngine{})
	abi.RegisterResultAdmitter(0, verdictAdmitter{abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "result-side-deny",
	}})
	k := New("e")

	// Full in-process path: Submit allows the call, the engine produces an OK
	// result, then Reap folds admitResult and the result-side Deny fires.
	r, v := k.Syscall(context.Background(), call("read_x", "{}"))

	// Syscall returns the PRE-CALL verdict (Allow); the result-side deny is
	// reflected in the produced result, not the returned verdict.
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("pre-call verdict = %v, want Allow (the deny is result-side)", v.Kind)
	}
	if r.Status != abi.StatusError || r.Meta["admit"] != "denied" {
		t.Fatalf("result not denied through the Reap fold: status=%v meta=%+v", r.Status, r.Meta)
	}
	if r.Meta["reason"] != "TRUST_VIOLATION" {
		t.Fatalf("denied result reason = %q, want TRUST_VIOLATION", r.Meta["reason"])
	}
	// The payload must be paged out — a denied result never carries its bytes.
	if len(r.Payload.Inline) != 0 || r.Payload.Kind != 0 {
		t.Fatalf("denied result must not carry the original payload, got %+v", r.Payload)
	}
	// A hard deny tallies a ResultDenies and never an Admitted (the branch that
	// existed before #691 would have counted this as Admitted).
	if c := k.Counters(); c.ResultDenies != 1 || c.Admitted != 0 {
		t.Fatalf("counters = %+v, want ResultDenies=1 and Admitted=0", c)
	}
}

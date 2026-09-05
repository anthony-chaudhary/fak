package abi

import (
	"context"
	"testing"
)

func TestPolicyContextRoundTrip(t *testing.T) {
	pc := PolicyContext{
		Posture:   PostureDefaultOpen,
		Profile:   "dev",
		SafeSinks: map[string]bool{"send_input": true},
	}
	ctx := ContextWithPolicy(context.Background(), pc)
	got, ok := PolicyFromContext(ctx)
	if !ok {
		t.Fatal("PolicyFromContext returned ok=false")
	}
	if got.Posture != PostureDefaultOpen || got.Profile != "dev" || !got.SafeSinks["send_input"] {
		t.Fatalf("unexpected PolicyContext: %+v", got)
	}

	if _, ok := PolicyFromContext(context.Background()); ok {
		t.Fatal("PolicyFromContext returned ok=true for bare context")
	}
	if _, ok := PolicyFromContext(nil); ok {
		t.Fatal("PolicyFromContext returned ok=true for nil context")
	}

	if PostureFailClosed.String() != "fail_closed" {
		t.Fatalf("PostureFailClosed.String() = %q, want fail_closed", PostureFailClosed.String())
	}
	if PostureAdmitAndLog.String() != "admit_and_log" {
		t.Fatalf("PostureAdmitAndLog.String() = %q, want admit_and_log", PostureAdmitAndLog.String())
	}
	if PostureDefaultOpen.String() != "default_open" {
		t.Fatalf("PostureDefaultOpen.String() = %q, want default_open", PostureDefaultOpen.String())
	}
	if Posture(99).String() != "unknown" {
		t.Fatalf("Posture(99).String() = %q, want unknown", Posture(99).String())
	}
}

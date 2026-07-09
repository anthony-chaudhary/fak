package stopped

import "testing"

// TestTriageSelfcheck runs the packaged no-I/O proof.
func TestTriageSelfcheck(t *testing.T) {
	if err := TriageSelfcheck(); err != nil {
		t.Fatalf("TriageSelfcheck: %v", err)
	}
}

// TestPartitionDefer proves the DEFER bucket splits by whether the wall clears on
// its own: only the auth wall stays on the human side; a throttle, a session
// limit, and a structural replay-safety block are all fleet-waits.
func TestPartitionDefer(t *testing.T) {
	d := Decisions{Defer: []Row{
		{Disp: DispStoppedAuth, BlockedBy: "account auth/subscription disabled"},
		{Disp: DispStoppedMidtool, BlockedBy: "account throttled, resets 2026-07-09T12:00:00Z"},
		{Disp: DispStoppedLimit, BlockedBy: "session limit, resets 2026-07-09T12:00:00Z"},
		{Disp: DispStoppedInterrupt, BlockedBy: "replayed transcript ~900000 tokens exceeds target context window 200000 — resume would overflow"},
	}}

	need, wait := PartitionDefer(d)
	if len(need) != 1 || need[0].Disp != DispStoppedAuth {
		t.Fatalf("only the auth wall should need a person, got %+v", need)
	}
	if len(wait) != 3 {
		t.Fatalf("throttle + limit + replay-safety should all be fleet-waits, got %d: %+v", len(wait), wait)
	}
	// PartitionDefer must not mutate the input bucket.
	if len(d.Defer) != 4 {
		t.Fatalf("PartitionDefer mutated the input Defer bucket, now %d rows", len(d.Defer))
	}
}

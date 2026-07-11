package resume

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// TestObjectivePinCarry is the #4121 done-condition: a real pin carried across a
// simulated OS relaunch (State.ObjectivePin -> DriveCarryRow -> reconstructed AFTER
// pin) reconciles as ObjectivePreserved (no refusal), whereas the SAME round-trip
// WITHOUT the carry reconciles as ObjectiveDropped — the silent-omission failure
// #1583 forbids, re-opened by the relaunch and closed here by the carry.
func TestObjectivePinCarry(t *testing.T) {
	// The standing objective as it stood BEFORE the relaunch (the live State.ObjectivePin).
	before := ctxplan.NewObjectivePin("obj-ship-carry", "carry the objective pin across a relaunch", 4)

	// WITH the carry: the write side projects the pin onto the budget row, the row
	// survives the relaunch on resume_drivecarry.jsonl, and the resumed child
	// reconstructs the AFTER pin from it for its first reset's reconcile.
	carried := (DriveCarryRow{Session: "carry-session", TurnsLeft: 7}).WithObjectivePin(before)
	if carried.ObjectivePinID != before.PinID || carried.ObjectiveDigest != before.Digest {
		t.Fatalf("carry did not record the pin triple: %+v", carried)
	}
	after := carried.ObjectivePin()
	dec := ctxplan.ReconcileObjective(before, after)
	if dec.Outcome != ctxplan.ObjectivePreserved {
		t.Fatalf("with carry: outcome=%s, want preserved (%s)", dec.Outcome, dec.Reason)
	}
	if dec.Outcome.Refusal() {
		t.Fatalf("with carry: preserved must not be a refusal, got %s", dec.Outcome)
	}

	// WITHOUT the carry: a relaunch that comes up with no carry row reconstructs the
	// ZERO pin, so the same prior objective reconciles as Dropped (a refusal).
	bare := DriveCarryRow{Session: "carry-session", TurnsLeft: 7}
	if !bare.ObjectivePin().IsZero() {
		t.Fatalf("bare row must carry no objective, got %+v", bare.ObjectivePin())
	}
	dropped := ctxplan.ReconcileObjective(before, bare.ObjectivePin())
	if dropped.Outcome != ctxplan.ObjectiveDropped {
		t.Fatalf("without carry: outcome=%s, want dropped (%s)", dropped.Outcome, dropped.Reason)
	}
	if !dropped.Outcome.Refusal() {
		t.Fatalf("without carry: dropped must be a refusal, got %s", dropped.Outcome)
	}
}

// TestDriveCarryObjectiveFoldRoundTrip guards that the objective triple survives the
// JSON-shaped fold the watchdog re-seed reads back (latest row per session wins), so
// the carried pin the fold returns is the one the relaunch re-pins from.
func TestDriveCarryObjectiveFoldRoundTrip(t *testing.T) {
	pin := ctxplan.NewObjectivePin("obj-a", "finish the migration", 2)
	rows := []DriveCarryRow{
		{Session: "sid", TurnsLeft: 9},
		(DriveCarryRow{Session: "sid", TurnsLeft: 5}).WithObjectivePin(pin),
	}
	got := FoldDriveCarryRows(rows)["sid"]
	if got.TurnsLeft != 5 {
		t.Fatalf("fold budget=%d, want latest 5", got.TurnsLeft)
	}
	if rt := got.ObjectivePin(); ctxplan.ReconcileObjective(pin, rt).Outcome != ctxplan.ObjectivePreserved {
		t.Fatalf("folded objective did not round-trip: %+v", rt)
	}
	// A zero pin carries nothing — WithObjectivePin is a no-op on the zero pin.
	if noop := (DriveCarryRow{Session: "sid"}).WithObjectivePin(ctxplan.ObjectivePin{}); noop.ObjectivePinID != "" {
		t.Fatalf("zero pin must carry nothing, got %+v", noop)
	}
}

func TestDriveCarryRowsFoldLastWins(t *testing.T) {
	rows := []DriveCarryRow{
		{Session: "sid-a", TurnsLeft: 8, TokensLeft: 2000},
		{Session: "sid-b", TurnsLeft: 4, TokensLeft: 500},
		{Session: " sid-a ", TurnsLeft: 6, TokensLeft: 1500, SpendMicroCentsLeft: 9, Generation: 2},
		{Session: " ", TurnsLeft: 99},
	}
	got := FoldDriveCarryRows(rows)
	if len(got) != 2 {
		t.Fatalf("len=%d, want 2: %+v", len(got), got)
	}
	if got["sid-a"].TurnsLeft != 6 || got["sid-a"].TokensLeft != 1500 || got["sid-a"].Generation != 2 {
		t.Fatalf("sid-a=%+v, want latest row", got["sid-a"])
	}
	if got["sid-b"].TurnsLeft != 4 {
		t.Fatalf("sid-b=%+v", got["sid-b"])
	}
	if empty := FoldDriveCarryRows(nil); len(empty) != 0 {
		t.Fatalf("nil fold=%+v", empty)
	}
}

package relay

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// Rung H5 (issue #1898) done condition: a test asserts that hitting the window ceiling
// with no safe point parks with RELAY_PARKED_UNSAFE, losing no committed work. These are
// that witness (run: `go test ./internal/relay -run ParkUnsafe`). The pure-rung tests pin
// the fold (anchor tracking, tombstone rendering); the driver test proves the whole leg
// parks rather than erroring or blowing the window.

// TestCeilingParkUnsafeAnchorsLastCommit pins the fold's core promise: the resume anchor
// is the LAST committed SHA any boundary observed, boundaries with no commit never rewind
// it, and the rendered tombstone carries the closed token anchored there.
func TestCeilingParkUnsafeAnchorsLastCommit(t *testing.T) {
	var p CeilingPark
	if p.Anchor() != "" {
		t.Errorf("fresh tracker Anchor = %q, want empty", p.Anchor())
	}
	p.Observe(drvSHA1, ReasonInFlight)  // first commit observed, held in flight
	p.Observe("", ReasonTreeDirty)      // a dirty boundary with no commit must NOT rewind the anchor
	p.Observe(drvSHA2, ReasonTreeDirty) // a later commit advances the anchor
	p.Observe("", ReasonInFlight)       // and again: no commit leaves the anchor at the last real one

	if p.Anchor() != drvSHA2 {
		t.Errorf("Anchor = %q, want the last committed SHA %s (an empty boundary must not rewind it)", p.Anchor(), drvSHA2)
	}
	tomb := p.Park(9, 4)
	if tomb.Reason != ReasonParkedUnsafe {
		t.Errorf("tombstone reason = %q, want %s", tomb.Reason, ReasonParkedUnsafe)
	}
	if tomb.AtSHA != drvSHA2 {
		t.Errorf("tombstone at_sha = %q, want the resume anchor %s — no committed work stranded", tomb.AtSHA, drvSHA2)
	}
	if tomb.Note == "" {
		t.Error("tombstone note is empty; want a display-only park note")
	}
}

// TestCeilingParkUnsafeEmptyAnchorColdResume covers the vacuous case: a leg that committed
// nothing parks with an empty anchor (a cold resume). "Lose no committed work" holds because
// there was none — and the empty anchor fails closed to a re-derive on reload, never a false
// handoff.
func TestCeilingParkUnsafeEmptyAnchorColdResume(t *testing.T) {
	var p CeilingPark
	p.Observe("", ReasonInFlight)
	p.Observe("", ReasonNoNextAction)

	if p.Anchor() != "" {
		t.Errorf("Anchor = %q, want empty for a leg that committed nothing", p.Anchor())
	}
	tomb := p.Park(1, 2)
	if tomb.Reason != ReasonParkedUnsafe || tomb.AtSHA != "" {
		t.Errorf("tombstone = %+v, want %s with an empty at_sha", tomb, ReasonParkedUnsafe)
	}
}

// TestDriveLegParkUnsafeOnCeiling is the H5 done-condition witness at the driver level: a
// leg whose window pressure has crossed (so it ARMS) but which never reaches a safe point
// exhausts MaxBoundaries and PARKS — RELAY_PARKED_UNSAFE, a resumable baton anchored at the
// last committed SHA, no successor minted — instead of erroring or forcing a mid-action
// rotation.
func TestDriveLegParkUnsafeOnCeiling(t *testing.T) {
	// Every boundary is armed (context past the 0.55 mark) but unsafe (a tool call is
	// always in flight), so the leg can arm but never fire. The leg commits at boundaries
	// 0 and 2; boundary 1 observes no new commit. The last committed SHA is drvSHA2.
	dirty := TreeStatus{DirtyPaths: []string{"internal/relay/driver.go"}}
	armedInFlight := func(atSHA string) BoundaryObs {
		return BoundaryObs{
			Usage:        BudgetUsage{Context: AxisUsage{Used: 80, Cap: 100}},
			TurnInFlight: true,
			Tree:         dirty,
			NextSteps:    []string{"finish the edit"},
			AtSHA:        atSHA,
		}
	}
	script := []BoundaryObs{
		armedInFlight(drvSHA1),
		armedInFlight(""), // no new commit this boundary — must not rewind the anchor
		armedInFlight(drvSHA2),
	}

	var wire []byte
	recontinued := false
	obj := ctxplan.NewObjectivePin("pin-park", "Ship the H5 park path (#1898).", 1)
	out, err := DriveLeg(LegConfig{
		RelayID:       "RLY-20260710-1898",
		Objective:     obj,
		DoneWhen:      "issue #1898 is closed by a witnessed commit",
		LedgerRef:     ".dos/runs/relay-1898.jsonl",
		HeldRegion:    []string{"internal/relay/**"},
		TraceID:       "trace-park",
		Triggers:      ArmTriggers{SoftMark: 0.55},
		MaxBoundaries: 3,
		DoneCheck:     func(string) (bool, error) { return false, nil },
		Work:          scriptedWork(t, script),
		WriteBaton:    func(w []byte) error { wire = append([]byte(nil), w...); return nil },
		Recontinue:    func(Baton) (string, error) { recontinued = true; return "trace-should-not-happen", nil },
	})
	if err != nil {
		t.Fatalf("DriveLeg parked leg returned an error, want a clean park: %v", err)
	}

	// Parked, not rotated: the closed token, the Parked flag, and NO successor.
	if out.Reason != ReasonParkedUnsafe {
		t.Errorf("Reason = %q, want %s", out.Reason, ReasonParkedUnsafe)
	}
	if !out.Parked {
		t.Error("Parked = false, want true for a hard-ceiling park")
	}
	if recontinued || out.SuccessorTrace != "" {
		t.Errorf("park minted a successor (recontinued=%v trace=%q); a park is a stop, not a rotation", recontinued, out.SuccessorTrace)
	}
	if out.Boundaries != 3 {
		t.Errorf("Boundaries = %d, want the full MaxBoundaries window (3)", out.Boundaries)
	}
	if want := []string{ReasonInFlight, ReasonInFlight, ReasonInFlight}; !reflect.DeepEqual(out.Holds, want) {
		t.Errorf("Holds = %v, want %v (every boundary held in flight)", out.Holds, want)
	}

	// Lose no committed work: the parked state is resumable, anchored at the LAST committed
	// SHA the leg observed — a resume re-verifies it and picks up from the last good commit.
	if out.Baton.Tombstone.Reason != ReasonParkedUnsafe || out.Baton.Tombstone.AtSHA != drvSHA2 {
		t.Errorf("park tombstone = %+v, want %s at %s", out.Baton.Tombstone, ReasonParkedUnsafe, drvSHA2)
	}
	if out.Baton.ProgressCursor.StartSHA != drvSHA2 {
		t.Errorf("resume anchor start_sha = %q, want the last committed SHA %s (no committed work lost)", out.Baton.ProgressCursor.StartSHA, drvSHA2)
	}

	// Identity is carried onto the park baton exactly as a rotation would carry it, and the
	// written wire is the canonical, durable, resumable record.
	if out.Baton.RelayID != "RLY-20260710-1898" || !reflect.DeepEqual(out.Baton.Objective, obj) || out.Baton.DoneWhen != "issue #1898 is closed by a witnessed commit" {
		t.Errorf("park baton identity not carried: relay_id=%q done_when=%q", out.Baton.RelayID, out.Baton.DoneWhen)
	}
	if out.Baton.ProgressCursor.LedgerRef != ".dos/runs/relay-1898.jsonl" || !reflect.DeepEqual(out.Baton.ProgressCursor.HeldRegion, []string{"internal/relay/**"}) {
		t.Errorf("park cursor did not carry ledger/region: %+v", out.Baton.ProgressCursor)
	}
	parsed, err := Parse(wire)
	if err != nil {
		t.Fatalf("Parse(written park wire): %v", err)
	}
	if !reflect.DeepEqual(parsed, out.Baton) {
		t.Errorf("written park wire and outcome baton disagree:\n parsed=%+v\n out=%+v", parsed, out.Baton)
	}
}

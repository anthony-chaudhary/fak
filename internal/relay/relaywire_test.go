// Rung H6 (issue #1899), external witness: session.RelayRecontinueHook drops into
// LegConfig.Recontinue and one full driver rotation drives the LIVE session lineage —
// child Generation = parent+1 with ParentTrace linked, and the drained parent's
// terminal record left intact. Package relay_test on purpose: the wiring site holds
// BOTH concrete packages, exactly like a real host, while session itself keeps the
// baton opaque (no import either direction).
package relay_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/relay"
	"github.com/anthony-chaudhary/fak/internal/session"
)

const relaywireSHA = "cccccccccccccccccccccccccccccccccccccccc"

// TestRecontinueWiringDrivesSessionLineageThroughRotation mirrors the H1 driver-loop
// witness (driver_test.go) with the Recontinue hook wired to a live session.Table:
// the closing leg's session is context-drained to terminal Stopped (the state
// Table.Recontinue is documented to follow), the leg rotates at its first safe
// boundary, and the successor trace the driver reports is the re-armed child session.
func TestRecontinueWiringDrivesSessionLineageThroughRotation(t *testing.T) {
	const parentTrace = "trace-leg-0"

	// Drain leg 0's session to terminal through the session machinery itself: the
	// context overdraw moves it to Draining and mints the continuation id, and the
	// next Decide takes the drain to Stopped.
	tbl := session.NewTable()
	tbl.SetBudget(parentTrace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 100})
	drained := tbl.DebitUsage(parentTrace, session.Usage{ContextTokens: 101})
	if drained.ContinuationID == "" {
		t.Fatal("context drain minted no continuation id to recontinue onto")
	}
	if v := tbl.Decide(parentTrace); v.State.Run != session.Stopped {
		t.Fatalf("parent after drain = %+v, want Stopped", v.State)
	}
	parentBefore := tbl.Get(parentTrace)

	// The H6 wiring under test: the session-side hook, instantiated with the
	// concrete baton type, reading the closing leg's trace off the baton and reusing
	// the drain-minted continuation id as the child trace.
	fresh := session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 100}
	hook := session.RelayRecontinueHook(tbl, fresh, func(b relay.Baton) (string, string) {
		return b.ParentTrace, drained.ContinuationID
	})

	var wrote []byte
	out, err := relay.DriveLeg(relay.LegConfig{
		RelayID:       "RLY-20260716-1899",
		Objective:     ctxplan.NewObjectivePin("pin-1899", "Wire session.Recontinue into the relay driver (#1899).", 1),
		DoneWhen:      "issue #1899 is closed by a witnessed commit",
		HeldRegion:    []string{"internal/session/**"},
		TraceID:       parentTrace,
		Triggers:      relay.ArmTriggers{SoftMark: 0.55},
		MaxBoundaries: 1,
		Work: func(_ relay.Orientation, _ int) (relay.BoundaryObs, error) {
			// One boundary that both crosses the soft mark (60/100 > 0.55) and is a
			// safe point (no in-flight turn, clean tree, one next step, no
			// transcript-only facts), so the arm/fire machine fires here.
			return relay.BoundaryObs{
				Usage:     relay.BudgetUsage{Context: relay.AxisUsage{Used: 60, Cap: 100}},
				NextSteps: []string{"reload the baton on the fresh leg"},
				AtSHA:     relaywireSHA,
			}, nil
		},
		WriteBaton: func(wire []byte) error { wrote = wire; return nil },
		Recontinue: hook,
	})
	if err != nil {
		t.Fatalf("DriveLeg error = %v, want a clean rotation", err)
	}
	if out.Reason != "RELAY_ROTATED" || out.Parked {
		t.Fatalf("outcome = reason %q parked %v, want a fired RELAY_ROTATED rotation", out.Reason, out.Parked)
	}
	if len(wrote) == 0 {
		t.Fatal("rotation fired but no baton bytes were persisted before Recontinue")
	}
	if out.SuccessorTrace != drained.ContinuationID {
		t.Fatalf("successor trace = %q, want the re-armed continuation %q", out.SuccessorTrace, drained.ContinuationID)
	}

	// Assertion 1: the child's Generation is the parent's + 1 (a leg IS a generation).
	child := tbl.Get(out.SuccessorTrace)
	if child.Generation != parentBefore.Generation+1 {
		t.Fatalf("child Generation = %d, want parent+1 = %d", child.Generation, parentBefore.Generation+1)
	}
	// Assertion 2: the child's ParentTrace links back to the closing leg's trace.
	if child.ParentTrace != parentTrace {
		t.Fatalf("child ParentTrace = %q, want the closing leg %q", child.ParentTrace, parentTrace)
	}
	if child.Run != session.Running || child.Budget.ContextTokensLeft != 100 {
		t.Fatalf("child = run %v context-left %d, want Running with the fresh 100-token re-arm", child.Run, child.Budget.ContextTokensLeft)
	}
	// Assertion 3: the parent is left stopped, its terminal record intact (same
	// closed Reason the drain wrote — the rotation revives nothing in place).
	parentAfter := tbl.Get(parentTrace)
	if parentAfter.Run != session.Stopped || parentAfter.Reason != parentBefore.Reason {
		t.Fatalf("parent after rotation = run %v reason %q, want Stopped with the drain reason %q intact",
			parentAfter.Run, parentAfter.Reason, parentBefore.Reason)
	}
}

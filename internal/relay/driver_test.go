package relay

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// H1 (issue #1894) done condition: a test drives a full leg through a rotation into a
// fresh leg with the pieces wired. These are that witness (run: `go test
// ./internal/relay -run DriverLoop`). The driver adds no policy — every verdict here
// is asserted to come out of the rung that owns it (D1/D2 reload, D3 progress, E2/E3/
// E4 safe-point axes, F2 externalize gate, G2/G3 arm/fire) with all I/O behind hooks.

const (
	drvSHA1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	drvSHA2 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

// drvLedger is a hermetic D3 LedgerReader that records whether it was consulted.
type drvLedger struct {
	steps  []ProgressStep
	called *bool
}

func (l drvLedger) ReadProgress(ledgerRef string) ([]ProgressStep, error) {
	if l.called != nil {
		*l.called = true
	}
	return l.steps, nil
}

// drvIncoming builds the leg-7 baton a predecessor wrote, anchored at drvSHA1.
func drvIncoming() Baton {
	return Baton{
		Schema:      Schema,
		RelayID:     "RLY-20260708-0001",
		Leg:         7,
		ParentTrace: "trace-leg-7",
		Objective:   ctxplan.NewObjectivePin("pin-driver", "Ship the relay driver loop (#1894).", 2),
		DoneWhen:    "issue #1894 is closed by a witnessed commit",
		ProgressCursor: ProgressCursor{
			StartSHA:   drvSHA1,
			LedgerRef:  ".dos/runs/relay-1894.jsonl",
			HeldRegion: []string{"internal/relay/**"},
		},
		NextAction:    "wire the driver loop",
		Artifacts:     []Artifact{{Kind: string(ArtifactIssue), Ref: "#1894"}},
		DoNotRederive: []string{"memory:relay-driver-dead-end"},
		Tombstone:     Tombstone{Reason: "RELAY_ROTATED", AtSHA: drvSHA1},
	}
}

// scriptedWork returns a Work hook that replays obs in order and fails the test if
// the driver asks for more boundaries than the script holds.
func scriptedWork(t *testing.T, obs []BoundaryObs) func(Orientation, int) (BoundaryObs, error) {
	t.Helper()
	return func(_ Orientation, b int) (BoundaryObs, error) {
		if b >= len(obs) {
			t.Fatalf("work hook called for boundary %d beyond the %d-boundary script", b, len(obs))
		}
		return obs[b], nil
	}
}

// TestDriverLoopFullLegRotatesIntoFreshLeg is the H1 done-condition witness: one leg
// reloads and verifies its predecessor's baton, works through held boundaries, arms at
// the soft mark, fires at the first safe externalized boundary, writes the canonical
// baton, Recontinues into a fresh leg — and the fresh leg reloads that baton and ends
// the relay on its done-check.
func TestDriverLoopFullLegRotatesIntoFreshLeg(t *testing.T) {
	resolver := fakeResolver{verified: map[string]bool{drvSHA1: true, drvSHA2: true}}
	ledgerCalled := false
	ledger := drvLedger{steps: []ProgressStep{{Ref: drvSHA1, Note: "leg 7 landed"}}, called: &ledgerCalled}
	incoming := drvIncoming()

	script := []BoundaryObs{
		{ // mid-work: below the mark, mid-turn, dirty tree, competing next steps, unbacked fact
			Usage:        BudgetUsage{Context: AxisUsage{Used: 30, Cap: 100}},
			TurnInFlight: true,
			Tree:         TreeStatus{DirtyPaths: []string{"internal/relay/driver.go"}},
			NextSteps:    []string{"write the driver", "write the test"},
			Facts:        []LoadBearingFact{{Label: "driver design"}},
			AtSHA:        drvSHA1,
		},
		{ // context crosses the 0.55 soft mark -> arms, but a tool call is still in flight
			Usage:        BudgetUsage{Context: AxisUsage{Used: 60, Cap: 100}},
			TurnInFlight: true,
			Tree:         TreeStatus{DirtyPaths: []string{"internal/relay/driver.go"}},
			NextSteps:    []string{"commit the driver"},
			Facts:        []LoadBearingFact{{Label: "driver design"}},
			AtSHA:        drvSHA1,
		},
		{ // safe point: boundary, green tree, one next step, everything externalized -> fires
			Usage:         BudgetUsage{Context: AxisUsage{Used: 62, Cap: 100}},
			TurnInFlight:  false,
			Tree:          TreeStatus{},
			NextSteps:     []string{"run go test ./internal/relay -run DriverLoop"},
			Facts:         []LoadBearingFact{{Label: "driver design", Backing: Artifact{Kind: string(ArtifactCommit), Ref: drvSHA2}}},
			AtSHA:         drvSHA2,
			Artifacts:     []Artifact{{Kind: string(ArtifactCommit), Ref: drvSHA2}},
			OpenQuestions: []string{"issue:#1895 decides lease continuity"},
			DoNotRederive: []string{"memory:relay-driver-dead-end", "issue:#1893-wrong-turn"},
		},
	}

	var wire []byte
	var handedOff Baton
	out, err := DriveLeg(LegConfig{
		Incoming:      incoming,
		TraceID:       "trace-leg-8",
		Triggers:      ArmTriggers{SoftMark: 0.55},
		MaxBoundaries: 5,
		Resolver:      resolver,
		Ledger:        ledger,
		DoneCheck:     func(string) (bool, error) { return false, nil },
		Work:          scriptedWork(t, script),
		WriteBaton:    func(w []byte) error { wire = append([]byte(nil), w...); return nil },
		Recontinue:    func(b Baton) (string, error) { handedOff = b; return "trace-leg-9", nil },
	})
	if err != nil {
		t.Fatalf("DriveLeg: %v", err)
	}

	// The leg rotated at the third boundary into a fresh leg.
	if out.Reason != "RELAY_ROTATED" {
		t.Errorf("Reason = %q, want RELAY_ROTATED", out.Reason)
	}
	if out.Boundaries != 3 {
		t.Errorf("Boundaries = %d, want 3", out.Boundaries)
	}
	if out.SuccessorTrace != "trace-leg-9" {
		t.Errorf("SuccessorTrace = %q, want trace-leg-9", out.SuccessorTrace)
	}
	if want := []string{ReasonInFlight, ReasonInFlight}; !reflect.DeepEqual(out.Holds, want) {
		t.Errorf("Holds = %v, want %v", out.Holds, want)
	}

	// Reload+verify ran and fed the hook verified state only.
	if out.Orientation.FirstLeg || out.Orientation.Stale.Stale {
		t.Errorf("orientation = %+v, want a fresh successor-leg orientation", out.Orientation)
	}
	if !ledgerCalled || out.Orientation.Progress.Verdict != ProgressVerified {
		t.Errorf("D3 progress read: called=%v verdict=%q, want a verified ledger read", ledgerCalled, out.Orientation.Progress.Verdict)
	}

	// The written baton is canonical wire (C2) and carries identity verbatim.
	parsed, err := Parse(wire)
	if err != nil {
		t.Fatalf("Parse(written wire): %v", err)
	}
	if !reflect.DeepEqual(parsed, out.Baton) || !reflect.DeepEqual(handedOff, out.Baton) {
		t.Errorf("written wire, Recontinue baton and outcome baton disagree:\n parsed=%+v\n handed=%+v\n out=%+v", parsed, handedOff, out.Baton)
	}
	b := out.Baton
	if b.Schema != Schema || b.RelayID != incoming.RelayID {
		t.Errorf("identity: schema=%q relay_id=%q, want carried verbatim", b.Schema, b.RelayID)
	}
	if b.Leg != 8 || b.ParentTrace != "trace-leg-8" {
		t.Errorf("lineage: leg=%d parent_trace=%q, want leg 8 under trace-leg-8", b.Leg, b.ParentTrace)
	}
	if !reflect.DeepEqual(b.Objective, incoming.Objective) || b.DoneWhen != incoming.DoneWhen {
		t.Errorf("objective/done_when not carried verbatim: %+v / %q", b.Objective, b.DoneWhen)
	}
	wantCursor := ProgressCursor{StartSHA: drvSHA2, LedgerRef: incoming.ProgressCursor.LedgerRef, HeldRegion: incoming.ProgressCursor.HeldRegion}
	if !reflect.DeepEqual(b.ProgressCursor, wantCursor) {
		t.Errorf("cursor = %+v, want %+v", b.ProgressCursor, wantCursor)
	}
	if b.NextAction != "run go test ./internal/relay -run DriverLoop" {
		t.Errorf("next_action = %q, want the E4-extracted single step", b.NextAction)
	}
	if want := []string{"memory:relay-driver-dead-end", "issue:#1893-wrong-turn"}; !reflect.DeepEqual(b.DoNotRederive, want) {
		t.Errorf("do_not_rederive = %v, want the deduped carried-forward union %v", b.DoNotRederive, want)
	}
	if b.Tombstone.Reason != "RELAY_ROTATED" || b.Tombstone.AtSHA != drvSHA2 {
		t.Errorf("tombstone = %+v, want RELAY_ROTATED at %s", b.Tombstone, drvSHA2)
	}

	// The fresh leg reloads the written baton and the relay ends on its done-check —
	// the rotation handed off something a successor can actually run from.
	freshWorked := false
	out2, err := DriveLeg(LegConfig{
		Incoming:      handedOff,
		TraceID:       "trace-leg-9",
		Triggers:      ArmTriggers{SoftMark: 0.55},
		MaxBoundaries: 1,
		Resolver:      resolver,
		DoneCheck:     func(doneWhen string) (bool, error) { return doneWhen == incoming.DoneWhen, nil },
		Work: func(Orientation, int) (BoundaryObs, error) {
			freshWorked = true
			return BoundaryObs{}, nil
		},
		WriteBaton: func([]byte) error { return nil },
		Recontinue: func(Baton) (string, error) { return "", nil },
	})
	if err != nil {
		t.Fatalf("DriveLeg (fresh leg): %v", err)
	}
	if out2.Reason != ReasonGoalDone || freshWorked || !out2.Baton.IsZero() {
		t.Errorf("fresh leg: reason=%q worked=%v baton.IsZero=%v, want RELAY_GOAL_DONE with no work and no new baton",
			out2.Reason, freshWorked, out2.Baton.IsZero())
	}
	if out2.Orientation.Stale.Stale {
		t.Errorf("fresh leg reloaded its own predecessor's baton as stale: %+v", out2.Orientation.Stale)
	}
}

// TestDriverLoopHoldsUntilExternalized pins the F2 wiring on a FIRST leg: an armed,
// otherwise-safe boundary carrying a transcript-only load-bearing fact cannot fire —
// the hold is the closed RELAY_NOT_EXTERNALIZED — and the next boundary, with the
// fact backed by a durable pointer, rotates.
func TestDriverLoopHoldsUntilExternalized(t *testing.T) {
	script := []BoundaryObs{
		{ // armed and at a safe point, but one fact is still transcript-only
			Usage:     BudgetUsage{Turns: AxisUsage{Used: 9, Cap: 10}},
			NextSteps: []string{"externalize the API-shape decision"},
			Facts:     []LoadBearingFact{{Label: "API-shape decision"}},
			AtSHA:     drvSHA1,
		},
		{ // the fact now has a durable pointer -> fires
			Usage:     BudgetUsage{Turns: AxisUsage{Used: 9, Cap: 10}},
			NextSteps: []string{"hand off to the successor"},
			Facts:     []LoadBearingFact{{Label: "API-shape decision", Backing: Artifact{Kind: string(ArtifactMemory), Ref: "api-shape"}}},
			AtSHA:     drvSHA1,
		},
	}
	out, err := DriveLeg(LegConfig{
		RelayID:       "RLY-20260708-0002",
		Objective:     ctxplan.NewObjectivePin("pin-first", "First-leg driver witness.", 1),
		DoneWhen:      "the witness test passes",
		HeldRegion:    []string{"internal/relay/**"},
		TraceID:       "trace-leg-0",
		Triggers:      ArmTriggers{SoftMark: 0.5},
		MaxBoundaries: 3,
		Work:          scriptedWork(t, script),
		WriteBaton:    func([]byte) error { return nil },
		Recontinue:    func(Baton) (string, error) { return "trace-leg-1", nil },
	})
	if err != nil {
		t.Fatalf("DriveLeg: %v", err)
	}
	if !out.Orientation.FirstLeg {
		t.Errorf("FirstLeg = false, want true for a zero incoming baton")
	}
	if want := []string{ReasonNotExternalized}; !reflect.DeepEqual(out.Holds, want) {
		t.Errorf("Holds = %v, want %v", out.Holds, want)
	}
	if out.Boundaries != 2 || out.Reason != "RELAY_ROTATED" {
		t.Errorf("boundaries=%d reason=%q, want a rotation at boundary 2", out.Boundaries, out.Reason)
	}
	if out.Baton.Leg != 0 || out.Baton.RelayID != "RLY-20260708-0002" {
		t.Errorf("first-leg baton = leg %d relay %q, want leg 0 of RLY-20260708-0002", out.Baton.Leg, out.Baton.RelayID)
	}
}

// TestDriverLoopStaleBatonRederivesNotTrusts pins the D2 routing: a diverged incoming
// cursor surfaces RELAY_BATON_STALE in the orientation, the D3 ledger read is SKIPPED
// (nothing unverified reaches the hook), and the leg still drives to a clean rotation
// anchored at fresh ground truth.
func TestDriverLoopStaleBatonRederivesNotTrusts(t *testing.T) {
	resolver := fakeResolver{verified: map[string]bool{drvSHA2: true}} // drvSHA1 has diverged
	ledgerCalled := false
	var seen Orientation
	out, err := DriveLeg(LegConfig{
		Incoming:      drvIncoming(), // anchored at drvSHA1
		TraceID:       "trace-leg-8",
		Triggers:      ArmTriggers{SoftMark: 0.5},
		MaxBoundaries: 2,
		Resolver:      resolver,
		Ledger:        drvLedger{steps: []ProgressStep{{Ref: "phantom"}}, called: &ledgerCalled},
		Work: func(o Orientation, b int) (BoundaryObs, error) {
			seen = o
			return BoundaryObs{
				Usage:     BudgetUsage{Context: AxisUsage{Used: 80, Cap: 100}},
				NextSteps: []string{"re-derive progress from the durable store"},
				AtSHA:     drvSHA2,
			}, nil
		},
		WriteBaton: func([]byte) error { return nil },
		Recontinue: func(Baton) (string, error) { return "trace-leg-9", nil },
	})
	if err != nil {
		t.Fatalf("DriveLeg: %v", err)
	}
	if !seen.Stale.Stale || seen.Stale.Reason != ReasonBatonStale || seen.Stale.Culprit != "start_sha" {
		t.Errorf("work hook orientation stale = %+v, want RELAY_BATON_STALE on start_sha", seen.Stale)
	}
	if ledgerCalled || seen.Progress.Verdict != "" {
		t.Errorf("stale baton must skip the ledger read: called=%v progress=%+v", ledgerCalled, seen.Progress)
	}
	if out.Reason != "RELAY_ROTATED" || out.Baton.ProgressCursor.StartSHA != drvSHA2 {
		t.Errorf("reason=%q start_sha=%q, want a rotation re-anchored at %s", out.Reason, out.Baton.ProgressCursor.StartSHA, drvSHA2)
	}
}

// TestDriverLoopDoneCheckFailsClosed pins both edges of the done-check: an evaluator
// error never claims done (the leg keeps working and rotates), and it is only a nil
// error with done=true that ends the relay before any work.
func TestDriverLoopDoneCheckFailsClosed(t *testing.T) {
	safe := BoundaryObs{
		Usage:     BudgetUsage{Context: AxisUsage{Used: 90, Cap: 100}},
		NextSteps: []string{"hand off"},
		AtSHA:     drvSHA1,
	}
	out, err := DriveLeg(LegConfig{
		RelayID: "RLY-20260708-0003", DoneWhen: "unreachable predicate", TraceID: "trace-leg-0",
		Triggers: ArmTriggers{SoftMark: 0.5}, MaxBoundaries: 1,
		DoneCheck:  func(string) (bool, error) { return true, errors.New("durable store unreachable") },
		Work:       scriptedWork(t, []BoundaryObs{safe}),
		WriteBaton: func([]byte) error { return nil },
		Recontinue: func(Baton) (string, error) { return "trace-leg-1", nil },
	})
	if err != nil {
		t.Fatalf("DriveLeg: %v", err)
	}
	if out.Reason != "RELAY_ROTATED" {
		t.Errorf("an erroring done-check claimed %q, want the leg to keep working and rotate", out.Reason)
	}
}

// TestDriverLoopExhaustionFailsClosed pins the driver's own bound: a leg that never
// reaches a rotation errors — naming the last closed hold — rather than spinning or
// fabricating an outcome. The park path that will absorb this is rung H5.
func TestDriverLoopExhaustionFailsClosed(t *testing.T) {
	inFlight := BoundaryObs{
		Usage:        BudgetUsage{Context: AxisUsage{Used: 90, Cap: 100}},
		TurnInFlight: true,
		NextSteps:    []string{"finish the tool call"},
		AtSHA:        drvSHA1,
	}
	rotated := false
	_, err := DriveLeg(LegConfig{
		RelayID: "RLY-20260708-0004", DoneWhen: "never", TraceID: "trace-leg-0",
		Triggers: ArmTriggers{SoftMark: 0.5}, MaxBoundaries: 3,
		Work:       func(Orientation, int) (BoundaryObs, error) { return inFlight, nil },
		WriteBaton: func([]byte) error { rotated = true; return nil },
		Recontinue: func(Baton) (string, error) { rotated = true; return "", nil },
	})
	if err == nil || !strings.Contains(err.Error(), ReasonInFlight) {
		t.Fatalf("err = %v, want an exhaustion error naming the %s hold", err, ReasonInFlight)
	}
	if rotated {
		t.Errorf("an exhausted leg must not write a baton or recontinue")
	}
}

// TestDriverLoopFiredBoundaryNeedsAnchor pins the handoff's fail-closed anchor rule:
// a rotation that fires at a boundary with no observed commit SHA is an error — the
// successor's reload (D1) could never verify such a baton, so it is never written.
func TestDriverLoopFiredBoundaryNeedsAnchor(t *testing.T) {
	wrote := false
	_, err := DriveLeg(LegConfig{
		RelayID: "RLY-20260708-0005", DoneWhen: "never", TraceID: "trace-leg-0",
		Triggers: ArmTriggers{SoftMark: 0.5}, MaxBoundaries: 1,
		Work: scriptedWork(t, []BoundaryObs{{
			Usage:     BudgetUsage{Context: AxisUsage{Used: 90, Cap: 100}},
			NextSteps: []string{"hand off"},
			// AtSHA deliberately empty
		}}),
		WriteBaton: func([]byte) error { wrote = true; return nil },
		Recontinue: func(Baton) (string, error) { return "", nil },
	})
	if err == nil || !strings.Contains(err.Error(), "no commit SHA") {
		t.Fatalf("err = %v, want a refusal to write an unverifiable baton", err)
	}
	if wrote {
		t.Errorf("a baton without a verifiable anchor must never be written")
	}
}

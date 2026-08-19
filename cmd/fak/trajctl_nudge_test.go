package main

// trajctl_nudge_test.go — issue #2765's named witness: the regime-gated re-anchor
// nudge, routed onto the #2753 control bus. Tested at two curve regimes exactly as
// the done condition states — a DEGRADING session receives a redirect op via the
// control plane; a HEALTHY one does not — plus the honest re-anchor payload (the
// objective's own statement, never the transient re-anchor prose) and the closed-
// reason capture when the bus refuses. The unpark half is asserted reachable through
// the same runTrajctl entry main.go's dispatch case invokes.

import (
	"bytes"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func nudgeObjective(id string) trajctl.Objective {
	return trajctl.Objective{
		ID:        id,
		Statement: "ship the trajctl nudge -> control-bus bridge",
		Status:    trajctl.StatusActive,
	}
}

func w3(objID string, value float64, ms int64) trajctl.ScoreRow {
	return trajctl.ScoreRow{
		ObjectiveID: objID,
		Value:       value,
		Method:      trajctl.CommitScorerMethod,
		Version:     "test",
		Witness:     trajctl.W3,
		UnixMillis:  ms,
	}
}

// degradingState folds a declining witnessed progress curve — the DRIFT regime.
func degradingState(objID string) trajctl.State {
	return trajctl.State{
		Objectives: map[string]trajctl.Objective{objID: nudgeObjective(objID)},
		Scores:     []trajctl.ScoreRow{w3(objID, 0.6, 1000), w3(objID, 0.3, 2000)},
	}
}

// healthyNudgeState folds a rising witnessed progress curve — the no-action regime.
func healthyNudgeState(objID string) trajctl.State {
	return trajctl.State{
		Objectives: map[string]trajctl.Objective{objID: nudgeObjective(objID)},
		Scores:     []trajctl.ScoreRow{w3(objID, 0.3, 1000), w3(objID, 0.6, 2000)},
	}
}

// TestTrajctlNudgeBusEmitsOnDegradingNotHealthy is the two-regime done-condition
// witness: a degrading curve emits exactly one redirect op onto the control bus and a
// healthy curve emits none — with the redirect re-asserting the objective's OWN
// statement, so the loop re-anchors on the real goal rather than adopting the
// re-anchor prose as its objective.
func TestTrajctlNudgeBusEmitsOnDegradingNotHealthy(t *testing.T) {
	const traceD, traceH = "sess-nudge-drift", "sess-nudge-healthy"
	defer sessionctl.ClearObjective(traceD)
	defer sessionctl.ClearObjective(traceH)

	// Degrading regime: one nudge lands on the bus.
	ds := trajctlNudgeBus(degradingState("obj-d"), traceD, trajctl.Stamp{SessionID: traceD, RunID: "run-1"}, 3000)
	if len(ds) != 1 || ds[0].Action != trajctl.ActionNudge || ds[0].Signal != trajctl.SignalDrift {
		t.Fatalf("degrading decisions = %+v, want one nudge on DRIFT", ds)
	}
	if !ds[0].Delivered || ds[0].DeliverErr != "" {
		t.Fatalf("nudge not delivered onto the bus: %+v", ds[0])
	}
	assertNudgeNext(t, ds[0], true)
	if ds[0].Packet == "" || !strings.Contains(ds[0].Packet, "re-anchor") {
		t.Errorf("ledger row lost its re-anchor packet: %q", ds[0].Packet)
	}
	if n := sessionctl.RedirectPendingLen(traceD); n != 1 {
		t.Fatalf("control bus holds %d redirects for the degrading session, want 1", n)
	}

	// The redirect re-anchors on the REAL objective (its statement), not the packet.
	applied, refused := sessionctl.ApplyPendingRedirect(traceD)
	if len(refused) != 0 {
		t.Fatalf("redirect refused: %+v", refused)
	}
	if len(applied) != 1 || applied[0].Goal != "ship the trajctl nudge -> control-bus bridge" {
		t.Fatalf("applied objective = %+v, want the objective's own statement as the goal", applied)
	}
	if cur, ok := sessionctl.CurrentObjective(traceD); !ok || cur.Goal != applied[0].Goal {
		t.Fatalf("live objective = %+v (ok=%v), want the re-asserted statement", cur, ok)
	}

	// Healthy regime: the regime gate suppresses — nothing reaches the control plane.
	hs := trajctlNudgeBus(healthyNudgeState("obj-h"), traceH, trajctl.Stamp{SessionID: traceH}, 3000)
	if len(hs) != 1 || hs[0].Action != trajctl.ActionNone || hs[0].Signal != trajctl.SignalHealthy {
		t.Fatalf("healthy decisions = %+v, want one ledgered no-action on HEALTHY", hs)
	}
	if hs[0].Delivered || hs[0].Packet != "" {
		t.Errorf("healthy no-action row carried a delivery/packet: %+v", hs[0])
	}
	if hs[0].Next != nil {
		t.Fatalf("healthy next = %#v, want nil for no-action decision", hs[0].Next)
	}
	if n := sessionctl.RedirectPendingLen(traceH); n != 0 {
		t.Fatalf("healthy session received %d redirects, want 0 (a HEALTHY curve is never nudged)", n)
	}
}

// TestTrajctlNudgeBusCapturesBusRefusal proves the fail-open contract: a redirect the
// bus refuses with its closed reason (a terminal current objective) is captured on
// the decision row, never unwound, and the episode stays armed (undelivered).
func TestTrajctlNudgeBusCapturesBusRefusal(t *testing.T) {
	const trace = "sess-nudge-terminal"
	defer sessionctl.ClearObjective(trace)
	// Seed a terminal (met) current objective — not redirectable.
	sessionctl.SetObjective(trace, sessionctl.Objective{Goal: "done", Status: sessionctl.ObjectiveMet})

	ds := trajctlNudgeBus(degradingState("obj-t"), trace, trajctl.Stamp{SessionID: trace}, 4000)
	if len(ds) != 1 || ds[0].Action != trajctl.ActionNudge {
		t.Fatalf("decisions = %+v, want one nudge attempt on DRIFT", ds)
	}
	if ds[0].Delivered {
		t.Fatalf("nudge reported delivered against a terminal objective: %+v", ds[0])
	}
	if !strings.Contains(ds[0].DeliverErr, string(sessionctl.RedirectNoRedirectableState)) {
		t.Fatalf("deliver error = %q, want the closed REDIRECT_NO_REDIRECTABLE_STATE reason", ds[0].DeliverErr)
	}
	assertNudgeNext(t, ds[0], false)
	if !strings.Contains(ds[0].Next.Refusal, string(sessionctl.RedirectNoRedirectableState)) {
		t.Fatalf("next refusal = %q, want %s", ds[0].Next.Refusal, sessionctl.RedirectNoRedirectableState)
	}
	if n := sessionctl.RedirectPendingLen(trace); n != 0 {
		t.Fatalf("a refused redirect entered the mailbox (%d queued), want 0", n)
	}
}

// TestTrajctlNudgeCLIReachable pins the unpark half: the runTrajctl entry main.go's
// `case "trajctl": cmdTrajctl(...)` dispatch invokes is live and routes its
// subcommands.
func TestTrajctlNudgeCLIReachable(t *testing.T) {
	if code := runTrajctl(io.Discard, io.Discard, []string{"help"}); code != 0 {
		t.Fatalf("fak trajctl help exit = %d, want 0 (CLI unparked and reachable)", code)
	}
	if code := runTrajctl(io.Discard, io.Discard, []string{"no-such-sub"}); code != 2 {
		t.Fatalf("unknown subcommand exit = %d, want 2 (dispatch routing live)", code)
	}
}

func assertNudgeNext(t *testing.T, outcome trajctlNudgeOutcome, applied bool) {
	t.Helper()
	if outcome.Next == nil {
		t.Fatal("next witness is nil")
	}
	n := outcome.Next
	if n.Applied != applied {
		t.Fatalf("next applied = %v, want %v", n.Applied, applied)
	}
	if n.Move.Kind != sessionctl.MoveRedirect || n.Move.Render != sessionctl.RenderSystemDirective {
		t.Fatalf("next move/render = %s/%s, want redirect/system-directive", n.Move.Kind, n.Move.Render)
	}
	if n.Move.Session != sessionctl.SessionAutonomous {
		t.Fatalf("next class = %s, want autonomous", n.Move.Session)
	}
	if n.Move.Source != "trajctl_nudge" || n.Move.Gate != "trajctl-regime" {
		t.Fatalf("next identity = source %q gate %q", n.Move.Source, n.Move.Gate)
	}
	if n.Move.Payload != outcome.Packet {
		t.Fatalf("next payload = %q, want decision packet %q", n.Move.Payload, outcome.Packet)
	}
}

func TestTrajctlFleetJSONAndHuman(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	obj := trajctl.Objective{ID: "epic", Statement: "fleet epic", Status: trajctl.StatusActive}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatal(err)
	}
	if err := trajctl.Append(ledger, trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "epic", Method: trajctl.CommitScorerMethod, Version: "v1", Value: .4, Witness: trajctl.W3, SessionID: "s1"})); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if rc := runTrajctl(&out, &errOut, []string{"fleet", "--ledger", ledger, "--json"}); rc != 0 {
		t.Fatalf("rc=%d err=%s", rc, errOut.String())
	}
	if !strings.Contains(out.String(), trajctl.FleetSchema) || !strings.Contains(out.String(), `"sessions": 1`) {
		t.Fatalf("json=%s", out.String())
	}
	out.Reset()
	errOut.Reset()
	if rc := runTrajctl(&out, &errOut, []string{"fleet", "--ledger", ledger}); rc != 0 || !strings.Contains(out.String(), "fleet objectives (worst-first)") {
		t.Fatalf("human=%s err=%s", out.String(), errOut.String())
	}
}

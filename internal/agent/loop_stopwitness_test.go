package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

func TestStopWitnessRunArmEmitsSharedNextWitness(t *testing.T) {
	const trace = "trace-stop-witness-runarm"
	const missing = "file:proof.sha256"
	const wantPayload = "STOP_UNWITNESSED: missing declared witness: " + missing + ". Continue working until that witness exists."
	sessionctl.ReadStopWitnessNextRecords(trace)
	p := &recordingPlanner{}
	gateCalls := 0
	gate := func() (bool, string) {
		gateCalls++
		if gateCalls == 1 {
			return false, missing
		}
		return true, ""
	}
	if _, err := RunArm(context.Background(), p, "task", true, 2, nil, WithSessionGate(SessionGate{}, trace), WithFinalGate(gate)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	var rendered *Message
	for i := range p.seen {
		if p.seen[i].Role == RoleUser && p.seen[i].Content == wantPayload {
			rendered = &p.seen[i]
		}
	}
	if rendered == nil {
		t.Fatalf("continuation was not rendered as a user splice: %+v", p.seen)
	}
	records := sessionctl.ReadStopWitnessNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("records=%d want 1", len(records))
	}
	r := records[0]
	if r.Move.Kind != sessionctl.MoveContinue || r.Move.Render != sessionctl.RenderUserSplice || r.Move.Session != sessionctl.SessionInteractive || r.Move.Gate != "stop-witness" || r.Move.Source != "agent-turn-boundary" || r.Move.Payload != rendered.Content || !r.Applied {
		t.Fatalf("record=%+v rendered=%+v", r, *rendered)
	}
}

func TestStopWitnessRunArmNoDenialNoNextWitness(t *testing.T) {
	const trace = "trace-stop-witness-allow"
	sessionctl.ReadStopWitnessNextRecords(trace)
	p := &recordingPlanner{}
	if _, err := RunArm(context.Background(), p, "task", true, 1, nil, WithSessionGate(SessionGate{}, trace), WithFinalGate(func() (bool, string) { return true, "" })); err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if records := sessionctl.ReadStopWitnessNextRecords(trace); len(records) != 0 {
		t.Fatalf("records=%+v want none", records)
	}
}

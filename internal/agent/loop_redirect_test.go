package agent

// loop_redirect_test.go — the #2755 loop-consumer witness, the objective-change twin of
// loop_steer_test.go. It proves the exact distinction the epic names: a redirect enqueued
// out of band is APPLIED as a first-class objective change by a running arm at its turn
// boundary — the next turn carries the new objective as a SYSTEM directive (not a
// user-message steer splice), the mailbox is drained (consumed, not merely enqueued), and
// the live objective reflects the change. A refusal for an illegal (terminal-objective)
// state is proven as a unit in internal/sessionctl.

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// TestRedirectAppliedAtTurnBoundary: a redirect ENQUEUED out of band is folded into the
// live objective by a running arm AT ITS TURN BOUNDARY and carried into the turn as a
// first-class objective directive — not spliced as a user message, and not left queued.
func TestRedirectAppliedAtTurnBoundary(t *testing.T) {
	const trace = "redirect-apply-trace-2755"
	const goal = "switch to shrinking the p99 tail"
	sessionctl.ClearObjective(trace)
	defer sessionctl.ClearObjective(trace)

	if r := sessionctl.EnqueueRedirect(trace, sessionctl.Redirect{ObjectiveID: "obj-2755", Goal: goal}); r != nil {
		t.Fatalf("EnqueueRedirect: %v", r)
	}

	p := &recordingPlanner{}
	if _, err := RunArm(context.Background(), p, "original task", false, 1, nil, WithSessionTable(nil, trace)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}

	// The new objective reached the turn as a FIRST-CLASS objective directive: a system
	// message carrying the goal, NOT a user-role steer splice. This is the assertion the
	// issue's confusion-risk demands — redirect must not be "just another enqueued steer
	// message".
	var carriedAsObjective, leakedAsUserTurn bool
	for _, m := range p.seen {
		if !strings.Contains(m.Content, goal) {
			continue
		}
		switch m.Role {
		case RoleSystem:
			carriedAsObjective = true
		case RoleUser:
			leakedAsUserTurn = true
		}
	}
	if !carriedAsObjective {
		t.Fatalf("redirect objective NOT carried as a first-class system directive; planner saw %d messages, none carried the objective as a goal", len(p.seen))
	}
	if leakedAsUserTurn {
		t.Fatalf("redirect leaked in as a USER message — that is a spliced steer, not a first-class objective change (the exact thing #2755 forbids)")
	}

	// Applied == consumed: the mailbox drained and the live objective now IS the new one.
	if n := sessionctl.RedirectPendingLen(trace); n != 0 {
		t.Fatalf("redirect mailbox not drained: %d ops still queued", n)
	}
	records := sessionctl.ReadRedirectNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("redirect Next witnesses = %d, want 1", len(records))
	}
	next := records[0]
	if !next.Applied || next.Move.Kind != sessionctl.MoveRedirect || next.Move.Render != sessionctl.RenderSystemDirective || next.Move.Session != sessionctl.SessionAutonomous {
		t.Fatalf("redirect Next witness = %+v", next)
	}
	if !strings.Contains(next.Move.Payload, goal) {
		t.Fatalf("redirect Next payload = %q, want goal", next.Move.Payload)
	}

	obj, ok := sessionctl.CurrentObjective(trace)
	if !ok || obj.Goal != goal {
		t.Fatalf("live objective after boundary = %+v (ok=%v), want the new goal %q", obj, ok, goal)
	}
}

// TestNoRedirectNoObjectiveDirective proves the no-op path: with nothing enqueued and no
// objective set, no objective directive is carried — the historical loop is byte-for-byte
// unchanged.
func TestNoRedirectNoObjectiveDirective(t *testing.T) {
	const trace = "redirect-noop-trace-2755"
	sessionctl.ClearObjective(trace)
	defer sessionctl.ClearObjective(trace)

	p := &recordingPlanner{}
	if _, err := RunArm(context.Background(), p, "lone task", false, 1, nil, WithSessionTable(nil, trace)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	for _, m := range p.seen {
		if m.Role == RoleSystem && strings.Contains(m.Content, "OBJECTIVE (redirected out of band)") {
			t.Fatalf("objective directive carried with nothing enqueued: %q", m.Content)
		}
	}
}

// TestApplyRedirectNoTraceIsNoop: a run with no trace has no mailbox — a redirect queued
// under some other session's trace is not picked up by an untraced run.
func TestApplyRedirectNoTraceIsNoop(t *testing.T) {
	const other = "someone-elses-redirect-trace-2755"
	sessionctl.ClearObjective(other)
	defer sessionctl.ClearObjective(other)
	if r := sessionctl.EnqueueRedirect(other, sessionctl.Redirect{Goal: "not for you"}); r != nil {
		t.Fatalf("EnqueueRedirect: %v", r)
	}
	c := runConfig{trace: ""}
	if got := c.applyRedirect(); got != "" {
		t.Fatalf("untraced run carried an objective it should not see: %q", got)
	}
	if n := sessionctl.RedirectPendingLen(other); n != 1 {
		t.Fatalf("untraced run drained a foreign mailbox: pending = %d, want 1", n)
	}
}

func TestRedirectRefusalAtTurnBoundaryEmitsUnappliedNext(t *testing.T) {
	const trace = "redirect-refusal-next-boundary"
	sessionctl.ClearObjective(trace)
	defer sessionctl.ClearObjective(trace)
	if ref := sessionctl.EnqueueRedirect(trace, sessionctl.Redirect{Goal: "late redirect"}); ref != nil {
		t.Fatalf("EnqueueRedirect: %v", ref)
	}
	sessionctl.SetObjective(trace, sessionctl.Objective{Goal: "completed objective", Status: sessionctl.ObjectiveMet})
	if got := (runConfig{trace: trace}).applyRedirect(); !strings.Contains(got, "completed objective") {
		t.Fatalf("standing directive = %q", got)
	}
	records := sessionctl.ReadRedirectNextRecords(trace)
	if len(records) != 1 || records[0].Applied || records[0].Refusal == "" {
		t.Fatalf("refused redirect Next witnesses = %+v", records)
	}
	if records[0].Move.Kind != sessionctl.MoveRedirect || records[0].Move.Render != sessionctl.RenderSystemDirective {
		t.Fatalf("refused redirect move = %+v", records[0].Move)
	}
}

func TestRedirectNoopAtTurnBoundaryEmitsNoNext(t *testing.T) {
	const trace = "redirect-noop-next-boundary"
	sessionctl.ClearObjective(trace)
	defer sessionctl.ClearObjective(trace)
	if got := (runConfig{trace: trace}).applyRedirect(); got != "" {
		t.Fatalf("directive = %q, want empty", got)
	}
	if records := sessionctl.ReadRedirectNextRecords(trace); len(records) != 0 {
		t.Fatalf("no-op redirect Next witnesses = %+v", records)
	}
}

package sessionctl

import (
	"strings"
	"testing"
)

// TestRedirectSetsObjectiveWhenNoneDeclared: where trajctl has NOT declared an
// objective, redirect SETS it — the mailbox drains into a live active objective.
func TestRedirectSetsObjectiveWhenNoneDeclared(t *testing.T) {
	const trace = "redirect-set-2755"
	ClearObjective(trace)
	defer ClearObjective(trace)

	if _, ok := CurrentObjective(trace); ok {
		t.Fatalf("precondition: trace already has an objective")
	}
	if r := EnqueueRedirect(trace, Redirect{ObjectiveID: "obj-1", Goal: "ship the redirect op", Witness: "go test green"}); r != nil {
		t.Fatalf("EnqueueRedirect: %v", r)
	}
	if n := RedirectPendingLen(trace); n != 1 {
		t.Fatalf("pending len after enqueue = %d, want 1", n)
	}
	applied, refused := ApplyPendingRedirect(trace)
	if len(refused) != 0 {
		t.Fatalf("unexpected refusals: %v", refused)
	}
	if len(applied) != 1 || applied[0].Goal != "ship the redirect op" {
		t.Fatalf("applied = %+v, want one objective with the new goal", applied)
	}
	obj, ok := CurrentObjective(trace)
	if !ok {
		t.Fatalf("objective not set after apply")
	}
	if obj.Goal != "ship the redirect op" || obj.ID != "obj-1" || obj.Status != ObjectiveActive {
		t.Fatalf("live objective = %+v, want active obj-1 with the new goal", obj)
	}
	if n := RedirectPendingLen(trace); n != 0 {
		t.Fatalf("mailbox not drained: %d still queued", n)
	}
}

// TestRedirectUpdatesLiveObjective: where an objective IS declared and live
// (active/paused), redirect UPDATES it to the new goal.
func TestRedirectUpdatesLiveObjective(t *testing.T) {
	const trace = "redirect-update-2755"
	ClearObjective(trace)
	defer ClearObjective(trace)

	SetObjective(trace, Objective{ID: "obj-old", Goal: "old goal", Status: ObjectivePaused})
	if r := EnqueueRedirect(trace, Redirect{ObjectiveID: "obj-new", Goal: "new goal"}); r != nil {
		t.Fatalf("EnqueueRedirect against a live (paused) objective refused: %v", r)
	}
	if _, refused := ApplyPendingRedirect(trace); len(refused) != 0 {
		t.Fatalf("unexpected refusals updating a live objective: %v", refused)
	}
	obj, _ := CurrentObjective(trace)
	if obj.Goal != "new goal" || obj.ID != "obj-new" || obj.Status != ObjectiveActive {
		t.Fatalf("objective after update = %+v, want active obj-new/new goal", obj)
	}
}

// TestRedirectRefusesTerminalObjective is the illegal-state refusal witness: a session
// whose current objective is TERMINAL (met/abandoned) has no redirectable state, so a
// redirect is refused at the enqueue edge with its closed reason — never enqueued.
func TestRedirectRefusesTerminalObjective(t *testing.T) {
	for _, st := range []ObjectiveStatus{ObjectiveMet, ObjectiveAbandoned} {
		trace := "redirect-terminal-2755-" + string(st)
		ClearObjective(trace)
		SetObjective(trace, Objective{ID: "done", Goal: "already finished", Status: st})

		r := EnqueueRedirect(trace, Redirect{Goal: "try to redirect a finished objective"})
		if r == nil {
			t.Fatalf("status %q: redirect against a terminal objective was NOT refused", st)
		}
		if r.Reason != RedirectNoRedirectableState {
			t.Fatalf("status %q: refusal reason = %q, want %q", st, r.Reason, RedirectNoRedirectableState)
		}
		if n := RedirectPendingLen(trace); n != 0 {
			t.Fatalf("status %q: refused op still entered the mailbox (%d queued)", st, n)
		}
		// The terminal objective is untouched by the refused op.
		if obj, _ := CurrentObjective(trace); obj.Goal != "already finished" {
			t.Fatalf("status %q: terminal objective mutated by a refused redirect: %+v", st, obj)
		}
		ClearObjective(trace)
	}
}

// TestRedirectRefusesMalformed: an empty-goal redirect has nothing to redirect to and
// is refused malformed at the edge.
func TestRedirectRefusesMalformed(t *testing.T) {
	const trace = "redirect-malformed-2755"
	ClearObjective(trace)
	defer ClearObjective(trace)

	r := EnqueueRedirect(trace, Redirect{ObjectiveID: "x", Goal: "   "})
	if r == nil || r.Reason != RedirectMalformed {
		t.Fatalf("empty-goal redirect: got %v, want %s", r, RedirectMalformed)
	}
	if n := RedirectPendingLen(trace); n != 0 {
		t.Fatalf("malformed op entered the mailbox: %d queued", n)
	}
}

// TestRedirectableStatuses pins the closed steerable/terminal split so the "no
// redirectable state" boundary can't drift silently.
func TestRedirectableStatuses(t *testing.T) {
	cases := map[ObjectiveStatus]bool{
		ObjectiveActive:    true,
		ObjectivePaused:    true,
		"":                 true, // unset status is treated as live
		ObjectiveMet:       false,
		ObjectiveAbandoned: false,
	}
	for st, want := range cases {
		if got := (Objective{Status: st}).Redirectable(); got != want {
			t.Fatalf("Redirectable(%q) = %v, want %v", st, got, want)
		}
	}
}

// TestObjectiveDirective proves the rendered directive carries the goal (and optional
// witness) and is empty for an empty goal (so the loop stays a no-op).
func TestObjectiveDirective(t *testing.T) {
	if d := (Objective{Goal: "  "}).Directive(); d != "" {
		t.Fatalf("empty-goal directive = %q, want \"\"", d)
	}
	d := Objective{Goal: "reach parity", Witness: "bench row green"}.Directive()
	if !strings.Contains(d, "reach parity") || !strings.Contains(d, "bench row green") {
		t.Fatalf("directive %q missing goal or witness", d)
	}
}

// TestRedirectNoTraceIsNoop: an empty trace has no mailbox — enqueue is refused and the
// query/drain paths are inert.
func TestRedirectNoTraceIsNoop(t *testing.T) {
	if r := EnqueueRedirect("", Redirect{Goal: "x"}); r == nil {
		t.Fatalf("empty-trace enqueue was accepted")
	}
	if _, ok := CurrentObjective(""); ok {
		t.Fatalf("empty trace reported an objective")
	}
	if applied, refused := ApplyPendingRedirect(""); applied != nil || refused != nil {
		t.Fatalf("empty-trace apply did work: applied=%v refused=%v", applied, refused)
	}
}

func TestApplyPendingRedirectEmitsNextWitness(t *testing.T) {
	const trace = "redirect-next-applied"
	ClearObjective(trace)
	defer ClearObjective(trace)
	if ref := EnqueueRedirect(trace, Redirect{ObjectiveID: "o-next", Goal: "ship the direct redirect witness"}); ref != nil {
		t.Fatalf("enqueue: %v", ref)
	}
	applied, refused := ApplyPendingRedirect(trace)
	if len(applied) != 1 || len(refused) != 0 {
		t.Fatalf("applied=%d refused=%d", len(applied), len(refused))
	}
	records := ReadRedirectNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("records=%d, want 1", len(records))
	}
	r := records[0]
	if !r.Applied || r.Move.Kind != MoveRedirect || r.Move.Render != RenderSystemDirective || r.Move.Session != SessionAutonomous {
		t.Fatalf("record=%+v", r)
	}
	if !strings.Contains(r.Move.Payload, "ship the direct redirect witness") {
		t.Fatalf("payload=%q", r.Move.Payload)
	}
	if again := ReadRedirectNextRecords(trace); len(again) != 0 {
		t.Fatalf("readback did not drain: %d", len(again))
	}
}

func TestApplyPendingRedirectRefusalEmitsUnappliedNextWitness(t *testing.T) {
	const trace = "redirect-next-refused"
	ClearObjective(trace)
	defer ClearObjective(trace)
	if ref := EnqueueRedirect(trace, Redirect{Goal: "queued before terminal transition"}); ref != nil {
		t.Fatalf("enqueue: %v", ref)
	}
	SetObjective(trace, Objective{Goal: "already done", Status: ObjectiveMet})
	applied, refused := ApplyPendingRedirect(trace)
	if len(applied) != 0 || len(refused) != 1 {
		t.Fatalf("applied=%d refused=%d", len(applied), len(refused))
	}
	records := ReadRedirectNextRecords(trace)
	if len(records) != 1 || records[0].Applied || records[0].Refusal == "" {
		t.Fatalf("records=%+v", records)
	}
	if records[0].Move.Kind != MoveRedirect || records[0].Move.Render != RenderSystemDirective {
		t.Fatalf("move=%+v", records[0].Move)
	}
}

func TestApplyPendingRedirectNoopEmitsNoNextWitness(t *testing.T) {
	const trace = "redirect-next-noop"
	ClearObjective(trace)
	defer ClearObjective(trace)
	ApplyPendingRedirect(trace)
	if records := ReadRedirectNextRecords(trace); len(records) != 0 {
		t.Fatalf("records=%+v, want none", records)
	}
}

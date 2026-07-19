package sessionctl

import (
	"strings"
	"testing"
)

// TestConstraintTightensAdjudicationAcrossBoundary is the #2756 done-condition
// witness: the SAME tool call is allowed before the add-constraint op and denied
// after it is applied at the boundary — and the constraint takes at the NEXT
// boundary, never mid-enqueue (an in-flight turn is not re-floored beneath
// itself).
func TestConstraintTightensAdjudicationAcrossBoundary(t *testing.T) {
	const trace = "ctl-2756-before-after"
	ClearConstraints(trace)
	defer ClearConstraints(trace)

	// BEFORE: the call is allowed — the floor is unconstrained.
	if ref := ConstraintDenies(trace, "shell", ""); ref != nil {
		t.Fatalf("unconstrained floor denied the call: %v", ref)
	}
	// Operator tightens out of band: forbid the shell tool.
	if ref := EnqueueConstraint(trace, Constraint{Kind: ConstraintForbidTool, Tool: "shell", Reason: "no more shell"}); ref != nil {
		t.Fatalf("legal forbid-tool refused at the edge: %v", ref)
	}
	// Enqueued is NOT applied: before the boundary the same call is still allowed.
	if ref := ConstraintDenies(trace, "shell", ""); ref != nil {
		t.Fatalf("constraint took effect before the boundary (mid-turn): %v", ref)
	}
	if n := ConstraintPendingLen(trace); n != 1 {
		t.Fatalf("pending=%d, want 1 queued constraint", n)
	}
	// The boundary applies it.
	applied, refused := ApplyPendingConstraints(trace)
	if len(applied) != 1 || len(refused) != 0 {
		t.Fatalf("apply: applied=%d refused=%d, want 1/0", len(applied), len(refused))
	}
	if n := ConstraintPendingLen(trace); n != 0 {
		t.Fatalf("mailbox not drained: %d queued", n)
	}
	// AFTER: the SAME call is denied with the closed reason.
	ref := ConstraintDenies(trace, "shell", "")
	if ref == nil {
		t.Fatal("forbidden tool still allowed after the boundary apply")
	}
	if ref.Reason != ConstraintToolForbidden {
		t.Fatalf("denied with %q, want the closed token %q", ref.Reason, ConstraintToolForbidden)
	}
	// The floor is live and renders a standing directive for the loop to carry.
	floor, ok := CurrentConstraintFloor(trace)
	if !ok || !floor.Constrained() {
		t.Fatalf("floor = %+v ok=%v, want a live constrained floor", floor, ok)
	}
	if d := floor.Directive(); !strings.Contains(d, "shell") || !strings.Contains(d, "narrow") {
		t.Fatalf("floor directive %q does not carry the tightened state", d)
	}
	// The apply emitted an independently re-readable Next witness.
	records := ReadConstraintNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("next records=%d, want 1", len(records))
	}
	r := records[0]
	if !r.Applied || r.Move.Kind != MoveAnnotate || r.Move.Render != RenderSystemDirective ||
		r.Move.Session != SessionAutonomous || r.Move.Gate != "sessionctl-constraint" ||
		!strings.Contains(r.Move.Payload, "forbid-tool: shell") {
		t.Fatalf("next witness = %+v, want an applied sessionctl-constraint system-directive row", r)
	}
	if again := ReadConstraintNextRecords(trace); len(again) != 0 {
		t.Fatalf("next records not cleared on read: %d", len(again))
	}
}

// TestConstraintWidenRefusedAtEdge proves the monotone property's closed
// refusal: once the lane is narrowed, an op naming a region OUTSIDE it is a
// widen attempt and is refused synchronously with CONSTRAINT_WIDENS_FLOOR —
// it never enters the mailbox.
func TestConstraintWidenRefusedAtEdge(t *testing.T) {
	const trace = "ctl-2756-widen-edge"
	ClearConstraints(trace)
	defer ClearConstraints(trace)

	if ref := EnqueueConstraint(trace, Constraint{Kind: ConstraintNarrowLane, Lane: "internal/auth"}); ref != nil {
		t.Fatalf("first narrow-lane refused: %v", ref)
	}
	if applied, refused := ApplyPendingConstraints(trace); len(applied) != 1 || len(refused) != 0 {
		t.Fatalf("apply: applied=%d refused=%d, want 1/0", len(applied), len(refused))
	}
	// Narrowing WITHIN the lane stays legal.
	if ref := EnqueueConstraint(trace, Constraint{Kind: ConstraintNarrowLane, Lane: "internal/auth/token"}); ref != nil {
		t.Fatalf("narrowing within the lane refused: %v", ref)
	}
	// Widening back out is refused with the closed reason, at the edge.
	ref := EnqueueConstraint(trace, Constraint{Kind: ConstraintNarrowLane, Lane: "internal"})
	if ref == nil {
		t.Fatal("widen attempt (lane internal ⊃ internal/auth) was accepted")
	}
	if ref.Reason != ConstraintWidens {
		t.Fatalf("widen refused with %q, want the closed token %q", ref.Reason, ConstraintWidens)
	}
	if n := ConstraintPendingLen(trace); n != 1 {
		t.Fatalf("pending=%d, want only the legal narrowing queued", n)
	}
}

// TestConstraintWidenRefusedAtApplyRace proves the race half: two ops that are
// each legal against the floor AT ENQUEUE can conflict by apply time; the one
// that would widen the by-then-narrower floor is refused at apply with the same
// closed reason, witnessed, and the mailbox still drains.
func TestConstraintWidenRefusedAtApplyRace(t *testing.T) {
	const trace = "ctl-2756-widen-race"
	ClearConstraints(trace)
	defer ClearConstraints(trace)

	// Both pass the edge check: the floor has no lane yet.
	if ref := EnqueueConstraint(trace, Constraint{Kind: ConstraintNarrowLane, Lane: "internal/auth"}); ref != nil {
		t.Fatalf("first lane refused: %v", ref)
	}
	if ref := EnqueueConstraint(trace, Constraint{Kind: ConstraintNarrowLane, Lane: "internal/db"}); ref != nil {
		t.Fatalf("second lane refused at edge (no lane was live yet): %v", ref)
	}
	applied, refused := ApplyPendingConstraints(trace)
	if len(applied) != 1 || len(refused) != 1 {
		t.Fatalf("apply: applied=%d refused=%d, want 1 applied + 1 widen-refused", len(applied), len(refused))
	}
	if refused[0].Reason != ConstraintWidens {
		t.Fatalf("race refusal = %q, want %q", refused[0].Reason, ConstraintWidens)
	}
	if floor, _ := CurrentConstraintFloor(trace); floor.Lane != "internal/auth" {
		t.Fatalf("floor lane = %q, want the first (narrower-kept) lane internal/auth", floor.Lane)
	}
	if n := ConstraintPendingLen(trace); n != 0 {
		t.Fatalf("mailbox not drained after a refusal: %d queued", n)
	}
	records := ReadConstraintNextRecords(trace)
	if len(records) != 2 {
		t.Fatalf("next records=%d, want one applied + one refused row", len(records))
	}
	if !records[0].Applied || records[1].Applied {
		t.Fatalf("witness order/state wrong: %+v", records)
	}
	if !strings.Contains(records[1].Refusal, string(ConstraintWidens)) {
		t.Fatalf("refused row carries %q, want the closed %s token", records[1].Refusal, ConstraintWidens)
	}
}

// TestConstraintMalformedRefused pins the closed malformed refusals at the edge.
func TestConstraintMalformedRefused(t *testing.T) {
	const trace = "ctl-2756-malformed"
	ClearConstraints(trace)
	defer ClearConstraints(trace)

	cases := []Constraint{
		{},                                      // no kind
		{Kind: "loosen"},                        // unknown kind
		{Kind: ConstraintForbidTool},            // no tool
		{Kind: ConstraintAddDeny},               // no rule
		{Kind: ConstraintNarrowLane},            // no lane
		{Kind: ConstraintNarrowLane, Lane: "."}, // whole tree narrows nothing
		{Kind: ConstraintNarrowLane, Lane: "../outside"}, // escapes the tree
	}
	for _, c := range cases {
		ref := EnqueueConstraint(trace, c)
		if ref == nil {
			t.Fatalf("malformed constraint %+v accepted", c)
		}
		if ref.Reason != ConstraintMalformed {
			t.Fatalf("constraint %+v refused with %q, want %q", c, ref.Reason, ConstraintMalformed)
		}
	}
	if ref := EnqueueConstraint("", Constraint{Kind: ConstraintForbidTool, Tool: "shell"}); ref == nil || ref.Reason != ConstraintMalformed {
		t.Fatalf("empty trace refusal = %v, want %s", ref, ConstraintMalformed)
	}
	if n := ConstraintPendingLen(trace); n != 0 {
		t.Fatalf("a malformed op entered the mailbox: %d queued", n)
	}
}

// TestConstraintDenyRuleAndLane exercises the other two tightening kinds
// through the same before/after seam: an added deny region and a narrowed lane
// deny exactly the calls they name, each with its own closed reason.
func TestConstraintDenyRuleAndLane(t *testing.T) {
	const trace = "ctl-2756-deny-lane"
	ClearConstraints(trace)
	defer ClearConstraints(trace)

	// BEFORE: all three calls are allowed.
	for _, target := range []string{"internal/auth/token.go", "internal/sessionctl/x.go", "cmd/fak/main.go"} {
		if ref := ConstraintDenies(trace, "edit", target); ref != nil {
			t.Fatalf("unconstrained floor denied edit %s: %v", target, ref)
		}
	}
	for _, c := range []Constraint{
		{Kind: ConstraintAddDeny, Rule: "internal/auth", Reason: "stop touching the auth module"},
		{Kind: ConstraintNarrowLane, Lane: "internal"},
	} {
		if ref := EnqueueConstraint(trace, c); ref != nil {
			t.Fatalf("legal constraint %+v refused: %v", c, ref)
		}
	}
	if applied, refused := ApplyPendingConstraints(trace); len(applied) != 2 || len(refused) != 0 {
		t.Fatalf("apply: applied=%d refused=%d, want 2/0", len(applied), len(refused))
	}
	ReadConstraintNextRecords(trace)

	// AFTER: the denied region refuses with CONSTRAINT_DENY_RULE...
	if ref := ConstraintDenies(trace, "edit", "internal/auth/token.go"); ref == nil || ref.Reason != ConstraintDenyRule {
		t.Fatalf("edit inside denied region = %v, want %s", ref, ConstraintDenyRule)
	}
	// ...a sibling prefix does NOT match the region (slash-boundary exact)...
	if ref := ConstraintDenies(trace, "edit", "internal/auth2/ok.go"); ref != nil {
		t.Fatalf("edit in sibling internal/auth2 denied: %v", ref)
	}
	// ...outside the narrowed lane refuses with CONSTRAINT_OUTSIDE_LANE...
	if ref := ConstraintDenies(trace, "edit", "cmd/fak/main.go"); ref == nil || ref.Reason != ConstraintOutsideLane {
		t.Fatalf("edit outside lane = %v, want %s", ref, ConstraintOutsideLane)
	}
	// ...inside the lane and outside the denied region stays allowed...
	if ref := ConstraintDenies(trace, "edit", "internal/sessionctl/x.go"); ref != nil {
		t.Fatalf("in-lane edit denied: %v", ref)
	}
	// ...and a pathless call is bound by neither the lane nor a path rule.
	if ref := ConstraintDenies(trace, "think", ""); ref != nil {
		t.Fatalf("pathless call denied by path constraints: %v", ref)
	}
	// A deny rule also matches a bare tool name.
	if ref := EnqueueConstraint(trace, Constraint{Kind: ConstraintAddDeny, Rule: "shell"}); ref != nil {
		t.Fatalf("tool-name deny rule refused: %v", ref)
	}
	ApplyPendingConstraints(trace)
	if ref := ConstraintDenies(trace, "shell", ""); ref == nil || ref.Reason != ConstraintDenyRule {
		t.Fatalf("tool-name deny rule = %v, want %s", ref, ConstraintDenyRule)
	}
}

// TestConstraintFloorCopyAndTeardown pins the accessor contract: the returned
// floor is a deep copy (mutating it never leaks into the registry), and
// ClearConstraints tears down all per-trace state idempotently.
func TestConstraintFloorCopyAndTeardown(t *testing.T) {
	const trace = "ctl-2756-copy"
	ClearConstraints(trace)
	defer ClearConstraints(trace)

	if _, ok := CurrentConstraintFloor(trace); ok {
		t.Fatal("fresh trace reports a live floor")
	}
	if ref := EnqueueConstraint(trace, Constraint{Kind: ConstraintForbidTool, Tool: "shell"}); ref != nil {
		t.Fatalf("enqueue: %v", ref)
	}
	ApplyPendingConstraints(trace)
	floor, ok := CurrentConstraintFloor(trace)
	if !ok {
		t.Fatal("no live floor after apply")
	}
	floor.ForbiddenTools[0] = "CLOBBERED"
	if fresh, _ := CurrentConstraintFloor(trace); fresh.ForbiddenTools[0] != "shell" {
		t.Fatal("mutating the returned floor leaked into the registry (shallow copy)")
	}
	ClearConstraints(trace)
	ClearConstraints(trace) // idempotent
	if ref := ConstraintDenies(trace, "shell", ""); ref != nil {
		t.Fatalf("cleared trace still denies: %v", ref)
	}
	if _, ok := CurrentConstraintFloor(trace); ok {
		t.Fatal("cleared trace reports a live floor")
	}
}

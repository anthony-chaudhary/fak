package promptmmu

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// stableedit_test.go — issue #2201 / autoctx rung R3. The done condition's unit half:
// CheckStableEdit is derived from PlanBreakpoints and classifies the two transforms that
// actually edit the prefix today — the #555 tools[] splice and the promptmmu prune —
// against the protected prefix the plan names.

// mutatedTailPlan builds a plan whose ProtectedPrefix is the leading tools+system run
// and whose MutableTail is the diverged message tail, mirroring a live turn where the
// cacheable prefix survived but the conversation tail moved. It returns the plan plus
// the segment indices of the protected boundary so a test reads intent, not magic ints.
func mutatedTailPlan(t *testing.T) (plan BreakpointPlan, protectedEnd int) {
	t.Helper()
	baseline := []cachemeta.PromptSegment{
		seg(cachemeta.SegToolSchema, 100, "tools"),
		seg(cachemeta.SegStable, 50, "system"),
		seg(cachemeta.SegMessage, 10, "turn-1"),
		seg(cachemeta.SegMessage, 10, "turn-2"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	tracker.Observe(baseline)

	// The tool schema + system prefix stays byte-identical; only the message tail moves.
	mutated := []cachemeta.PromptSegment{
		seg(cachemeta.SegToolSchema, 100, "tools"),
		seg(cachemeta.SegStable, 50, "system"),
		seg(cachemeta.SegMessage, 10, "turn-1-EDITED"),
		seg(cachemeta.SegMessage, 10, "turn-3"),
	}
	score := tracker.Observe(mutated)
	if score.State != cachemeta.PrefixMutated {
		t.Fatalf("test setup: score.State = %v, want PrefixMutated", score.State)
	}
	plan = PlanBreakpoints(mutated, score)
	if plan.ProtectedPrefix != (Span{Start: 0, End: 2}) {
		t.Fatalf("test setup: ProtectedPrefix = %+v, want {0,2} (tools+system survived)", plan.ProtectedPrefix)
	}
	if plan.MutableTail != (Span{Start: 2, End: 4}) {
		t.Fatalf("test setup: MutableTail = %+v, want {2,4}", plan.MutableTail)
	}
	return plan, plan.ProtectedPrefix.End
}

// TestCheckStableEdit555SpliceBoundedToMutableTail: the #555 tools[] splice, when its
// edit range is bounded to the mutable tail (past the last cache_control breakpoint),
// must classify ok — this is the by-construction proof the splice preserves the cached
// prefix.
func TestCheckStableEdit555SpliceBoundedToMutableTail(t *testing.T) {
	plan, protectedEnd := mutatedTailPlan(t)

	splice := Span{Start: protectedEnd, End: 4} // the whole mutable tail
	ok, reason := CheckStableEdit(plan, splice)
	if !ok {
		t.Fatalf("CheckStableEdit(tail splice) = (false, %q), want ok — a splice past the protected prefix preserves the cached prefix", reason)
	}
}

// TestCheckStableEdit555SpliceCrossingProtectedPrefix: a splice whose boundary crosses
// back into the protected prefix must classify stable-segment-edit — it would bust the
// warm provider prefix (law L4).
func TestCheckStableEdit555SpliceCrossingProtectedPrefix(t *testing.T) {
	plan, protectedEnd := mutatedTailPlan(t)

	splice := Span{Start: protectedEnd - 1, End: 4} // reaches one segment into the prefix
	ok, reason := CheckStableEdit(plan, splice)
	if ok {
		t.Fatalf("CheckStableEdit(prefix-crossing splice) = ok, want stable-segment-edit — the edit reaches into the protected prefix")
	}
	if reason != StableEditStable {
		t.Errorf("reason = %q, want %q", reason, StableEditStable)
	}
}

// TestCheckStableEditPruneOutsideProtectedPrefix: the promptmmu prune of a floor-denied
// tool schema that sits OUTSIDE the protected prefix must classify ok.
func TestCheckStableEditPruneOutsideProtectedPrefix(t *testing.T) {
	// A turn whose cacheable prefix broke early, so a later tool schema sits in the
	// mutable tail: pruning it costs no cache warmth.
	baseline := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegMessage, 10, "turn-1"),
		seg(cachemeta.SegToolSchema, 40, "denied-tool"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	tracker.Observe(baseline)
	// Break the prefix at segment 1 so segments 1..2 (incl. the tool schema) are tail.
	mutated := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegMessage, 10, "turn-1-EDITED"),
		seg(cachemeta.SegToolSchema, 40, "denied-tool"),
	}
	score := tracker.Observe(mutated)
	plan := PlanBreakpoints(mutated, score)
	if plan.ProtectedPrefix.End != 1 {
		t.Fatalf("test setup: ProtectedPrefix = %+v, want End 1", plan.ProtectedPrefix)
	}

	prune := Span{Start: 2, End: 3} // drop the denied tool schema in the tail
	ok, reason := CheckStableEdit(plan, prune)
	if !ok {
		t.Fatalf("CheckStableEdit(tail prune) = (false, %q), want ok — the denied tool sits in the mutable tail", reason)
	}
}

// TestCheckStableEditPruneInsideProtectedPrefix: pruning a tool schema that sits INSIDE
// a protected SegToolSchema/SegStable run must classify stable-segment-edit.
func TestCheckStableEditPruneInsideProtectedPrefix(t *testing.T) {
	turn := []cachemeta.PromptSegment{
		seg(cachemeta.SegToolSchema, 40, "denied-tool"),
		seg(cachemeta.SegToolSchema, 60, "kept-tool"),
		seg(cachemeta.SegMessage, 10, "hello"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	// First observe -> PrefixUnknown -> the whole turn is the protected prefix.
	score := tracker.Observe(turn)
	plan := PlanBreakpoints(turn, score)
	if plan.ProtectedPrefix != (Span{Start: 0, End: 3}) {
		t.Fatalf("test setup: ProtectedPrefix = %+v, want the whole turn protected", plan.ProtectedPrefix)
	}

	prune := Span{Start: 0, End: 1} // drop the denied tool schema, inside the prefix
	ok, reason := CheckStableEdit(plan, prune)
	if ok {
		t.Fatalf("CheckStableEdit(prefix prune) = ok, want stable-segment-edit — the tool schema is inside the protected prefix")
	}
	if reason != StableEditStable {
		t.Errorf("reason = %q, want %q", reason, StableEditStable)
	}
}

// TestCheckStableEditSealedSpanIsUnsafeAnywhere: an edit intersecting a SegSealed span is
// unsafe wherever it sits (refusal rule 3), reported as sealed-segment-edit — checked
// ahead of the protected-prefix reason.
func TestCheckStableEditSealedSpanIsUnsafeAnywhere(t *testing.T) {
	turn := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegSealed, 20, "quarantined-secret"),
		seg(cachemeta.SegMessage, 10, "hello"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	score := tracker.Observe(turn)
	plan := PlanBreakpoints(turn, score)
	// Sanity: the sealed span is named in the plan.
	var sealedNamed bool
	for _, u := range plan.UnsafeToCompact {
		if u.Reason == UnsafeSealed {
			sealedNamed = true
		}
	}
	if !sealedNamed {
		t.Fatalf("test setup: plan.UnsafeToCompact = %+v, want a sealed span", plan.UnsafeToCompact)
	}

	edit := Span{Start: 1, End: 2} // the sealed segment itself
	ok, reason := CheckStableEdit(plan, edit)
	if ok {
		t.Fatalf("CheckStableEdit(sealed edit) = ok, want sealed-segment-edit")
	}
	if reason != StableEditSealed {
		t.Errorf("reason = %q, want %q (sealed takes precedence over the prefix reason)", reason, StableEditSealed)
	}
}

// TestCheckStableEditEmptyEditIsOk: an empty edit range touches nothing and is always ok.
func TestCheckStableEditEmptyEditIsOk(t *testing.T) {
	plan, _ := mutatedTailPlan(t)
	if ok, reason := CheckStableEdit(plan, Span{Start: 2, End: 2}); !ok {
		t.Errorf("CheckStableEdit(empty edit) = (false, %q), want ok", reason)
	}
}

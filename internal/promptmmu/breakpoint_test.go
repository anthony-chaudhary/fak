package promptmmu

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// seg builds a cachemeta.PromptSegment for these tests: content bytes are only
// compared by cachemeta.Diverge, so a distinct string per logical segment is enough to
// drive divergence.
func seg(kind cachemeta.SegmentKind, tokens int64, content string) cachemeta.PromptSegment {
	return cachemeta.PromptSegment{Kind: kind, Tokens: tokens, Content: []byte(content)}
}

func TestBreakpointPlanNamesProtectedPrefix(t *testing.T) {
	turn := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegToolSchema, 50, "tools"),
		seg(cachemeta.SegMessage, 10, "hello"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("5m", abi.ScopeAgent)

	// First Observe: no baseline yet -> PrefixUnknown -> the whole turn is protected.
	first := tracker.Observe(turn)
	plan := PlanBreakpoints(turn, first)
	if plan.State != cachemeta.PrefixUnknown {
		t.Fatalf("State = %v, want PrefixUnknown", plan.State)
	}
	if plan.ProtectedPrefix != (Span{Start: 0, End: len(turn)}) {
		t.Fatalf("ProtectedPrefix = %+v, want the whole turn on first observation", plan.ProtectedPrefix)
	}
	if !plan.MutableTail.Empty() {
		t.Fatalf("MutableTail = %+v, want empty on first observation", plan.MutableTail)
	}

	// Second Observe with the identical turn: PrefixStable -> still the whole turn is
	// protected (byte-identical to the baseline), MutableTail stays empty.
	second := tracker.Observe(turn)
	plan = PlanBreakpoints(turn, second)
	if plan.State != cachemeta.PrefixStable {
		t.Fatalf("State = %v, want PrefixStable", plan.State)
	}
	if plan.ProtectedPrefix != (Span{Start: 0, End: len(turn)}) {
		t.Fatalf("ProtectedPrefix = %+v, want the whole turn when stable", plan.ProtectedPrefix)
	}
	if !plan.MutableTail.Empty() {
		t.Fatalf("MutableTail = %+v, want empty when stable", plan.MutableTail)
	}
	if plan.TTLMillis != 5*60*1000 {
		t.Errorf("TTLMillis = %d, want the tracker's configured 5m TTL", plan.TTLMillis)
	}
	if plan.ShareScope != abi.ScopeAgent {
		t.Errorf("ShareScope = %v, want ScopeAgent", plan.ShareScope)
	}
}

func TestBreakpointPlanNamesMutableTailOnDivergence(t *testing.T) {
	baseline := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegToolSchema, 50, "tools"),
		seg(cachemeta.SegMessage, 10, "turn-1"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	tracker.Observe(baseline) // seed the baseline

	// Mutate the FIRST segment (the system prompt) so the divergence lands inside what
	// was the protected span: only segment 0's 100 stable tokens survive.
	mutated := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system-EDITED"),
		seg(cachemeta.SegToolSchema, 50, "tools"),
		seg(cachemeta.SegMessage, 10, "turn-1"),
	}
	score := tracker.Observe(mutated)
	if score.State != cachemeta.PrefixMutated {
		t.Fatalf("test setup: score.State = %v, want PrefixMutated", score.State)
	}

	plan := PlanBreakpoints(mutated, score)
	if plan.State != cachemeta.PrefixMutated {
		t.Fatalf("State = %v, want PrefixMutated", plan.State)
	}
	if plan.ProtectedPrefix.End != 0 {
		t.Fatalf("ProtectedPrefix = %+v, want End 0 (the divergence broke at segment 0)", plan.ProtectedPrefix)
	}
	if plan.MutableTail != (Span{Start: 0, End: len(mutated)}) {
		t.Fatalf("MutableTail = %+v, want the whole turn once the prefix broke at segment 0", plan.MutableTail)
	}
}

func TestBreakpointPlanNamesUnsafeToCompactSpans(t *testing.T) {
	turn := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegSealed, 20, "quarantined-secret"),
		seg(cachemeta.SegMessage, 10, "hello"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	score := tracker.Observe(turn)

	plan := PlanBreakpoints(turn, score)
	if len(plan.UnsafeToCompact) != 1 {
		t.Fatalf("UnsafeToCompact = %+v, want exactly one sealed-span entry", plan.UnsafeToCompact)
	}
	u := plan.UnsafeToCompact[0]
	if u.Reason != UnsafeSealed {
		t.Errorf("Reason = %q, want %q", u.Reason, UnsafeSealed)
	}
	if u.Kind != cachemeta.SegSealed {
		t.Errorf("Kind = %v, want SegSealed", u.Kind)
	}
	if u.Span != (Span{Start: 1, End: 2}) {
		t.Errorf("Span = %+v, want {1,2} (the sealed segment's own index)", u.Span)
	}
}

func TestBreakpointPlanNamesDivergentSegmentAsUnsafe(t *testing.T) {
	baseline := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegToolSchema, 50, "tools"),
		seg(cachemeta.SegMessage, 10, "turn-1"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	tracker.Observe(baseline)

	// Mutate only the SECOND segment: segment 0 (100 tokens) is still a stable
	// prefix, so the divergence lands INSIDE the (previously) protected span at
	// segment index 1.
	mutated := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegToolSchema, 50, "tools-EDITED"),
		seg(cachemeta.SegMessage, 10, "turn-1"),
	}
	score := tracker.Observe(mutated)
	if score.FirstDivergentSegment != 1 {
		t.Fatalf("test setup: FirstDivergentSegment = %d, want 1", score.FirstDivergentSegment)
	}

	plan := PlanBreakpoints(mutated, score)
	if plan.ProtectedPrefix != (Span{Start: 0, End: 1}) {
		t.Fatalf("ProtectedPrefix = %+v, want {0,1} (only segment 0 survived)", plan.ProtectedPrefix)
	}

	var found bool
	for _, u := range plan.UnsafeToCompact {
		if u.Reason == UnsafeDivergence && u.Span == (Span{Start: 1, End: 2}) {
			found = true
		}
	}
	if !found {
		t.Errorf("UnsafeToCompact = %+v, want an UnsafeDivergence entry naming segment 1", plan.UnsafeToCompact)
	}
}

func TestBreakpointPlanSealedSpanBrokenStillNamesSealedEvenWhenStable(t *testing.T) {
	// A ProtectedSpanBroken score (Observe hit a sealed segment while otherwise
	// matching) still names the sealed segment via UnsafeToCompact — sealed spans are
	// a hazard independent of the state's stable/mutated shape.
	turn := []cachemeta.PromptSegment{
		seg(cachemeta.SegStable, 100, "system"),
		seg(cachemeta.SegSealed, 20, "quarantined-secret"),
	}
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	tracker.Observe(turn)
	score := tracker.Observe(turn) // identical turn again, but the sealed span persists

	if !score.ProtectedSpanBroken {
		t.Fatalf("test setup: score.ProtectedSpanBroken = false, want true")
	}

	plan := PlanBreakpoints(turn, score)
	var sealedFound bool
	for _, u := range plan.UnsafeToCompact {
		if u.Reason == UnsafeSealed {
			sealedFound = true
		}
	}
	if !sealedFound {
		t.Errorf("UnsafeToCompact = %+v, want a sealed-span entry even though the score state is %v", plan.UnsafeToCompact, score.State)
	}
}

func TestBreakpointPlanEmptyTurnIsAllEmptySpans(t *testing.T) {
	tracker := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	score := tracker.Observe(nil)
	plan := PlanBreakpoints(nil, score)
	if !plan.ProtectedPrefix.Empty() || !plan.MutableTail.Empty() {
		t.Errorf("plan = %+v, want empty ProtectedPrefix and MutableTail for an empty turn", plan)
	}
	if len(plan.UnsafeToCompact) != 0 {
		t.Errorf("UnsafeToCompact = %+v, want none for an empty turn", plan.UnsafeToCompact)
	}
}

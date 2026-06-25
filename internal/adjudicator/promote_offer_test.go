package adjudicator

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// feedClean folds n clean complain-mode admits for a tool.
func feedClean(l *Ledger, tool string, n int) {
	for i := 0; i < n; i++ {
		l.Emit(decideEvent(abi.EvDecide, tool, admitAndLogVerdict()))
	}
}

// TestPromotableReturnsToolsAtThresholdWithNoHardEvents is the #673 anchor: Promotable(n)
// returns exactly the tools at >= n clean events with zero hard-refusal events, sorted.
func TestPromotableReturnsToolsAtThresholdWithNoHardEvents(t *testing.T) {
	l := NewLedger()
	feedClean(l, "ready_tool", 10) // at threshold, clean
	feedClean(l, "early_tool", 3)  // below threshold
	feedClean(l, "also_ready", 10) // at threshold, clean

	// dirty_tool reaches the threshold but then provokes a hard refusal -> disqualified.
	feedClean(l, "dirty_tool", 10)
	l.Emit(decideEvent(abi.EvDecide, "dirty_tool", abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonSelfModify}))
	feedClean(l, "dirty_tool", 10) // even rebuilding clean events does not re-qualify it

	got := l.Promotable(10)
	var names []string
	for _, o := range got {
		names = append(names, o.Tool)
		if o.CleanEvents < 10 {
			t.Errorf("offer %q has %d clean events, want >= 10", o.Tool, o.CleanEvents)
		}
	}
	want := []string{"also_ready", "ready_tool"} // sorted, dirty_tool + early_tool excluded
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("Promotable(10) = %v, want %v (sorted, no early/dirty)", names, want)
	}
}

// TestPromotableMutatesNothing is the emit-a-diff-and-STOP guarantee: Promotable reads
// the ledger and returns offers — it changes neither the ledger counts nor any Policy.
// Calling it twice yields the same result and leaves the counts intact.
func TestPromotableMutatesNothing(t *testing.T) {
	l := NewLedger()
	feedClean(l, "trial_tool", 5)

	before := l.Clean("trial_tool")
	first := l.Promotable(5)
	second := l.Promotable(5)

	if l.Clean("trial_tool") != before {
		t.Fatalf("Promotable mutated the ledger: clean %d -> %d", before, l.Clean("trial_tool"))
	}
	if len(first) != 1 || len(second) != 1 || first[0] != second[0] {
		t.Fatalf("Promotable is not pure: first=%v second=%v", first, second)
	}
}

// TestPromotionOfferReviewIsReviewableNotApplied confirms the offer's only output is a
// reviewable fragment naming the tool — never an applied Policy mutation. (Promotable
// takes no Policy and returns data; this pins the human-facing review string.)
func TestPromotionOfferReviewIsReviewableNotApplied(t *testing.T) {
	o := PromotionOffer{Tool: "provision_widget", CleanEvents: 12}
	r := o.Review()
	if !strings.Contains(r, "provision_widget") || !strings.Contains(r, "Policy.Allow") {
		t.Fatalf("Review() should propose adding the tool to Policy.Allow: %q", r)
	}
}

// TestPromotableEmptyWhenNothingReady covers the no-candidates case.
func TestPromotableEmptyWhenNothingReady(t *testing.T) {
	l := NewLedger()
	feedClean(l, "tool_a", 2)
	if got := l.Promotable(10); len(got) != 0 {
		t.Fatalf("Promotable(10) = %v, want empty", got)
	}
}

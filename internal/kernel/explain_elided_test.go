package kernel

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestFoldExplainSurfacesElidedRungs is the #668 anchor: when a max-rank verdict
// (a Deny) short-circuits the fold, the rungs AFTER it did not run — and the trace
// must record them, marked Elided, so "which rung decided vs which were elided" is
// observable instead of those rungs silently vanishing.
func TestFoldExplainSurfacesElidedRungs(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer, By: "grammar"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonSelfModify, By: "monitor"}}, // max-rank: short-circuits
		fakeAdj{abi.Verdict{Kind: abi.VerdictAllow, By: "late-allow"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer, By: "tail"}},
	}
	v, d := FoldExplain(ctx, chain, callInline("edit_kernel", "{}"))

	// The verdict is unchanged by the trace addition (the Deny still wins).
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("verdict = %s/%s, want DENY/SELF_MODIFY", kindName(v.Kind), abi.ReasonName(v.Reason))
	}
	// All four rungs appear in the trace, in chain order.
	if len(d.Rungs) != 4 {
		t.Fatalf("rungs = %d, want 4 (2 run + 2 elided)", len(d.Rungs))
	}
	if d.Rungs[0].Elided || d.Rungs[1].Elided {
		t.Errorf("rungs 0,1 ran and must not be Elided: %+v / %+v", d.Rungs[0], d.Rungs[1])
	}
	if !d.Rungs[1].Winner {
		t.Errorf("rung[1] (the Deny) must be the winner")
	}
	if !d.Rungs[2].Elided || !d.Rungs[3].Elided {
		t.Errorf("rungs 2,3 came after the max-rank verdict and must be Elided: %+v / %+v", d.Rungs[2], d.Rungs[3])
	}
	// An elided rung is recorded but NOT evaluated: it carries its identity (Rung) but
	// no decision (no By / Reason / Winner).
	if d.Rungs[2].By != "" || d.Rungs[2].Winner || d.Rungs[2].Reason != "" {
		t.Errorf("elided rung must carry no decision, got %+v", d.Rungs[2])
	}
	if d.Rungs[2].Rung == "" {
		t.Errorf("elided rung must still record its adjudicator identity")
	}
	// The human text marks the elided rungs.
	if !strings.Contains(d.Text(), "ELIDED") {
		t.Errorf("Text() should mark elided rungs:\n%s", d.Text())
	}
}

// TestFoldExplainNoElisionWhenNoShortCircuit confirms the addition is inert when no
// max-rank verdict short-circuits: a chain that only defers/quarantines records every
// rung as run, none Elided — so existing traces are unchanged.
func TestFoldExplainNoElisionWhenNoShortCircuit(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer, By: "a"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonMalformed, By: "b"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer, By: "c"}},
	}
	_, d := FoldExplain(ctx, chain, callInline("t", "{}"))
	if len(d.Rungs) != 3 {
		t.Fatalf("rungs = %d, want 3 (all ran)", len(d.Rungs))
	}
	for i, r := range d.Rungs {
		if r.Elided {
			t.Errorf("rung[%d] must not be Elided when nothing short-circuited: %+v", i, r)
		}
	}
}

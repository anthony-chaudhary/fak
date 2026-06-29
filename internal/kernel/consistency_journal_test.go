package kernel

// consistency_journal_test.go — the #1317 acceptance witness: the call's declared
// consistency level appears VERBATIM in the decision journal (kernel.Decision, the
// off-hot-path trace fak preflight --explain/--json, the gateway decision trace, and
// the agent run report all surface), and threading it changes NO adjudication verdict.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// callWithConsistency is callInline plus a consistency level on Meta.
func callWithConsistency(tool, args, level string) *abi.ToolCall {
	c := callInline(tool, args)
	c.Meta = map[string]string{abi.MetaConsistency: level}
	return c
}

// TestDecisionJournalRecordsConsistency proves (b): the chosen level is recorded
// verbatim, and an unset call defaults to STRICT in the journal (the default is applied
// AND recorded, never left blank).
func TestDecisionJournalRecordsConsistency(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}}}

	// Unset → STRICT in the journal.
	_, d := FoldExplain(ctx, chain, callInline("t", "{}"))
	if d.Consistency != "STRICT" {
		t.Fatalf("unset call: Decision.Consistency = %q, want %q (the default must be applied AND recorded)", d.Consistency, "STRICT")
	}

	// Each declared level is recorded verbatim.
	for _, level := range []string{"STRICT", "BOUNDED_STALE", "BEST_EFFORT", "SPECULATIVE"} {
		_, d := FoldExplain(ctx, chain, callWithConsistency("t", "{}", level))
		if d.Consistency != level {
			t.Errorf("level %q: Decision.Consistency = %q, want it recorded verbatim", level, d.Consistency)
		}
	}
}

// TestConsistencyChangesNoVerdict proves (a): threading the field changes no
// adjudication verdict — the call with a relaxed level resolves to the byte-identical
// verdict the unset call does, across the lattice corners.
func TestConsistencyChangesNoVerdict(t *testing.T) {
	ctx := context.Background()
	deny := abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "monitor"}
	allow := abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}
	chains := map[string][]abi.Adjudicator{
		"allow":     {fakeAdj{allow}},
		"deny-wins": {fakeAdj{allow}, fakeAdj{deny}},
		"empty":     nil,
	}
	for name, chain := range chains {
		base := Fold(ctx, chain, callInline("t", "{}"))
		for _, level := range []string{"BOUNDED_STALE", "BEST_EFFORT", "SPECULATIVE"} {
			got := Fold(ctx, chain, callWithConsistency("t", "{}", level))
			if got.Kind != base.Kind || got.Reason != base.Reason || got.By != base.By {
				t.Errorf("%s @ %s: verdict {%s,%s,%s} != unset {%s,%s,%s} — the consistency field must not change adjudication",
					name, level, kindName(got.Kind), abi.ReasonName(got.Reason), got.By,
					kindName(base.Kind), abi.ReasonName(base.Reason), base.By)
			}
		}
	}
}

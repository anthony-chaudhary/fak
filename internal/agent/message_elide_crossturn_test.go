package agent

import (
	"fmt"
	"strings"
	"testing"
)

// dedupWitnessBody builds a multi-line shell-style tool result that clears the cross-turn floors
// (>= crossTurnMinDupLines lines, >= crossTurnMinDupBytes bytes) while staying far UNDER the 16 KB
// default head+tail threshold — the exact size class #5254 measured (~7.85 KB per duplicated row).
func dedupWitnessBody(tag string) string {
	var b strings.Builder
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&b, "%s pkg/mod/scan line %02d: checked symbol at offset %d\n", tag, i, i*7+3)
	}
	return b.String()
}

// TestElideMessagesFoldsSubThresholdRepeatedResults is the #5254 witness for the decoded wire (the
// OpenAI / Responses path a native Codex session takes through fak). A shell result repeated
// verbatim across turns folds to ONE verbatim copy plus pointers, even though every body is far
// under the head+tail threshold — the case a size gate provably misses, because repetition is a
// property of the SET and the gate only ever looks at one member.
func TestElideMessagesFoldsSubThresholdRepeatedResults(t *testing.T) {
	const threshold = 16384 // gateway.DefaultElideResultBytes
	body := dedupWitnessBody("OUT")
	if len(body) >= threshold {
		t.Fatalf("fixture must sit UNDER the head+tail threshold to prove the gap: len=%d", len(body))
	}
	in := []Message{
		{Role: "system", Content: "You are a coding agent."}, // 0
		{Role: "tool", ToolCallID: "t1", Content: body},      // 1 earliest → stays verbatim
		{Role: "assistant", Content: "re-running the scan"},  // 2
		{Role: "tool", ToolCallID: "t3", Content: body},      // 3 repeat → folds
		{Role: "assistant", Content: "once more"},            // 4
		{Role: "tool", ToolCallID: "t5", Content: body},      // 5 repeat → folds
		{Role: "assistant", Content: "checking the recent"},  // 6
		{Role: "tool", ToolCallID: "t7", Content: body},      // 7 repeat but RECENT → protected
		{Role: "assistant", Content: "done"},                 // 8
		{Role: "user", Content: "summarize"},                 // 9
	}
	orig3 := in[3].Content

	// Head+tail alone is a no-op on this input: nothing is over threshold. Any fire is dedup's.
	out, oc := ElideMessages(in, threshold)
	if oc.Reason != ElideReasonNone {
		t.Fatalf("repeated sub-threshold results must fire dedup, got reason=%q", oc.Reason)
	}
	if oc.Elided != 2 {
		t.Fatalf("expected exactly messages 3 and 5 to fold, got elided=%d", oc.Elided)
	}
	if oc.ShedBytes <= 0 {
		t.Fatalf("expected positive ShedBytes, got %d", oc.ShedBytes)
	}
	if out[1].Content != body {
		t.Error("the EARLIEST occurrence must stay verbatim (keep-earliest); its bytes are the referent")
	}
	for _, i := range []int{3, 5} {
		if !strings.Contains(out[i].Content, "fak dedup") {
			t.Errorf("repeated result %d was not folded to a pointer: %q", i, out[i].Content)
		}
		if len(out[i].Content) >= len(body) {
			t.Errorf("fold of message %d did not shrink: %d >= %d", i, len(out[i].Content), len(body))
		}
		// The pointer must name the earliest source (message 1), not a nearer repeat.
		if !strings.Contains(out[i].Content, "turn 1,") {
			t.Errorf("message %d must point at the earliest occurrence, got %q", i, out[i].Content)
		}
	}
	if out[7].Content != body {
		t.Error("a repeat inside the recent working set must be left intact")
	}
	if in[3].Content != orig3 {
		t.Error("input slice was mutated in place (must be copy-on-write)")
	}
	if out[3].Role != "tool" || out[3].ToolCallID != "t3" {
		t.Error("fold corrupted the tool message's Role/ToolCallID")
	}
}

// TestDedupMessagesCrossTurnPrefixMonotonic asserts the one property the decoded path may NOT
// inherit from the Anthropic witness: a message's folded rendering is a function of the messages
// STRICTLY BEFORE it alone, so appending a turn never rewrites an earlier turn's bytes. On the
// byte-splice wire cache-safety rests on splicing after a cache_control breakpoint; the decoded
// wire has no breakpoint to anchor on and instead needs this stability directly, so it is asserted
// here rather than assumed. lastEligible is pinned to the full length so the recent-window band
// cannot mask a drift.
func TestDedupMessagesCrossTurnPrefixMonotonic(t *testing.T) {
	a, b := dedupWitnessBody("AAA"), dedupWitnessBody("BBB")
	full := []Message{
		{Role: "tool", ToolCallID: "t0", Content: a},
		{Role: "assistant", Content: "step"},
		{Role: "tool", ToolCallID: "t2", Content: b},
		{Role: "tool", ToolCallID: "t3", Content: a},
		{Role: "assistant", Content: "step"},
		{Role: "tool", ToolCallID: "t5", Content: b},
		{Role: "tool", ToolCallID: "t6", Content: a + b},
	}
	ref, folded, _ := dedupMessagesCrossTurn(full, len(full))
	if folded == 0 {
		t.Fatal("fixture folded nothing — it cannot witness monotonicity")
	}
	for k := 2; k <= len(full); k++ {
		got, _, _ := dedupMessagesCrossTurn(full[:k], k)
		for i := 0; i < k; i++ {
			if got[i].Content != ref[i].Content {
				t.Fatalf("prefix k=%d changed message %d's rendering: appending a turn must not rewrite an earlier turn\n got: %q\nwant: %q",
					k, i, got[i].Content, ref[i].Content)
			}
		}
	}
}

// TestDedupMessagesCrossTurnIdentityCases pins the fail-safe returns: fewer than two tool blocks
// has nothing to match against, and a non-repeating transcript must come back untouched.
func TestDedupMessagesCrossTurnIdentityCases(t *testing.T) {
	body := dedupWitnessBody("ONE")
	single := []Message{
		{Role: "user", Content: "q"},
		{Role: "tool", ToolCallID: "t", Content: body},
		{Role: "assistant", Content: "a"},
	}
	if out, folded, shed := dedupMessagesCrossTurn(single, len(single)); folded != 0 || shed != 0 || out[1].Content != body {
		t.Errorf("a lone tool result has no earlier source and must be identity, got folded=%d shed=%d", folded, shed)
	}
	distinct := []Message{
		{Role: "tool", ToolCallID: "t0", Content: dedupWitnessBody("XXX")},
		{Role: "tool", ToolCallID: "t1", Content: dedupWitnessBody("YYY")},
		{Role: "tool", ToolCallID: "t2", Content: dedupWitnessBody("ZZZ")},
	}
	if _, folded, _ := dedupMessagesCrossTurn(distinct, len(distinct)); folded != 0 {
		t.Errorf("non-repeating results must not fold, got folded=%d", folded)
	}
	// A short repeat is below the min-span floors and must stay verbatim (the pointer would cost
	// more structure than it saves).
	short := "ok\n"
	tiny := []Message{
		{Role: "tool", ToolCallID: "t0", Content: short},
		{Role: "tool", ToolCallID: "t1", Content: short},
	}
	if _, folded, _ := dedupMessagesCrossTurn(tiny, len(tiny)); folded != 0 {
		t.Errorf("a sub-floor repeat must not fold, got folded=%d", folded)
	}
}

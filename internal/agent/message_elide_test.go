package agent

import (
	"strings"
	"testing"
)

// TestElideMessagesShrinksOldToolResultsKeepsWorkingSet is the witness for the decoded
// (OpenAI / in-kernel) elision path a LOCAL model served by fak takes — GLM-5.2 / Qwen-3.6-27B,
// where anthropicPassthrough() is false and the byte-splice transform never fires. An OLD
// oversized tool-role message is shrunk to head+tail; a RECENT oversized tool result, a small
// tool result, and every non-tool message are left intact; the input slice is never mutated.
func TestElideMessagesShrinksOldToolResultsKeepsWorkingSet(t *testing.T) {
	const threshold = 1024
	bigOld := strings.Repeat("OLD scrolled-past command output line. ", 200)  // ~7.6 KB, eligible → shed
	bigRecent := strings.Repeat("RECENT working-set output line. ", 200)      // ~6.4 KB, recent → kept
	in := []Message{
		{Role: "system", Content: "You are a coding agent."},     // 0
		{Role: "user", Content: "refactor the parser"},           // 1
		{Role: "tool", ToolCallID: "t2", Content: bigOld},        // 2 OLD oversized tool result → shrink
		{Role: "assistant", Content: "analyzing"},                // 3
		{Role: "user", Content: "now the lexer"},                 // 4
		{Role: "assistant", Content: "calling read_file"},        // 5
		{Role: "tool", ToolCallID: "t6", Content: bigRecent},     // 6 RECENT (last-4 window) → keep
		{Role: "assistant", Content: "done"},                     // 7
	}
	// Snapshot to detect in-place mutation of the caller's slice.
	origIdx2 := in[2].Content

	out, oc := ElideMessages(in, threshold)
	if oc.Reason != ElideReasonNone || oc.Elided != 1 {
		t.Fatalf("expected exactly the old tool result to shed, got reason=%q elided=%d", oc.Reason, oc.Elided)
	}
	if oc.ShedBytes <= 0 {
		t.Fatalf("expected positive ShedBytes, got %d", oc.ShedBytes)
	}
	if len(out[2].Content) >= len(bigOld) || !strings.Contains(out[2].Content, "tool_result output elided") {
		t.Errorf("old tool result was not shrunk to head+tail: len=%d", len(out[2].Content))
	}
	if out[6].Content != bigRecent {
		t.Error("recent (working-set) tool result was wrongly shrunk")
	}
	for _, i := range []int{0, 1, 3, 4, 5, 7} {
		if out[i].Content != in[i].Content {
			t.Errorf("non-eligible message %d was changed", i)
		}
	}
	if in[2].Content != origIdx2 {
		t.Error("input slice was mutated in place (must be copy-on-write)")
	}
	// ToolCallID and Role survive the shrink.
	if out[2].Role != "tool" || out[2].ToolCallID != "t2" {
		t.Error("shrink corrupted the tool message's Role/ToolCallID")
	}
}

// TestElideMessagesIdentityCases pins the fail-safe identity returns.
func TestElideMessagesIdentityCases(t *testing.T) {
	big := strings.Repeat("x", 4000)
	msgs := []Message{
		{Role: "user", Content: "q"},
		{Role: "tool", ToolCallID: "t", Content: big},
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "b"},
		{Role: "assistant", Content: "c"},
	}
	// Disabled (threshold 0).
	if out, oc := ElideMessages(msgs, 0); oc.Reason != ElideReasonOff || len(out) != len(msgs) {
		t.Errorf("threshold 0 must be off/identity, got reason=%q", oc.Reason)
	}
	// The only big tool result sits INSIDE the recent window (len-4 = 1, so idx 1 is protected) → no fire.
	if out, oc := ElideMessages(msgs, 1024); oc.Reason != ElideReasonUnderThreshold {
		t.Errorf("a big result inside the recent window must be protected (under_threshold), got reason=%q", oc.Reason)
	} else if out[1].Content != big {
		t.Error("protected recent result must be left unchanged")
	}
	// Empty input.
	if _, oc := ElideMessages(nil, 1024); oc.Reason != ElideReasonOff {
		t.Errorf("empty input must be off, got reason=%q", oc.Reason)
	}
}

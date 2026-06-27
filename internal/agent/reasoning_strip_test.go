package agent

import (
	"strings"
	"testing"
)

// TestStripReasoningSingleBlock witnesses the common case: one well-formed <think>…</think>
// block is removed and the text after it (the user-visible answer) survives intact.
func TestStripReasoningSingleBlock(t *testing.T) {
	in := "<think>let me work out 2+2 step by step</think>The answer is 4."
	got := StripReasoning(in)
	if got != "The answer is 4." {
		t.Fatalf("single block not stripped cleanly: %q", got)
	}
	if strings.Contains(got, "think") {
		t.Errorf("reasoning leaked into output: %q", got)
	}
}

// TestStripReasoningPreservesSurroundingText pins that text BEFORE a block is kept too, not just
// text after it.
func TestStripReasoningPreservesSurroundingText(t *testing.T) {
	in := "Prefix. <think>hidden</think> Suffix."
	got := StripReasoning(in)
	if got != "Prefix.  Suffix." {
		t.Fatalf("surrounding text not preserved: %q", got)
	}
}

// TestStripReasoningMultipleBlocks witnesses that any number of blocks are removed and the
// interleaved answer text survives in order.
func TestStripReasoningMultipleBlocks(t *testing.T) {
	in := "<think>plan A</think>First.<think>plan B</think>Second.<think>plan C</think>Third."
	got := StripReasoning(in)
	if got != "First.Second.Third." {
		t.Fatalf("multiple blocks not stripped: %q", got)
	}
}

// TestStripReasoningUnclosedBlock pins the truncated-stream policy: an open <think> with no
// closing tag is stripped from the open tag to end-of-string. Text BEFORE the open tag survives.
func TestStripReasoningUnclosedBlock(t *testing.T) {
	in := "Here goes: <think>I am reasoning but the stream got cut off mid-thought"
	got := StripReasoning(in)
	if got != "Here goes:" {
		t.Fatalf("unclosed block not stripped to end: %q", got)
	}
	if strings.Contains(got, "reasoning") {
		t.Errorf("truncated reasoning leaked: %q", got)
	}
}

// TestStripReasoningIdentityNoTags pins the identity case: input with no <think> tag is returned
// completely unchanged.
func TestStripReasoningIdentityNoTags(t *testing.T) {
	in := "Just a plain answer with no reasoning markers at all."
	if got := StripReasoning(in); got != in {
		t.Fatalf("input without think tags must be identity, got %q", got)
	}
	if got := StripReasoning(""); got != "" {
		t.Fatalf("empty input must be identity, got %q", got)
	}
}

// TestStripReasoningWhitespaceTidy pins that stripping a leading block does not leave a blank
// line at the top, and that a block sitting on its own line collapses cleanly rather than leaving
// a triple-newline gap.
func TestStripReasoningWhitespaceTidy(t *testing.T) {
	// A leading block on its own line: removal must not leave a top blank line.
	in := "<think>scratch</think>\n\nThe answer."
	if got := StripReasoning(in); got != "The answer." {
		t.Fatalf("leading-block whitespace not tidied: %q", got)
	}
	// A block between two paragraphs must not leave a 3+ newline gap.
	in2 := "Para one.\n\n<think>scratch</think>\n\nPara two."
	got := StripReasoning(in2)
	if strings.Contains(got, "\n\n\n") {
		t.Errorf("collapsed gap still has a triple newline: %q", got)
	}
	if !strings.Contains(got, "Para one.") || !strings.Contains(got, "Para two.") {
		t.Errorf("paragraph text lost during tidy: %q", got)
	}
}

// TestStripReasoningCaseInsensitiveTags pins that non-canonical-case tags (<Think>/<THINK>) some
// checkpoints emit are still recognised and stripped.
func TestStripReasoningCaseInsensitiveTags(t *testing.T) {
	in := "<THINK>upper</THINK>kept<Think>mixed</Think>also"
	if got := StripReasoning(in); got != "keptalso" {
		t.Fatalf("case-insensitive tags not stripped: %q", got)
	}
}

package agent

import "testing"

// TestSplitReasoning exercises every case the in-kernel reasoning split must handle:
// the normal open+close Ornith emission, the pre-seeded close-only template, a bare
// non-reasoning turn (must be byte-identical), an unclosed/in-progress block, and the
// whitespace/edge variants in between.
func TestSplitReasoning(t *testing.T) {
	cases := []struct {
		name          string
		in            string
		wantReasoning string
		wantContent   string
	}{
		{
			name:          "open and close",
			in:            "<think>reasoning</think>answer",
			wantReasoning: "reasoning",
			wantContent:   "answer",
		},
		{
			name:          "open and close with inner and trailing whitespace",
			in:            "<think>\nlet me think\n</think>\nthe answer\n",
			wantReasoning: "let me think",
			wantContent:   "the answer",
		},
		{
			name:          "leading whitespace before open tag",
			in:            "  \n<think>reasoning</think>answer",
			wantReasoning: "reasoning",
			wantContent:   "answer",
		},
		{
			name:          "close tag only (prompt pre-seeded open)",
			in:            "reasoning here</think>the answer",
			wantReasoning: "reasoning here",
			wantContent:   "the answer",
		},
		{
			name:          "close tag only with whitespace",
			in:            "\nreasoning here\n</think>\nthe answer",
			wantReasoning: "reasoning here",
			wantContent:   "the answer",
		},
		{
			name:          "no think tags at all returns input untouched",
			in:            "just a plain answer",
			wantReasoning: "",
			wantContent:   "just a plain answer",
		},
		{
			name:          "no think tags preserves surrounding whitespace byte-for-byte",
			in:            "  leading and trailing  ",
			wantReasoning: "",
			wantContent:   "  leading and trailing  ",
		},
		{
			name:          "unclosed think block is all reasoning, no content",
			in:            "<think>still reasoning when cut off",
			wantReasoning: "still reasoning when cut off",
			wantContent:   "",
		},
		{
			name:          "unclosed think block with only the open tag",
			in:            "<think>",
			wantReasoning: "",
			wantContent:   "",
		},
		{
			name:          "empty reasoning block",
			in:            "<think></think>answer",
			wantReasoning: "",
			wantContent:   "answer",
		},
		{
			name:          "empty string",
			in:            "",
			wantReasoning: "",
			wantContent:   "",
		},
		{
			name:          "content after close may itself contain a think tag literal",
			in:            "<think>r</think>the word <think> appears here",
			wantReasoning: "r",
			wantContent:   "the word <think> appears here",
		},
		{
			name:          "answer with embedded newlines is preserved",
			in:            "<think>r</think>line one\nline two",
			wantReasoning: "r",
			wantContent:   "line one\nline two",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotReasoning, gotContent := splitReasoning(tc.in)
			if gotReasoning != tc.wantReasoning {
				t.Errorf("reasoning = %q, want %q", gotReasoning, tc.wantReasoning)
			}
			if gotContent != tc.wantContent {
				t.Errorf("content = %q, want %q", gotContent, tc.wantContent)
			}
		})
	}
}

// TestSplitReasoningNonReasoningIsByteIdentical pins the always-on safety property:
// when there is no think block, the returned content is the EXACT input (same bytes,
// no trimming), so wiring the split into the planner cannot change a non-reasoning turn.
func TestSplitReasoningNonReasoningIsByteIdentical(t *testing.T) {
	inputs := []string{
		"",
		"hello",
		"  spaces around  ",
		"multi\nline\nanswer",
		"a <tool_call> block but no think tags",
	}
	for _, in := range inputs {
		reasoning, content := splitReasoning(in)
		if reasoning != "" {
			t.Errorf("splitReasoning(%q) reasoning = %q, want empty", in, reasoning)
		}
		if content != in {
			t.Errorf("splitReasoning(%q) content = %q, want byte-identical input", in, content)
		}
	}
}

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

// reasoningSelectorInputs is the representative input table shared by the
// named-selector tests below; it covers every branch TestSplitReasoning exercises.
var reasoningSelectorInputs = []string{
	" thinkingreasoning</think>answer",
	"reasoning here</think>the answer",
	"just a plain answer",
	"  leading and trailing  ",
	"thinking out loud with no close",
	"",
}

// TestReasoningParserSelectorQwen3Parity proves the named `qwen3` selector resolves
// and produces EXACTLY the segmentation of splitReasoning (the parity surface for
// SGLang/vLLM --reasoning-parser qwen3).
func TestReasoningParserSelectorQwen3Parity(t *testing.T) {
	p, ok := LookupReasoningParser(ReasoningParserQwen3)
	if !ok {
		t.Fatalf("LookupReasoningParser(%q) not found", ReasoningParserQwen3)
	}
	selected := SelectReasoningParser(ReasoningParserQwen3)
	for _, in := range reasoningSelectorInputs {
		wantReasoning, wantContent := splitReasoning(in)
		gotReasoning, gotContent := p(in)
		if gotReasoning != wantReasoning || gotContent != wantContent {
			t.Errorf("lookup qwen3(%q) = (%q,%q), want (%q,%q)", in, gotReasoning, gotContent, wantReasoning, wantContent)
		}
		gotReasoning, gotContent = selected(in)
		if gotReasoning != wantReasoning || gotContent != wantContent {
			t.Errorf("select qwen3(%q) = (%q,%q), want (%q,%q)", in, gotReasoning, gotContent, wantReasoning, wantContent)
		}
	}
}

// TestReasoningParserSelectorUnsetIsDefault pins that an UNSET selector falls back
// to splitReasoning byte-for-byte, so wiring selection in cannot change behavior.
func TestReasoningParserSelectorUnsetIsDefault(t *testing.T) {
	selected := SelectReasoningParser("")
	for _, in := range reasoningSelectorInputs {
		wantReasoning, wantContent := splitReasoning(in)
		gotReasoning, gotContent := selected(in)
		if gotReasoning != wantReasoning || gotContent != wantContent {
			t.Errorf("SelectReasoningParser(\"\")(%q) = (%q,%q), want (%q,%q)", in, gotReasoning, gotContent, wantReasoning, wantContent)
		}
	}
	for _, in := range []string{"just a plain answer", "  leading and trailing  ", ""} {
		_, content := SelectReasoningParser("")(in)
		if content != in {
			t.Errorf("unset selector content = %q, want byte-identical input %q", content, in)
		}
	}
}

// TestReasoningParserSelectorUnknownFallsBackToDefault documents the unknown-name
// contract: LookupReasoningParser reports not-found, and SelectReasoningParser falls
// back to the default splitReasoning.
func TestReasoningParserSelectorUnknownFallsBackToDefault(t *testing.T) {
	if _, ok := LookupReasoningParser("no-such-parser"); ok {
		t.Fatal("LookupReasoningParser(unknown) = found, want not-found")
	}
	unknown := SelectReasoningParser("no-such-parser")
	defaulted := SelectReasoningParser("")
	for _, in := range reasoningSelectorInputs {
		gotReasoning, gotContent := unknown(in)
		wantReasoning, wantContent := defaulted(in)
		if gotReasoning != wantReasoning || gotContent != wantContent {
			t.Errorf("unknown selector(%q) = (%q,%q), want default (%q,%q)", in, gotReasoning, gotContent, wantReasoning, wantContent)
		}
	}
}

// TestReasoningParserSelectorValidList ensures the registry advertises qwen3.
func TestReasoningParserSelectorValidList(t *testing.T) {
	found := false
	for _, name := range ValidReasoningParsers() {
		if name == ReasoningParserQwen3 {
			found = true
		}
	}
	if !found {
		t.Errorf("ValidReasoningParsers() = %v, want to contain %q", ValidReasoningParsers(), ReasoningParserQwen3)
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

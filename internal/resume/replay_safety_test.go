package resume

import "testing"

func TestReplaySafeEnumeratesPartialBlockKinds(t *testing.T) {
	tests := []struct {
		name   string
		blocks []EmittedBlock
		want   bool
	}{
		{name: "empty", want: true},
		{name: "thinking only", blocks: []EmittedBlock{{Kind: BlockThinking, Text: "private reasoning"}}, want: true},
		{name: "whitespace text", blocks: []EmittedBlock{{Kind: BlockText, Text: " \n\t"}}, want: true},
		{name: "thinking and whitespace", blocks: []EmittedBlock{{Kind: BlockThinking}, {Kind: BlockText, Text: "  "}}, want: true},
		{name: "visible text", blocks: []EmittedBlock{{Kind: BlockText, Text: "hello"}}},
		{name: "image", blocks: []EmittedBlock{{Kind: BlockImage}}},
		{name: "tool call", blocks: []EmittedBlock{{Kind: BlockToolCall, ToolCallID: "call-1"}}},
		{name: "server tool", blocks: []EmittedBlock{{Kind: BlockServerTool}}},
		{name: "tool result", blocks: []EmittedBlock{{Kind: BlockToolResult, ToolCallID: "call-1"}}},
		{name: "unknown fails safe", blocks: []EmittedBlock{{Kind: "future_block"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ReplaySafe(tt.blocks); got != tt.want {
				t.Fatalf("ReplaySafe() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDecidePartialRetryGatesRetryableErrorOnOutput(t *testing.T) {
	got := DecidePartialRetry(RetryableError, []EmittedBlock{{Kind: BlockToolCall, ToolCallID: "charged-card"}})
	if got.Action != PartialRetrySuppressed || got.Reason != ReasonReplayUnsafeOutput {
		t.Fatalf("decision = %#v, want observable replay-unsafe suppression", got)
	}
}

func TestDecidePartialRetryAllowsThinkingOnlyRetry(t *testing.T) {
	got := DecidePartialRetry(RetryableError, []EmittedBlock{{Kind: BlockThinking, Text: "reasoning"}})
	if got.Action != PartialRetry || got.Reason != ReasonReplaySafe {
		t.Fatalf("decision = %#v, want retry", got)
	}
}

func TestDecidePartialRetryGatesClassifierRefusal(t *testing.T) {
	unsafe := DecidePartialRetry(RetryClassifierRefusal, []EmittedBlock{{Kind: BlockText, Text: "visible"}})
	if unsafe.Action != PartialRetrySuppressed || unsafe.Reason != ReasonReplayUnsafeOutput {
		t.Fatalf("unsafe refusal decision = %#v", unsafe)
	}
	safe := DecidePartialRetry(RetryClassifierRefusal, []EmittedBlock{{Kind: BlockThinking}})
	if safe.Action != PartialRetry {
		t.Fatalf("safe refusal decision = %#v, want retry", safe)
	}
}

func TestDecidePartialRetryPreservesCompletedToolEffects(t *testing.T) {
	blocks := []EmittedBlock{
		{Kind: BlockToolCall, ToolCallID: "side-effect-1"},
		{Kind: BlockToolResult, ToolCallID: "side-effect-1"},
	}
	got := DecidePartialRetry(RetryableError, blocks)
	if got.Action != PartialPreserveContinue || got.Reason != ReasonCompletedToolEffects {
		t.Fatalf("decision = %#v, want preserve-and-continue", got)
	}
}

func TestDecidePartialRetryDoesNotReplaceErrorClassification(t *testing.T) {
	got := DecidePartialRetry(RetryNotEligible, nil)
	if got.Action != PartialRetrySuppressed || got.Reason != ReasonRetryNotEligible {
		t.Fatalf("decision = %#v, want existing classification to suppress retry", got)
	}
}

func TestHasDanglingToolCall(t *testing.T) {
	tests := []struct {
		name   string
		blocks []EmittedBlock
		want   bool
	}{
		{name: "empty"},
		{name: "text only", blocks: []EmittedBlock{{Kind: BlockText, Text: "answer"}}},
		{name: "call without result", blocks: []EmittedBlock{{Kind: BlockToolCall, ToolCallID: "c1"}}, want: true},
		{name: "call with result", blocks: []EmittedBlock{
			{Kind: BlockToolCall, ToolCallID: "c1"}, {Kind: BlockToolResult, ToolCallID: "c1"}}},
		{name: "one of two matched", blocks: []EmittedBlock{
			{Kind: BlockToolCall, ToolCallID: "c1"}, {Kind: BlockToolResult, ToolCallID: "c1"},
			{Kind: BlockToolCall, ToolCallID: "c2"}}, want: true},
		{name: "empty-id call can never match", blocks: []EmittedBlock{
			{Kind: BlockToolCall}, {Kind: BlockToolResult, ToolCallID: "c1"}}, want: true},
		{name: "dangling server tool", blocks: []EmittedBlock{{Kind: BlockServerTool, ToolCallID: "s1"}}, want: true},
		{name: "server tool with result", blocks: []EmittedBlock{
			{Kind: BlockServerTool, ToolCallID: "s1"}, {Kind: BlockToolResult, ToolCallID: "s1"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := HasDanglingToolCall(tt.blocks); got != tt.want {
				t.Fatalf("HasDanglingToolCall() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEmittedBlocksFromContent(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want []EmittedBlock
	}{
		{name: "empty", raw: ""},
		{name: "null", raw: "null"},
		{name: "bare string is text", raw: `"typed prompt"`,
			want: []EmittedBlock{{Kind: BlockText, Text: "typed prompt"}}},
		{name: "blank bare string folds to nil", raw: `"  "`},
		{name: "malformed folds to nil", raw: `{"not":"an array"`},
		{name: "vocabulary", raw: `[
			{"type":"thinking","thinking":"private"},
			{"type":"redacted_thinking"},
			{"type":"text","text":"visible"},
			{"type":"image"},
			{"type":"tool_use","id":"c1","name":"Bash"},
			{"type":"server_tool_use","id":"s1","name":"web_search"},
			{"type":"web_search_tool_result","tool_use_id":"s1"},
			{"type":"tool_result","tool_use_id":"c1"},
			{"type":"future_block"}]`,
			want: []EmittedBlock{
				{Kind: BlockThinking},
				{Kind: BlockThinking},
				{Kind: BlockText, Text: "visible"},
				{Kind: BlockImage},
				{Kind: BlockToolCall, ToolCallID: "c1"},
				{Kind: BlockServerTool, ToolCallID: "s1"},
				{Kind: BlockToolResult, ToolCallID: "s1"},
				{Kind: BlockToolResult, ToolCallID: "c1"},
				{Kind: "future_block"},
			}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EmittedBlocksFromContent([]byte(tt.raw))
			if len(got) != len(tt.want) {
				t.Fatalf("got %d blocks %#v, want %d", len(got), got, len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("block %d = %#v, want %#v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

package agent

import (
	"strings"
	"testing"
)

var recencyPagingBody = strings.Repeat("func helper() int { return 42 }\n", 640)

func recencyPagingTurn(tool string) []Message {
	return []Message{
		{Role: RoleUser, Content: "do it"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_1", Type: "function", Function: Func{Name: tool, Arguments: `{}`}}}},
		{Role: RoleTool, ToolCallID: "call_1", Content: recencyPagingBody},
	}
}

// The current turn's result (nothing after it but tool messages) is what the model just
// asked for; the pre-send pass forwards it whole even above the 4 KiB threshold.
func TestPreSendNeverPagesTrailingToolResult(t *testing.T) {
	safe, _ := QuarantineOutboundMessages(recencyPagingTurn("fetch_doc"))
	if safe[2].Content != recencyPagingBody {
		t.Fatalf("trailing tool result was rewritten: %.200s", safe[2].Content)
	}
}

// A nameless older result is classified under the tool its tool_call_id answers, so a
// 20 KiB read stays below the read-class threshold instead of the 4 KiB "other" one.
func TestPreSendResolvesNamelessResultClassFromToolCall(t *testing.T) {
	msgs := append(recencyPagingTurn("read"), Message{Role: RoleAssistant, Content: "next"})
	safe, _ := QuarantineOutboundMessages(msgs)
	if safe[2].Content != recencyPagingBody {
		t.Fatalf("older nameless read result was paged: %.200s", safe[2].Content)
	}
}

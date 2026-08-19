package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestDecodedStaleReadElisionReachesOpenAIPlannerInput(t *testing.T) {
	big := strings.Repeat("stale snapshot ", 200)
	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "inspect then edit"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "read-1", Type: "function", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"src/app.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "read-1", Content: big},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "edit-1", Type: "function", Function: agent.Func{Name: "Edit", Arguments: `{"file_path":"src/app.go","old_string":"a","new_string":"b"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "edit-1", Content: "edited"},
		{Role: agent.RoleAssistant, Content: "continuing"},
		{Role: agent.RoleUser, Content: "verify"},
		{Role: agent.RoleAssistant, Content: "working"},
		{Role: agent.RoleUser, Content: "latest"},
	}
	s := &Server{elideStaleReads: true}
	got := s.maybeElideStaleReadMessages("decoded-openai", messages)
	if got[2].Content == big || !strings.Contains(got[2].Content, "superseded") || !strings.Contains(got[2].Content, "restore_id=") {
		t.Fatalf("stale read was not replaced: %q", got[2].Content)
	}
	if messages[2].Content != big {
		t.Fatal("decoded stale-read elision mutated the caller's transcript")
	}
	id := strings.TrimSuffix(strings.Split(got[2].Content, "restore_id=")[1], "]...")
	result, err := s.restoreFromStash("decoded-openai", id)
	if err != nil || string(result.Bytes) != big {
		t.Fatalf("stale read original was not restorable: %v", err)
	}
}

func TestDecodedStaleReadElisionProtectsRecentAndUneditedReads(t *testing.T) {
	read := agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "read-1", Type: "function", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"src/app.go"}`}}}}
	result := agent.Message{Role: agent.RoleTool, ToolCallID: "read-1", Content: "snapshot"}
	edit := agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "edit-1", Type: "function", Function: agent.Func{Name: "Edit", Arguments: `{"file_path":"src/app.go"}`}}}}
	s := &Server{elideStaleReads: true}
	if got := s.maybeElideStaleReadMessages("recent", []agent.Message{read, result, edit}); got[1].Content != "snapshot" {
		t.Fatal("recent working-set read was elided")
	}
	padding := make([]agent.Message, defaultStaleProtectTail)
	if got := s.maybeElideStaleReadMessages("unedited", append([]agent.Message{read, result}, padding...)); got[1].Content != "snapshot" {
		t.Fatal("read without a later edit was elided")
	}
}

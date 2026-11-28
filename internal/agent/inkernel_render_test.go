package agent

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=5ms lane=default
func TestEnforceForcedToolChoiceUniqueCrossTurnIDs(t *testing.T) {
	t.Parallel()
	tools := []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:       "record_probe",
			Parameters: json.RawMessage(`{"type":"object","properties":{"probe":{"type":"string"}},"required":["probe"]}`),
		},
	}}
	choice := json.RawMessage(`{"type":"function","function":{"name":"record_probe"}}`)
	msgs := []Message{{Role: RoleUser, Content: "Set probe to alpha."}}

	c1 := enforceForcedToolChoice(&Completion{FinishReason: "stop"}, choice, tools, msgs)
	c2 := enforceForcedToolChoice(&Completion{FinishReason: "stop"}, choice, tools, msgs)
	if len(c1.Message.ToolCalls) != 1 || len(c2.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 forced call per turn, got %d and %d", len(c1.Message.ToolCalls), len(c2.Message.ToolCalls))
	}
	if c1.Message.ToolCalls[0].ID == c2.Message.ToolCalls[0].ID {
		t.Fatalf("forced tool call IDs collided across turns: %q", c1.Message.ToolCalls[0].ID)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestMultiTurnPrefixImmutability(t *testing.T) {
	t.Parallel()
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Read", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"}}}`)}},
		{Type: "function", Function: ToolDefFunction{Name: "Bash", Parameters: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`)}},
		{Type: "function", Function: ToolDefFunction{Name: "Write", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}`)}},
	}
	offered := NewOfferedTools(tools)

	liftWithUniqueID := func(raw string) Message {
		m := LiftTextToolCalls(Message{Role: RoleAssistant, Content: raw}, offered)
		for i := range m.ToolCalls {
			m.ToolCalls[i].ID = ""
		}
		normalizeToolCallFields(&m)
		return m
	}

	turn1Asst := liftWithUniqueID(`<tool_call>{"name":"Read","arguments":{"file_path":"a.go"}}</tool_call>`)
	turn2Asst := liftWithUniqueID(`<tool_call>{"name":"Bash","arguments":{"command":"ls"}}</tool_call>`)
	turn3Asst := liftWithUniqueID(`<tool_call>{"name":"Write","arguments":{"file_path":"b.go","content":"ok"}}</tool_call>`)

	if len(turn1Asst.ToolCalls) != 1 || len(turn2Asst.ToolCalls) != 1 || len(turn3Asst.ToolCalls) != 1 {
		t.Fatalf("failed to lift tool calls across turns")
	}

	turns := [][]Message{
		{
			{Role: RoleSystem, Content: "You are a helpful assistant."},
			{Role: RoleUser, Content: "Inspect and update."},
			turn1Asst,
			{Role: RoleTool, ToolCallID: turn1Asst.ToolCalls[0].ID, Content: "package main"},
		},
		{
			{Role: RoleSystem, Content: "You are a helpful assistant."},
			{Role: RoleUser, Content: "Inspect and update."},
			turn1Asst,
			{Role: RoleTool, ToolCallID: turn1Asst.ToolCalls[0].ID, Content: "package main"},
			turn2Asst,
			{Role: RoleTool, ToolCallID: turn2Asst.ToolCalls[0].ID, Content: "a.go"},
		},
		{
			{Role: RoleSystem, Content: "You are a helpful assistant."},
			{Role: RoleUser, Content: "Inspect and update."},
			turn1Asst,
			{Role: RoleTool, ToolCallID: turn1Asst.ToolCalls[0].ID, Content: "package main"},
			turn2Asst,
			{Role: RoleTool, ToolCallID: turn2Asst.ToolCalls[0].ID, Content: "a.go"},
			turn3Asst,
			{Role: RoleTool, ToolCallID: turn3Asst.ToolCalls[0].ID, Content: "wrote b.go"},
		},
	}

	renderers := []struct {
		name   string
		render func([]Message) []byte
	}{
		{
			name: "general_chatml",
			render: func(msgs []Message) []byte {
				return []byte(renderChatMLTools(msgs, tools))
			},
		},
		{
			name: "qwen35_ornith",
			render: func(msgs []Message) []byte {
				cfg := model.Config{ModelType: "qwen3_5", LayerTypes: []string{"linear_attention", "full_attention"}}
				return []byte(renderInKernelChatMLRequest(msgs, tools, cfg, nil, nil))
			},
		},
		{
			name: "inkling2",
			render: func(msgs []Message) []byte {
				return []byte(renderInkling2Chat(msgs, tools, SampleParams{}))
			},
		},
	}

	for _, r := range renderers {
		t.Run(r.name, func(t *testing.T) {
			rendered := make([][]byte, len(turns))
			for i, msgs := range turns {
				rendered[i] = r.render(msgs)
				s := string(rendered[i])
				for _, bad := range []string{"ReadBash", "BashRead", "ReadWrite", "WriteRead", "BashWrite", "WriteBash"} {
					if strings.Contains(s, bad) {
						t.Fatalf("turn %d rendered concatenated tool name %q in:\n%s", i+1, bad, s)
					}
				}
			}
			for i := 0; i+1 < len(rendered); i++ {
				if !bytes.HasPrefix(rendered[i+1], rendered[i]) {
					t.Fatalf("prefix immutability violated between turn %d and %d:\n--- turn %d ---\n%s\n--- turn %d ---\n%s",
						i+1, i+2, i+1, rendered[i], i+2, rendered[i+1])
				}
			}
		})
	}
}

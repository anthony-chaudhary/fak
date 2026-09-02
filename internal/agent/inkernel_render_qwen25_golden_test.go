package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestRenderTranscriptToolsQwen25Golden pins renderChatMLTools's tools-bearing output to
// the REAL Qwen2.5 chat template. The fixture
// (testdata/qwen25-coder-7b-instruct-tools-chatml.golden) was produced by applying the
// actual chat_template from Qwen/Qwen2.5-Coder-7B-Instruct's tokenizer_config.json with
// jinja2 (transformers environment: trim_blocks/lstrip_blocks, the HF tojson filter =
// json.dumps defaults) to the transcript below, with add_generation_prompt=True — the
// grammar the checkpoint was trained on and the one llama.cpp serves (issue #10600:
// rendering a different tool grammar taught the model to improvise formats its lift path
// never recognized, so native tool_calls never engaged). The fixture lives with this
// test; regenerate ground truth from the model's tokenizer_config, never from this
// renderer.
func TestRenderTranscriptToolsQwen25Golden(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "You are a coding agent inside the fak kernel. Be precise."},
		{Role: RoleUser, Content: "List the Go files in the current directory."},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{
			ID:       "call_0",
			Type:     "function",
			Function: Func{Name: "Bash", Arguments: `{"command":"ls -la"}`},
		}}},
		{Role: RoleUser, Content: "Now show the first ten lines of go.mod."},
	}
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{
			Name:        "Bash",
			Description: "Runs a shell command in the workspace and returns its output.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"command":{"type":"string","description":"The shell command to run."}},"required":["command"]}`),
		}},
		{Type: "function", Function: ToolDefFunction{
			Name:        "Read",
			Description: "Reads a file from the workspace and returns its contents.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"The file path to read."}},"required":["path"]}`),
		}},
	}
	got := renderChatMLTools(messages, tools)
	want, err := os.ReadFile(filepath.Join("testdata", "qwen25-coder-7b-instruct-tools-chatml.golden"))
	if err != nil {
		t.Fatalf("read golden fixture: %v", err)
	}
	if got != string(want) {
		t.Fatalf("rendered transcript diverges from the Qwen2.5 template render (want %d bytes, got %d):\n--- got ---\n%s\n--- want (template ground truth) ---\n%s",
			len(want), len(got), got, want)
	}
}

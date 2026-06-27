package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// TestInKernelForwardEmitsLiftableToolCall is the FAK_GGUF-gated end-to-end guard the
// seam tests cannot give. inkernel_toolcall_test.go pins the lift on SYNTHESIZED
// completions and internal/gateway/inkernel_toolcall_wire_test.go pins the
// tool_calls->stop_reason:"tool_use" wire mapping on a MOCK planner — both green without
// any weights. Neither boots the REAL weighted model, so a regression in renderChatMLTools
// or the decode path that breaks real tool emission would pass CI today (#609).
//
// This test closes that gap: it loads a real GGUF through the exact path fak serve --gguf
// uses (LoadModelQuant + the embedded tokenizer), builds the in-kernel planner directly,
// advertises one tool, and asserts the actual forward emits a <tool_call> that Complete
// lifts into a structured ToolCall with FinishReason "tool_calls" — the internal signal
// the gateway maps to the Anthropic wire's stop_reason:"tool_use" (proven manually in
// experiments/agent-live/inkernel-toolcall-witness-2026-06-24.json). The planner defaults
// to temperature 0 (argmax), so the decode is deterministic for a given GGUF.
//
// Model discovery mirrors TestGGUFChatCoherence so it skips cleanly with no weights and
// never blocks CI on a box without models:
//   - FAK_GGUF=/path/to/model.gguf  (explicit; any Qwen2.5 instruct Q8/Q4 GGUF), or
//   - the cached fixture ~/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct.Q8_0.gguf
func TestInKernelForwardEmitsLiftableToolCall(t *testing.T) {
	path := os.Getenv("FAK_GGUF")
	if path == "" {
		home, _ := os.UserHomeDir()
		cand := filepath.Join(home, ".cache", "fak-models", "gguf", "Qwen2.5-1.5B-Instruct.Q8_0.gguf")
		if _, err := os.Stat(cand); err == nil {
			path = cand
		}
	}
	if path == "" {
		t.Skip("no GGUF: set FAK_GGUF=/path/to/Qwen2.5-*.gguf (or cache " +
			"~/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct.Q8_0.gguf) to run the live in-kernel tool-call gate")
	}

	f, err := ggufload.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Skipf("%s has no embedded tokenizer", filepath.Base(path))
	}
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	m, err := ggufload.LoadModelQuant(path)
	if err != nil {
		t.Fatalf("LoadModelQuant: %v", err)
	}

	// nil backend = the CPU reference decode path (the fak serve --gguf default without
	// --backend); q4k=false matches the Q8 fixture.
	p := NewInKernelPlanner(m, tok, "Qwen2.5-1.5B-Instruct", false, nil, false)

	tools := []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:        "list_files",
			Description: "List the files in a directory on the local filesystem.",
			Parameters: json.RawMessage(
				`{"type":"object","properties":{"path":{"type":"string",` +
					`"description":"the directory to list"}},"required":["path"]}`),
		},
	}}
	messages := []Message{{Role: "user", Content: "List the files in the current directory."}}

	comp, err := p.Complete(context.Background(), messages, tools)
	if err != nil {
		t.Fatalf("Complete on the real in-kernel forward: %v", err)
	}
	if comp.ToolCallsDropped {
		t.Fatalf("the real forward emitted a <tool_call> the lift could not recover (content=%q)", comp.Message.Content)
	}
	if len(comp.Message.ToolCalls) == 0 {
		t.Fatalf("real in-kernel forward emitted no liftable tool call; FinishReason=%q content=%q",
			comp.FinishReason, comp.Message.Content)
	}
	if comp.FinishReason != "tool_calls" {
		t.Fatalf("FinishReason = %q, want tool_calls (the signal the gateway maps to wire stop_reason:tool_use)",
			comp.FinishReason)
	}
	if got := comp.Message.ToolCalls[0].Function.Name; got != "list_files" {
		t.Fatalf("lifted tool call name = %q, want list_files (content=%q)", got, comp.Message.Content)
	}
}
